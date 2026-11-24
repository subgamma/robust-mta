"""E4, Criteo controlled intervention (primary construction, K=20).

Reuses the pairwise certification machinery of `e4_confident_reversal` on real Criteo
journey structure. The materialized primary linear credits (`results/credits/linear/0.parquet`,
days 8-30, K=20, A4 cap applied) are the reference stream; its observed shares
a_j^ref, a_k^ref stand in for truth (the dataset has no other ground truth). A known
extra thinning at ratio r in {1.25, 1.5, 2.0} (`config/criteo.yaml` gamma_grid, excluding
Gamma=1) is applied on top, as in the synthetic E4.

Every primary journey touches exactly one campaign (credits are one-hot, verified at
load), so thinning at rate p is a per-touch Bernoulli(p) draw and T_ij = n_touches iff
the journey's campaign is j, else 0.

Pair (0-indexed in the top_20 list): j = 1 (campaign 15184511, a_ref = 0.1129),
k = 10 (campaign 28351001, a_ref = 0.0641), rho = a_j_ref / a_k_ref = 1.762. It lies
in the same regime window as the synthetic design (r < sqrt(rho) resumes, sqrt(rho) < r < rho
gives no reversal, r > rho reverses): sqrt(1.762) = 1.328 > 1.25, 1.328 < 1.5 < 1.762,
1.762 < 2.0. No other pair was hand-picked beyond checking this condition.

Journeys are processed in conversion-time order; only the thinning is redrawn per replication.

Multiplicity: `cs.MultiCampaignCumulativeShareCS` with K_for_beta=2 (j, k and the shared
denominator), beta = alpha/3 per process, crossing probability beta/2. The naive rule
uses raw alpha. E5-E8 use beta = alpha/(K+1) with K=20, a different denominator.

The robust rule is well-specified at Gamma_used = r for every r (the guarantee applies
only then). `population_ceiling_rho_over_r` is a population-level delta=0 quantity, not
`certify.gamma_star`.

`results[MISSPECIFIED_R]["misspecified"]` has two rows at r_true = MISSPECIFIED_R = 2.0
(asymptotic reference ratio a_k/a_j = r_true/rho = 1.1354): Gamma_used=1.25 (above 1.1354,
never crosses) and Gamma_used=1.1 (below, crosses asymptotically). Both reuse the CS
intervals of the well-specified r=2.0 row. Unlike the synthetic module, which generates
a 200,000-conversion stream for this demonstration, Criteo has a fixed 109,221 monitored
journeys; there lo_k/hi_j = 0.948, below even the 1.1 threshold, so robust_wrong_rate = 0
for both configurations within the real data's finite size.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from ..certify import Certification
from ..pair_cert import PairStream
from ..criteo import config_hash, load_config
from ..cs import MultiCampaignCumulativeShareCS, drift_test_cs, first_directional_exclusion

CREDITS_PATH = "results/credits/linear/0.parquet"
CONFIG_PATH = "config/criteo.yaml"

J_IDX, K_IDX = 1, 10

ALPHA = 0.05
DELTA = 0.005
MAX_TOUCHES = 10
B_BOUND = float(MAX_TOUCHES)
# Grid includes r=1.1 (resumption regime) and 4.0 (multi-fold stress); rho_ref**2 is added in run as a prespecified
# log-ratio-symmetric stress point, not a theoretical threshold.
R_GRID = (1.1, 1.25, 1.5, 2.0, 4.0)
N_REPS = 40
N_CHECKPOINTS = 20
SEED = 0

MISSPECIFIED_R = 2.0
MISSPECIFIED_CONFIGS = (
    (1.1, "misspecified Gamma -- guarantee not claimed"),
    (1.25, "misspecified but uncrossed"),
)


def _load_reference_journeys(credits_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (campaign_idx, n_touches, conversion_time_days) per primary monitored journey, sorted by conversion time.

    Verifies the one-hot-credit fact that the thinning shortcut depends on.
    """
    df = pl.read_parquet(credits_path)
    credit_cols = [c for c in df.columns if c.startswith("credit_")]
    credits = df.select(credit_cols).to_numpy()
    n_touches = df["n_touches"].to_numpy().astype(float)
    conv_time = df["conversion_time_days"].to_numpy()

    row_sums = credits.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        raise ValueError(
            "PRIMARY construction is expected to be single-campaign-per-journey "
            "(linear credit sums to 1 with all mass on one campaign) -- this no "
            "longer holds; the thinning shortcut in this module is invalid."
        )
    campaign_idx = np.argmax(credits, axis=1)
    order = np.argsort(conv_time, kind="stable")
    return campaign_idx[order], n_touches[order], conv_time[order]


def _reference_shares(campaign_idx: np.ndarray, n_touches: np.ndarray, n_campaigns: int) -> np.ndarray:
    mass = np.zeros(n_campaigns)
    for c, n in zip(campaign_idx, n_touches):
        mass[c] += n
    return mass / mass.sum()


def _stream_with_intervention(
    campaign_idx: np.ndarray, n_touches: np.ndarray, p_j: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-journey (T_j, T_k, T) after thinning only campaign J_IDX at rate p_j; every other campaign passes through (p=1).

    A fully dropped journey keeps its slot in the stream with zero mass, so the
    total-mass denominator stays aligned.
    """
    n = len(campaign_idx)
    Tj = np.zeros(n)
    Tk = np.zeros(n)
    T = np.zeros(n)
    is_j = campaign_idx == J_IDX
    retained_j = rng.binomial(n_touches[is_j].astype(int), p_j).astype(float)
    Tj[is_j] = retained_j
    T[is_j] = retained_j
    not_j = ~is_j
    T[not_j] = n_touches[not_j]
    Tk[campaign_idx == K_IDX] = n_touches[campaign_idx == K_IDX]
    return Tj, Tk, T


def run(
    n_reps: int = N_REPS,
    n_checkpoints: int = N_CHECKPOINTS,
    r_grid: tuple[float, ...] = R_GRID,
    seed: int = SEED,
    credits_path: str = CREDITS_PATH,
    config_path: str = CONFIG_PATH,
) -> dict:
    cfg = load_config(config_path)
    campaign_idx, n_touches, conv_time = _load_reference_journeys(credits_path)
    n_total = len(campaign_idx)
    n_campaigns = cfg["campaigns"]["k_primary"]
    top_20 = cfg["campaigns"]["top_20"]

    a_ref = _reference_shares(campaign_idx, n_touches, n_campaigns)
    a_j_ref, a_k_ref = float(a_ref[J_IDX]), float(a_ref[K_IDX])
    rho = a_j_ref / a_k_ref

    rho_ref_squared = rho ** 2
    full_r_grid = tuple(sorted(set(list(r_grid) + [rho_ref_squared])))

    checkpoints = tuple(sorted(set(np.geomspace(1, n_total, n_checkpoints).astype(int).tolist())))
    rng = np.random.default_rng(seed)
    results = {}

    for r in full_r_grid:
        p_j = 1.0 / r
        population_ceiling_rho_over_r = rho / r
        is_misspec_row = (r == MISSPECIFIED_R)
        is_rho_ref_squared = bool(abs(r - rho_ref_squared) < 1e-9)

        denom_post = p_j * a_j_ref + (1.0 - a_j_ref)
        a_j_post = p_j * a_j_ref / denom_post
        a_k_post = a_k_ref / denom_post
        expected_post_thinning_gap = a_j_post - a_k_post
        gap_below_neg_delta = bool(expected_post_thinning_gap < -DELTA)

        naive_wrong_at = {L: 0 for L in checkpoints}
        robust_correct_at = {L: 0 for L in checkpoints}
        robust_wrong_at = {L: 0 for L in checkpoints}
        robust_unresolved_at = {L: 0 for L in checkpoints}
        misspec_correct_at = {g: {L: 0 for L in checkpoints} for g, _ in MISSPECIFIED_CONFIGS}
        misspec_wrong_at = {g: {L: 0 for L in checkpoints} for g, _ in MISSPECIFIED_CONFIGS}
        misspec_unresolved_at = {g: {L: 0 for L in checkpoints} for g, _ in MISSPECIFIED_CONFIGS}
        gamma1_correct_at = {L: 0 for L in checkpoints}
        gamma1_wrong_at = {L: 0 for L in checkpoints}
        gamma1_unresolved_at = {L: 0 for L in checkpoints}

        t0 = time.time()
        checkpoint_set = set(checkpoints)
        for rep in range(n_reps):
            Tj, Tk, T = _stream_with_intervention(campaign_idx, n_touches, p_j, rng)
            D = Tj - Tk

            naive_look, naive_dir = first_directional_exclusion(D, 0.0, ALPHA, lo=-B_BOUND, hi=B_BOUND)
            for L in checkpoints:
                if naive_look is not None and naive_look <= L and naive_dir == -1:
                    naive_wrong_at[L] += 1

            # One incremental pass over the CS, no per-checkpoint replay. Full-pool estimand (T includes the other 18
            # campaigns, delta = 0.005 > 0): the production rule is direct numerator plus an explicit "other" bucket
            # (see pair_cert.py); legacy configurations use the ratio box. Same beta = alpha/3.
            ps = PairStream(ALPHA, B_BOUND, pool="full")
            n_stream = len(Tj)
            LU_at = {}
            for i in range(n_stream):
                ps.update(Tj[i], Tk[i], T[i])
                if (i + 1) in checkpoint_set:
                    LU_at[i + 1] = ps.snap()

            for L in checkpoints:
                snap = LU_at[L]
                cert = ps.rule(snap, r, DELTA)
                if cert == Certification.J_OVER_K:
                    robust_correct_at[L] += 1
                elif cert == Certification.K_OVER_J:
                    robust_wrong_at[L] += 1
                else:
                    robust_unresolved_at[L] += 1

                if is_misspec_row:
                    for g, _ in MISSPECIFIED_CONFIGS:
                        mis_cert = ps.rule(snap, g, DELTA)
                        if mis_cert == Certification.J_OVER_K:
                            misspec_correct_at[g][L] += 1
                        elif mis_cert == Certification.K_OVER_J:
                            misspec_wrong_at[g][L] += 1
                        else:
                            misspec_unresolved_at[g][L] += 1

                cert1 = ps.rule(snap, 1.0, DELTA)
                if cert1 == Certification.J_OVER_K:
                    gamma1_correct_at[L] += 1
                elif cert1 == Certification.K_OVER_J:
                    gamma1_wrong_at[L] += 1
                else:
                    gamma1_unresolved_at[L] += 1

            if (rep + 1) % 10 == 0:
                print(f"  r={r} rep {rep+1}/{n_reps} ({time.time()-t0:.1f}s)", flush=True)

        diag_rng = np.random.default_rng(seed + 9000 + int(r * 1000))
        Tj_diag, Tk_diag, T_diag = _stream_with_intervention(campaign_idx, n_touches, p_j, diag_rng)
        alpha_eff_diag = ALPHA / 2
        drift_j = drift_test_cs(Tj_diag.tolist(), T_diag.tolist(), alpha_eff_diag, B_BOUND)
        drift_k = drift_test_cs(Tk_diag.tolist(), T_diag.tolist(), alpha_eff_diag, B_BOUND)

        # Headline naive baseline: the cumulative-share CS's robust rule at Gamma=1 (see `gamma1_reference`). The
        # fixed-mean WSR result (`naive_wrong_at`) is a labelled legacy comparator.
        results[r] = {
            "a_j_ref": a_j_ref,
            "a_k_ref": a_k_ref,
            "rho_ref": float(rho),
            "gamma_used": float(r),
            "population_ceiling_rho_over_r": float(population_ceiling_rho_over_r),
            "population_ceiling_label": "population ceiling rho/r (delta=0)",
            "resumes": bool(r <= population_ceiling_rho_over_r),
            "is_rho_ref_squared_stress_point": is_rho_ref_squared,
            "a_j_post_thinning": float(a_j_post),
            "a_k_post_thinning": float(a_k_post),
            "expected_post_thinning_gap": float(expected_post_thinning_gap),
            "expected_gap_below_neg_delta": gap_below_neg_delta,
            "naive_correct_rate": {L: gamma1_correct_at[L] / n_reps for L in checkpoints},
            "naive_wrong_rate": {L: gamma1_wrong_at[L] / n_reps for L in checkpoints},
            "naive_unresolved_rate": {L: gamma1_unresolved_at[L] / n_reps for L in checkpoints},
            "naive_label": "production CS, robust rule at Gamma=1 (drift-valid)",
            "robust_correct_rate": {L: robust_correct_at[L] / n_reps for L in checkpoints},
            "robust_wrong_rate": {L: robust_wrong_at[L] / n_reps for L in checkpoints},
            "robust_unresolved_rate": {L: robust_unresolved_at[L] / n_reps for L in checkpoints},
            "wall_time_s": time.time() - t0,
            "legacy_naive_wsr": {
                "label": "LEGACY/diagnostic only -- fixed-mean betting sequence (first_directional_exclusion), "
                         "constant-share assumption NOT the production assumption on this drifting stream "
                         "(see the paper's remark on betting sequences)",
                "wrong_rate": {L: naive_wrong_at[L] / n_reps for L in checkpoints},
                "constant_share_diagnostic": {
                    "note": "drift_test_cs (Prop 14) on ONE representative intervened replication at this r "
                            "-- a spot-check of whether the legacy comparator's own constant-share assumption "
                            "holds here, not an n_reps-averaged quantity.",
                    "campaign_j_status": drift_j.status,
                    "campaign_k_status": drift_k.status,
                },
            },
        }
        if is_misspec_row:
            results[r]["misspecified"] = {
                str(g): {
                    "gamma_assumed": g,
                    "r_true": MISSPECIFIED_R,
                    "label": label,
                    "checkpoints": list(checkpoints),
                    "robust_correct_rate": {L: misspec_correct_at[g][L] / n_reps for L in checkpoints},
                    "robust_wrong_rate": {L: misspec_wrong_at[g][L] / n_reps for L in checkpoints},
                    "robust_unresolved_rate": {L: misspec_unresolved_at[g][L] / n_reps for L in checkpoints},
                }
                for g, label in MISSPECIFIED_CONFIGS
            }

    # Smallest prespecified grid point at which the naive rule certifies the reversed ordering (k>j) in >=50% of
    # replications at full N.
    last_checkpoint = checkpoints[-1]
    smallest_reversal_grid_point = None
    for r in sorted(full_r_grid):
        if results[r]["naive_wrong_rate"][last_checkpoint] >= 0.5:
            smallest_reversal_grid_point = r
            break

    # A nonzero wrong rate at any well-specified robust row (Gamma_used=r) violates the guarantee and must be
    # reported, not tuned away.
    robust_gate_violations = [
        r for r in full_r_grid
        if any(v > 0.0 for v in results[r]["robust_wrong_rate"].values())
    ]

    return {
        "construction": "primary",
        "k_primary": n_campaigns,
        "campaign_j": {"idx": J_IDX, "id": top_20[J_IDX]},
        "campaign_k": {"idx": K_IDX, "id": top_20[K_IDX]},
        "n_journeys": n_total,
        "alpha": ALPHA,
        "delta": DELTA,
        "b_bound": B_BOUND,
        "r_grid": list(full_r_grid),
        "rho_ref_squared": float(rho_ref_squared),
        "n_reps": n_reps,
        "checkpoints": list(checkpoints),
        "seed": seed,
        "config_hash": config_hash(cfg),
        "credits_path": credits_path,
        "cs_method": "howard_eb_ratio",
        "cert_method": PairStream(ALPHA, B_BOUND, pool="full").label, "boundary": PairStream(ALPHA, B_BOUND, pool="full").boundary, "pool": "full",
        "beta": ALPHA / 3.0, "alpha_used": ALPHA / 3.0 / 2.0, "cs_running_intersection": False,
        "results": results,
        "smallest_prespecified_grid_point_with_naive_reversal_ge_50pct": smallest_reversal_grid_point,
        "robust_gate_violations": robust_gate_violations,
    }


def write_outputs(result: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e4_criteo.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    result = run()
    for r, res in result["results"].items():
        print(f"r={r} (Gamma_used={res['gamma_used']}, well-specified): rho_ref={res['rho_ref']:.3f} "
              f"population_ceiling_rho/r={res['population_ceiling_rho_over_r']:.3f} resumes={res['resumes']}")
        if "misspecified" in res:
            for m in res["misspecified"].values():
                last_L = result["checkpoints"][-1]
                wr = m["robust_wrong_rate"][last_L]
                print(f"  misspecified: Gamma_assumed={m['gamma_assumed']} while r_true={m['r_true']} -- "
                      f"{m['label']} (robust_wrong_rate@N={last_L}: {wr:.3f})")
    write_outputs(result, "results/e4")
