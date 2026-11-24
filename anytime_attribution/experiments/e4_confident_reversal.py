"""E4: confident reversal (synthetic, verification only).

Two campaigns j (target) and k (competitor) with true A_j/A_k = 1.75 (A_j=0.35,
A_k=0.20); j is under-observed at ratio r in {1.25, 1.5, 2.0} (p_j = p_k/r). The
observed ratio a_j/a_k = (A_j/A_k)/r falls into three regimes:

  r=1.25: a_j/a_k = 1.40 > r. The robust rule (Gamma_used=r) can eventually
          certify j>k correctly (r <= Gamma*).
  r=1.5:  a_j/a_k = 1.167 < r. No reversal, but too little margin for the robust
          rule at Gamma_used=1.5 (observability-limited, r > Gamma*).
  r=2.0:  a_j/a_k = 0.875 < 1. A real reversal: the naive rule eventually
          certifies k>j with high probability, which is wrong since A_j > A_k.
          The robust rule (Gamma_used=2.0) abstains in both directions.

Naive rule: `certify.robust_rule` at Gamma=1 on the same cumulative-share CS used
by the robust rows and by `e4_criteo_intervention.py`. The fixed-mean betting result
(`first_directional_exclusion`) is kept only as `legacy_naive_wsr`. Robust rule:
`certify.robust_rule` on the same per-campaign intervals (`_cs_history_pair`,
`cs.MultiCampaignCumulativeShareCS`), well-specified at every r in R_GRID
(Gamma_used = r). The guarantee robust_wrong_rate = 0 holds only when
Gamma_used >= the true worst-case retention ratio.

`population_ceiling_rho_over_r` (rho = A_j/A_k before intervention) is a
population-level delta=0 quantity, not the bisection estimate `certify.gamma_star`.

`result["misspecified_r_true_3"]` holds four rows at r_true=MISSPECIFIED_R_TRUE=3.0
with Gamma_used in MISSPECIFIED_GAMMA_USED_GRID={1.1,1.5,2.0,3.0}; only 3.0 is
well-specified. Population quantities (true gap, post-thinning observed gap,
population robust lower gap per Gamma_used) are computed before any simulation
(`_misspec_population_quantities`). The misspecified demonstration uses its own
larger stream (`MISSPEC_N_LOOKS`=200,000 cap, `MISSPEC_N_REPS`=15,
`MISSPEC_CHECKPOINTS`); the main R_GRID rows use N_LOOKS=20,000.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..certify import Certification, robust_lower_gap, robust_rule
from ..pair_cert import PairStream
from ..cs import MultiCampaignCumulativeShareCS, first_directional_exclusion
from ..estimands import campaign_masses
from ..journeys import apply_cap
from ..synth import generate_synthetic_journeys

K = 2
J, KK = 0, 1
A_TRUE = np.array([0.35, 0.20])
P_K = 0.5
R_GRID = (1.25, 1.5, 2.0)
ALPHA = 0.05
DELTA = 0.0
MAX_TOUCHES = 10
B_BOUND = float(MAX_TOUCHES)
N_REPS = 40
# 20,000 looks give margin for the r=1.25 case; at N=500 the per-campaign CS is too wide to resolve at any r.
N_LOOKS = 20_000
SEED = 0

CHECKPOINTS = tuple(sorted(set(np.geomspace(1, N_LOOKS, 20).astype(int).tolist())))

# Misspecification demonstration: r_true=3.0 is a population reversal; Gamma_used sweeps {1.1, 1.5, 2.0, 3.0} so only
# 3.0 is well-specified.
MISSPECIFIED_R_TRUE = 3.0
MISSPECIFIED_GAMMA_USED_GRID = (1.1, 1.5, 2.0, 3.0)
MISSPEC_N_LOOKS = 200_000
MISSPEC_N_REPS = 15
MISSPEC_CHECKPOINTS = tuple(sorted(set(np.geomspace(1000, MISSPEC_N_LOOKS, 15).astype(int).tolist())))


def _weights_for_A(A: np.ndarray) -> np.ndarray:
    # Weights equal the true aggregate shares because true_aggregate_share(w) = w/sum(w).
    return A


def _stream_Tj_T(n: int, p_j: float, p_k: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = _weights_for_A(A_TRUE)
    journeys = generate_synthetic_journeys(n, weights, rng, window_days=7.0)
    journeys = [apply_cap(j, lookback_days=7.0, max_touches=MAX_TOUCHES) for j in journeys]
    p = {J: p_j, KK: p_k}
    from ..journeys import thin

    Tj = np.zeros(n)
    Tk = np.zeros(n)
    T = np.zeros(n)
    for i, journey in enumerate(journeys):
        obs = thin(journey, p, rng)
        if obs is None:
            continue
        m = campaign_masses(obs, K)
        Tj[i], Tk[i] = m[J], m[KK]
        T[i] = m.sum()
    return Tj, Tk, T


def _cs_history_pair(Tj: np.ndarray, Tk: np.ndarray, T: np.ndarray, checkpoints) -> dict:
    """One incremental pass through the pairwise stream (`pair_cert.PairStream`, K=2, beta=alpha/3, T = T_j + T_k).

    Returns {checkpoint: snapshot}; certify with `PairStream.rule`. A checkpoint
    beyond the stream length gets the final snapshot.
    """
    ps = PairStream(ALPHA, B_BOUND, pool="tracked")
    n_stream = len(Tj)
    checkpoint_set = {min(int(c), n_stream) for c in checkpoints}
    out = {}
    for i in range(n_stream):
        ps.update(Tj[i], Tk[i], T[i])
        if (i + 1) in checkpoint_set:
            out[i + 1] = ps.snap()
    final = out[n_stream] if n_stream in out else ps.snap()
    snaps = {c: out.get(min(int(c), n_stream), final) for c in checkpoints}
    return snaps, ps


def _misspec_population_quantities(r_true: float, gamma_used_grid: tuple[float, ...]) -> dict:
    """Population-level quantities computed before any simulation.

    The full-observability gap, the post-thinning observed gap at r_true, and per
    Gamma_used the population robust lower gap (at the degenerate box
    L=U=[a_j_true, a_k_true], delta=0) and whether it exceeds -delta.
    """
    p_j = P_K / r_true
    p_k = P_K
    a_j_true = p_j * A_TRUE[J] / (p_j * A_TRUE[J] + p_k * A_TRUE[KK])
    a_k_true = 1 - a_j_true
    true_full_obs_gap = float(A_TRUE[J] - A_TRUE[KK])
    population_post_thinning_gap = float(a_j_true - a_k_true)

    # a_j_true < a_k_true at r_true=3, so the j-over-k lower gap is negative at every Gamma_used; the k-over-j lower
    # gap decides whether a wrong (k>j) certification is possible.
    per_gamma = {}
    for g in gamma_used_grid:
        gap_j_over_k = float(robust_lower_gap([a_j_true, a_k_true], [a_j_true, a_k_true], g, J, KK))
        gap_k_over_j = float(robust_lower_gap([a_j_true, a_k_true], [a_j_true, a_k_true], g, KK, J))
        population_cert = robust_rule([a_j_true, a_k_true], [a_j_true, a_k_true], g, DELTA, J, KK)
        per_gamma[g] = {
            "gamma_used": g,
            "population_robust_lower_gap_j_over_k": gap_j_over_k,
            "population_robust_lower_gap_k_over_j": gap_k_over_j,
            "exceeds_neg_delta": bool(gap_k_over_j > -DELTA),
            "population_certification": population_cert.value,
        }
    return {
        "r_true": r_true,
        "a_j_true": float(a_j_true),
        "a_k_true": float(a_k_true),
        "true_full_observability_gap": true_full_obs_gap,
        "population_post_thinning_observed_gap": population_post_thinning_gap,
        "per_gamma_used": per_gamma,
    }


def _run_misspecified_demo(
    r_true: float = MISSPECIFIED_R_TRUE,
    gamma_used_grid: tuple[float, ...] = MISSPECIFIED_GAMMA_USED_GRID,
    n_looks: int = MISSPEC_N_LOOKS,
    n_reps: int = MISSPEC_N_REPS,
    checkpoints: tuple[int, ...] = MISSPEC_CHECKPOINTS,
    seed: int = SEED + 1,
) -> dict:
    """r_true=3.0 (a population reversal) with Gamma_used swept over {1.1,1.5,2.0,3.0}; only 3.0 is well-specified.

    Adds simulated correct/wrong/unresolved trajectories to the population
    quantities from `_misspec_population_quantities`.
    """
    population = _misspec_population_quantities(r_true, gamma_used_grid)
    rng = np.random.default_rng(seed)
    p_j = P_K / r_true
    p_k = P_K

    correct_at = {g: {L: 0 for L in checkpoints} for g in gamma_used_grid}
    wrong_at = {g: {L: 0 for L in checkpoints} for g in gamma_used_grid}
    unresolved_at = {g: {L: 0 for L in checkpoints} for g in gamma_used_grid}

    t0 = time.time()
    for rep in range(n_reps):
        Tj, Tk, T = _stream_Tj_T(n_looks, p_j, p_k, rng)
        LU_at, ps = _cs_history_pair(Tj, Tk, T, checkpoints)
        for L in checkpoints:
            for g in gamma_used_grid:
                cert = ps.rule(LU_at[L], g, DELTA)
                if cert == Certification.J_OVER_K:
                    correct_at[g][L] += 1
                elif cert == Certification.K_OVER_J:
                    wrong_at[g][L] += 1
                else:
                    unresolved_at[g][L] += 1

    return {
        "population": population,
        "by_gamma_used": {
            str(g): {
                "gamma_assumed": g,
                "r_true": r_true,
                "well_specified": bool(g >= r_true),
                "label": "well-specified" if g >= r_true else "misspecified Gamma -- guarantee not claimed",
                "n_looks": n_looks,
                "n_reps": n_reps,
                "checkpoints": list(checkpoints),
                "robust_correct_rate": {L: correct_at[g][L] / n_reps for L in checkpoints},
                "robust_wrong_rate": {L: wrong_at[g][L] / n_reps for L in checkpoints},
                "robust_unresolved_rate": {L: unresolved_at[g][L] / n_reps for L in checkpoints},
            }
            for g in gamma_used_grid
        },
    }


def run(
    n_reps: int = N_REPS,
    n_looks: int = N_LOOKS,
    seed: int = SEED,
    misspec_n_looks: int = MISSPEC_N_LOOKS,
    misspec_n_reps: int = MISSPEC_N_REPS,
    misspec_checkpoints: tuple[int, ...] = MISSPEC_CHECKPOINTS,
) -> dict:
    rng = np.random.default_rng(seed)
    results = {}

    for r in R_GRID:
        p_j = P_K / r
        p_k = P_K
        a_j_true = p_j * A_TRUE[J] / (p_j * A_TRUE[J] + p_k * A_TRUE[KK])
        a_k_true = 1 - a_j_true
        population_ceiling_rho_over_r = a_j_true / a_k_true

        robust_correct_at = {L: 0 for L in CHECKPOINTS}
        robust_wrong_at = {L: 0 for L in CHECKPOINTS}
        robust_unresolved_at = {L: 0 for L in CHECKPOINTS}
        naive_correct_at = {L: 0 for L in CHECKPOINTS}
        naive_wrong_at = {L: 0 for L in CHECKPOINTS}
        naive_unresolved_at = {L: 0 for L in CHECKPOINTS}
        legacy_naive_wrong_at = {L: 0 for L in CHECKPOINTS}

        t0 = time.time()
        for rep in range(n_reps):
            Tj, Tk, T = _stream_Tj_T(n_looks, p_j, p_k, rng)
            D = Tj - Tk

            naive_look, naive_dir = first_directional_exclusion(D, 0.0, ALPHA, lo=-B_BOUND, hi=B_BOUND)
            for L in CHECKPOINTS:
                if naive_look is not None and naive_look <= L and naive_dir == -1:
                    legacy_naive_wrong_at[L] += 1

            LU_at, ps = _cs_history_pair(Tj, Tk, T, CHECKPOINTS)
            for L in CHECKPOINTS:
                snap = LU_at[L]
                cert = ps.rule(snap, r, DELTA)
                if cert == Certification.J_OVER_K:
                    robust_correct_at[L] += 1
                elif cert == Certification.K_OVER_J:
                    robust_wrong_at[L] += 1
                else:
                    robust_unresolved_at[L] += 1

                cert1 = ps.rule(snap, 1.0, DELTA)
                if cert1 == Certification.J_OVER_K:
                    naive_correct_at[L] += 1
                elif cert1 == Certification.K_OVER_J:
                    naive_wrong_at[L] += 1
                else:
                    naive_unresolved_at[L] += 1

            if (rep + 1) % 20 == 0:
                print(f"  r={r} rep {rep+1}/{n_reps} ({time.time()-t0:.1f}s)", flush=True)

        results[r] = {
            "a_j_true": float(a_j_true),
            "a_k_true": float(a_k_true),
            "gamma_used": float(r),
            "population_ceiling_rho_over_r": float(population_ceiling_rho_over_r),
            "population_ceiling_label": "population ceiling rho/r (delta=0)",
            # bool(...) because np.bool_ is not JSON-serializable and default=str would turn it into the truthy string
            # "False".
            "resumes": bool(r <= population_ceiling_rho_over_r),
            "naive_correct_rate": {L: naive_correct_at[L] / n_reps for L in CHECKPOINTS},
            "naive_wrong_rate": {L: naive_wrong_at[L] / n_reps for L in CHECKPOINTS},
            "naive_unresolved_rate": {L: naive_unresolved_at[L] / n_reps for L in CHECKPOINTS},
            "naive_label": "production CS, robust rule at Gamma=1 (drift-valid) -- "
                           "SAME definition and pairwise beta=alpha/3 convention as e4_criteo_intervention.py",
            "robust_correct_rate": {L: robust_correct_at[L] / n_reps for L in CHECKPOINTS},
            "robust_wrong_rate": {L: robust_wrong_at[L] / n_reps for L in CHECKPOINTS},
            "robust_unresolved_rate": {L: robust_unresolved_at[L] / n_reps for L in CHECKPOINTS},
            "wall_time_s": time.time() - t0,
            "legacy_naive_wsr": {
                "label": "LEGACY/diagnostic only -- fixed-mean betting sequence (first_directional_exclusion), "
                         "NOT the production naive definition (see the paper's remark on betting sequences)",
                "wrong_rate": {L: legacy_naive_wrong_at[L] / n_reps for L in CHECKPOINTS},
            },
        }

    print("  running misspecified-Gamma demonstration (r_true=3.0, separate larger stream)...", flush=True)
    misspecified = _run_misspecified_demo(
        n_looks=misspec_n_looks, n_reps=misspec_n_reps, checkpoints=misspec_checkpoints, seed=seed + 1)

    return {
        "A_true": A_TRUE.tolist(),
        "p_k": P_K,
        "r_grid": list(R_GRID),
        "alpha": ALPHA,
        "delta": DELTA,
        "n_reps": n_reps,
        "n_looks": n_looks,
        "checkpoints": list(CHECKPOINTS),
        "seed": seed,
        "results": results,
        "misspecified_r_true_3": misspecified,
        "cs_method": "howard_eb_ratio",
        "cert_method": PairStream(ALPHA, B_BOUND, pool="tracked").label, "boundary": PairStream(ALPHA, B_BOUND, pool="tracked").boundary, "pool": "tracked",
        "beta": ALPHA / 3.0, "alpha_used": ALPHA / 3.0 / 2.0, "cs_running_intersection": False,
        "pairwise_multiplicity_convention": "K_for_beta=2 (this pair + shared denominator), "
                                             "beta=alpha/3, SAME as e4_criteo_intervention.py",
    }


def write_outputs(result: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e4_synthetic.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    result = run()
    for r, res in result["results"].items():
        print(f"r={r} (Gamma_used={res['gamma_used']}, well-specified): a_j/a_k={res['a_j_true']/res['a_k_true']:.3f} "
              f"population_ceiling_rho/r={res['population_ceiling_rho_over_r']:.3f} resumes={res['resumes']}")
    ms = result["misspecified_r_true_3"]
    pop = ms["population"]
    print(f"\nmisspecified demo, r_true={pop['r_true']}: a_j_true={pop['a_j_true']:.4f} a_k_true={pop['a_k_true']:.4f} "
          f"true_full_obs_gap={pop['true_full_observability_gap']:.4f} "
          f"population_post_thinning_gap={pop['population_post_thinning_observed_gap']:.4f}")
    for g, pg in pop["per_gamma_used"].items():
        print(f"  Gamma_used={g}: gap(j>k)={pg['population_robust_lower_gap_j_over_k']:.4f} "
              f"gap(k>j)={pg['population_robust_lower_gap_k_over_j']:.4f} "
              f"pop_cert={pg['population_certification']} exceeds_neg_delta(k>j)={pg['exceeds_neg_delta']}")
    for g, m in ms["by_gamma_used"].items():
        wr = m["robust_wrong_rate"][m["checkpoints"][-1]]
        cr = m["robust_correct_rate"][m["checkpoints"][-1]]
        ur = m["robust_unresolved_rate"][m["checkpoints"][-1]]
        print(f"  Gamma_used={g} ({m['label']}): correct={cr:.3f} wrong={wr:.3f} unresolved={ur:.3f} @N={m['checkpoints'][-1]}")
    write_outputs(result, "results/e4")
