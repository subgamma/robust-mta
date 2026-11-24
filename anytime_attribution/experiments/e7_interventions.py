"""E7: controlled observability interventions (primary, K=20, linear model).

The linear model is the representative guaranteed model: the interventions concern
between-journey observability, which does not depend on the additive model, and each
model has its own aggregate-share scale, so G1-G3 are not repeated per model.

G1 (uniform loss): retention p=1-rho on every campaign, rho in {0.1,...,0.7}. The true
observed ratio a_j/a_k is unchanged in expectation, so robust_wrong_rate should stay 0
(Gamma_true=1); this tests whether certification remains valid as conversions accumulate
more slowly, not whether an ordering can be reversed.

G2 (differential loss): each replication splits the K=20 campaigns into two random
groups; the low-observability group gets p_lo=BASE_P/r and the other keeps p_hi=BASE_P,
for r in {1.25, 1.5, 2.0}. This is the E4 mechanism applied graph-wide with random
group assignment.

G3 (identity fragmentation, out of model): stylized pre-cut orphaning. With probability
rho a conversion's journey is cut at a position uniform over its n post-touchpoint gaps
(including the gap before the conversion); touchpoints up to the cut are orphaned, later
ones keep their original weight, and a fully orphaned conversion stays in the stream as a
zero-mass row. One row per conversion in the original order preserves the arrival clock;
`_stream_g3_legacy_duplicating` (one conversion split across two rows, which breaks the
clock) is kept for comparison only. Retention depends on position, violating the
marginal-retention assumption, so no coverage guarantee is claimed and a favourable DRR is
not evidence of validity.

Reference ranking R_0: the certified graph from the un-intervened day-30 linear data at
Gamma=REFERENCE_GAMMA=1.25, delta=0.005 (config). Every intervention is compared with R_0
using the same Gamma=1.25 for the robust rule. This is not re-tuned per r, so G2 acts as a
mild-to-severe misspecification stress rather than a matched Gamma_used=r trial as in E4.
The naive rule is `robust_rule` at Gamma=1.0, applied graph-wide.

DRR(rho) = |R_0 cap R_rho| / |R_0|: same-direction retention only; a pair that flips
direction is not retained.

Diagnostics added to the interventions:

(a) G2 cross-check: for every R_0 pair and G2 replication, record whether the realized
assignment put the stronger campaign (R_0's certified winner) in the LOW group and
the weaker in the HIGH group, and whether e6's Gamma*_jk is below r. Every R_0 pair is
certified at Gamma=1.25, so Gamma*_jk >= 1.25 by monotonicity (`certify.gamma_star`) and r=1.25
can never satisfy Gamma*_jk < r; for r=1.5 and 2.0 the risk depends on e6's values.

(b) G2_targeted: a deliberate, non-random variant that assigns e6's least-robust
linear-certified pair (over all certified pairs, not just R_0) to opposite groups. It
tests whether the null result of random assignment means no risky pair was drawn, or that
the mechanism cannot produce a reversal.

Gamma*_jk is a margin- and resolution-limited certification ceiling at the configured
delta (close pairs can have Gamma* barely above 1 from CS half-widths alone), not the
population sign-reversal threshold. For G2 the population threshold is r = rho_ref_jk:
the post-thinning ratio is rho_ref_jk * p_lo/p_hi = rho_ref_jk/r, independent of every
other campaign's assignment. The cross-check and G2_targeted therefore use
`sign_reversal_possible` (risky assignment and r > rho_ref_jk) and
`margin_reversal_possible` (expected post-thinning gap, K-way normalised using the full
realized assignment, below -delta); see `_g2_cross_check_records`. G2_targeted runs at
r_low=0.9*rho_ref_jk (no sign reversal predicted) and r_high=1.3*rho_ref_jk (sign reversal
predicted), both with Gamma_used=r.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from ..certify import Certification, active_cert_method, box_gamma_star, box_rule, snapshot
from ..criteo import load_config
from ..cs import MultiCampaignCumulativeShareCS
from ._primary_journeys import build_and_cache_primary_journeys

CONFIG_PATH = "config/criteo.yaml"
E6_RESULTS_PATH = "results/e6/e6_results.json"
CS_METHOD = "howard_eb_ratio"  
REFERENCE_GAMMA = 1.25
NAIVE_GAMMA = 1.0
G1_RHO_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
G2_R_GRID = (1.25, 1.5, 2.0)
G2_BASE_P = 0.5
G3_RHO_GRID = (0.2, 0.4, 0.6)
N_REPS = 3
N_REPS_TARGETED = 10
SEED = 0
GAMMA_STAR_TOL = 1e-4
G2_TARGETED_R_LOW_MULT = 0.9   # predicts NO population sign reversal (r < rho_ref)
G2_TARGETED_R_HIGH_MULT = 1.3  # predicts a population sign reversal (r > rho_ref)


def _load_e6_meta(e6_results_path: str = E6_RESULTS_PATH) -> dict:
    with open(e6_results_path) as f:
        e6 = json.load(f)
    return {"cert_method": e6.get("cert_method"), "boundary": e6.get("boundary")}


def _load_e6_linear_gamma_star(e6_results_path: str = E6_RESULTS_PATH) -> dict[tuple[int, int], float]:
    """{(j,k): Gamma*_jk} for j<k, linear model, from e6's per-pair distribution; pairs not certified at Gamma=1 are
    dropped.
    """
    with open(e6_results_path) as f:
        e6 = json.load(f)
    gstars = e6["guaranteed"]["linear"]["gamma_star_by_pair"]
    out = {}
    for key, g in gstars.items():
        if g is None:
            continue
        j, k = (int(x) for x in key.split("-"))
        out[(j, k)] = g
    return out


def _campaign_idx_and_n_touches(journeys):
    campaign_idx = np.array([j.touches[0].campaign for j in journeys])
    n_touches = np.array([j.n_touches for j in journeys], dtype=int)
    return campaign_idx, n_touches


def _stream_uniform(campaign_idx, n_touches, p, rng, K):
    retained = rng.binomial(n_touches, p).astype(float)
    Tj = np.zeros((len(campaign_idx), K))
    Tj[np.arange(len(campaign_idx)), campaign_idx] = retained
    return Tj


def _stream_g2(campaign_idx, n_touches, low_group: np.ndarray, p_hi: float, p_lo: float, rng, K):
    is_low = low_group[campaign_idx]
    p_per_journey = np.where(is_low, p_lo, p_hi)
    retained = rng.binomial(n_touches, p_per_journey).astype(float)
    Tj = np.zeros((len(campaign_idx), K))
    Tj[np.arange(len(campaign_idx)), campaign_idx] = retained
    return Tj


def _stream_g3_legacy_duplicating(campaign_idx, n_touches, rho, rng, K):
    """Legacy G3 for the structural comparison and regression test only; not used by E7.

    Splits each journey into two stream rows with probability rho at a random touch cut.
    It preserves all touchpoints and campaign mass but violates the one-row-per-conversion
    arrival clock and can narrow the confidence intervals.
    """
    n = len(campaign_idx)
    split = rng.random(n) < rho
    n_touches = n_touches.copy()
    cuts = np.zeros(n, dtype=int)
    can_split = split & (n_touches >= 2)
    for i in np.nonzero(can_split)[0]:
        cuts[i] = rng.integers(1, n_touches[i])
    frag1 = np.where(can_split, cuts, n_touches)
    frag2 = np.where(can_split, n_touches - cuts, 0)

    rows_c = np.concatenate([campaign_idx, campaign_idx[can_split]])
    rows_n = np.concatenate([frag1, frag2[can_split]])
    Tj = np.zeros((len(rows_c), K))
    Tj[np.arange(len(rows_c)), rows_c] = rows_n.astype(float)
    return Tj


def g3_orphan_cut(n_touches, rho, rng):
    """Cut positions of the corrected G3, one per conversion.

    A journey of n >= 1 touchpoints has n gaps after the touchpoints (the last is the gap
    before the conversion). With probability rho a cut falls uniformly in {1,...,n}; a cut
    after touchpoint k discards touchpoints 1..k. Returns k_i in {0,...,n_i} (0 = no cut).
    Draws are made for every conversion in fixed order (one uniform for whether to cut,
    one for where), so a seed fixes the output regardless of rho on the where-draw.
    """
    n_touches = np.asarray(n_touches, dtype=int)
    if np.any(n_touches < 1):
        raise ValueError("every latent journey must have n >= 1 touchpoints")
    cut_here = rng.random(len(n_touches)) < rho
    where = np.floor(rng.random(len(n_touches)) * n_touches).astype(int) + 1  # uniform on {1..n}
    where = np.minimum(where, n_touches)  # guard the measure-zero u == 1.0 edge
    return np.where(cut_here, where, 0)


def _stream_g3(campaign_idx, n_touches, rho, rng, K):
    """Corrected G3: stylized pre-cut orphaning, out of model.

    Each latent conversion i (n_i >= 1 touchpoints on campaign c_i, unit linear weights,
    so S_i = n_i <= B) is cut with probability rho at a position uniform over its n_i
    post-touchpoint gaps (`g3_orphan_cut`). Touchpoints 1..k are orphaned and the other
    n_i - k keep weight 1 (nothing is rescaled). One row per conversion is returned in the
    original order; a fully orphaned conversion (k = n_i, including every cut n_i = 1
    journey) stays as an all-zero row, preserving the arrival clock. Observed mass
    T_ij = n_i - k_i on campaign c_i lies in [0, B]; nothing is clipped. Retention depends
    on touchpoint position, so no coverage claim is made.
    """
    campaign_idx = np.asarray(campaign_idx)
    n_touches = np.asarray(n_touches, dtype=int)
    k = g3_orphan_cut(n_touches, rho, rng)
    kept = (n_touches - k).astype(float)
    Tj = np.zeros((len(campaign_idx), K))
    Tj[np.arange(len(campaign_idx)), campaign_idx] = kept
    return Tj


def _certified_graph(box, Gamma, delta, K):
    graph = {}
    for j, k in combinations(range(K), 2):
        cert = box_rule(box, Gamma, delta, j, k)   # production hybrid unless inside cert_method_override
        if cert != Certification.UNRESOLVED:
            graph[(j, k)] = cert
    return graph


def _drr(R0: dict, R: dict) -> float:
    if not R0:
        return float("nan")
    retained = sum(1 for pair, cert in R0.items() if R.get(pair) == cert)
    return retained / len(R0)


def _wrong_rate(R0: dict, R: dict) -> float:
    """Fraction of R_0 pairs where R certifies the opposite direction."""
    if not R0:
        return float("nan")
    opposite = {Certification.J_OVER_K: Certification.K_OVER_J, Certification.K_OVER_J: Certification.J_OVER_K}
    wrong = sum(1 for pair, cert in R0.items() if R.get(pair) == opposite[cert])
    return wrong / len(R0)


def _expected_post_thinning_gap(
    S_raw: np.ndarray, low_group: np.ndarray, p_hi: float, p_lo: float, stronger: int, weaker: int,
) -> float:
    """Population expected post-thinning gap a_stronger - a_weaker under G2, K-way normalised using the full realized
    group assignment.

    This does not cancel as the pairwise ratio does, so it is computed directly.
    """
    p_m = np.where(low_group, p_lo, p_hi)
    D_post = float((S_raw * p_m).sum())
    if D_post <= 0:
        return 0.0
    a_post = S_raw * p_m / D_post
    return float(a_post[stronger] - a_post[weaker])


def _run_targeted_pair_settings(
    stronger: int, weaker: int, r_values: list[tuple[str, float]],
    campaign_idx, n_touches, n_total: int, K: int, alpha: float, B: float, delta: float,
    S_raw: np.ndarray, rng: np.random.Generator, n_reps: int, rho_ref: float = None,
) -> dict:
    """Generalizes G2_targeted to an arbitrary (stronger, weaker) pair and list of (label, r) settings.

    Also used for the E4 pair (1,10) in addition to e6's least-robust pair.
    """
    settings = {}
    for label, r_t in r_values:
        low_group = np.zeros(K, dtype=bool)
        low_group[stronger] = True
        p_lo_t = G2_BASE_P / r_t
        expected_gap = _expected_post_thinning_gap(S_raw, low_group, G2_BASE_P, p_lo_t, stronger, weaker)
        crosses_margin = expected_gap < -delta
        population_post_thinning_ratio = (rho_ref / r_t) if rho_ref else None
        sign_reversal_possible = bool(rho_ref is not None and r_t > rho_ref)

        naive_wr_t, naive_cr_t, naive_ur_t = [], [], []
        robust_wr_t, robust_cr_t, robust_ur_t = [], [], []
        for _ in range(n_reps):
            Tj = _stream_g2(campaign_idx, n_touches, low_group, G2_BASE_P, p_lo_t, rng, K)
            T = Tj.sum(axis=1)
            mg = MultiCampaignCumulativeShareCS(K, alpha, B)
            for i in range(n_total):
                mg.update(Tj[i], T[i])
            box = snapshot(mg)
            naive_cert = box_rule(box, NAIVE_GAMMA, delta, stronger, weaker)
            robust_cert = box_rule(box, r_t, delta, stronger, weaker)
            naive_wr_t.append(1.0 if naive_cert == Certification.K_OVER_J else 0.0)
            naive_cr_t.append(1.0 if naive_cert == Certification.J_OVER_K else 0.0)
            naive_ur_t.append(1.0 if naive_cert == Certification.UNRESOLVED else 0.0)
            robust_wr_t.append(1.0 if robust_cert == Certification.K_OVER_J else 0.0)
            robust_cr_t.append(1.0 if robust_cert == Certification.J_OVER_K else 0.0)
            robust_ur_t.append(1.0 if robust_cert == Certification.UNRESOLVED else 0.0)

        settings[label] = {
            "r": r_t, "gamma_used": r_t,
            "population_post_thinning_ratio": population_post_thinning_ratio,
            "sign_reversal_possible": sign_reversal_possible,
            "expected_post_thinning_gap": expected_gap,
            "margin_reversal_possible": bool(crosses_margin),
            "crosses_margin_delta": bool(crosses_margin),
            "n_reps": n_reps,
            "naive_correct_rate": float(np.mean(naive_cr_t)),
            "naive_wrong_rate": float(np.mean(naive_wr_t)),
            "naive_unresolved_rate": float(np.mean(naive_ur_t)),
            "robust_correct_rate": float(np.mean(robust_cr_t)),
            "robust_wrong_rate": float(np.mean(robust_wr_t)),
            "robust_unresolved_rate": float(np.mean(robust_ur_t)),
        }
    return settings


def _g2_cross_check_records(
    R0: dict, low_group: np.ndarray, r: float, R_naive: dict,
    a_hat_full: np.ndarray, S_raw: np.ndarray, delta: float,
) -> list[dict]:
    """Per-pair records for the G2 cross-check.

    For a pair whose stronger reference campaign (R_0's certified winner) is in the LOW
    group while the weaker is in the HIGH group ("risky_assignment"):

    - `sign_reversal_possible`: the population post-thinning ratio
      a_j_post/a_k_post = rho_ref_jk * (p_lo/p_hi) = rho_ref_jk / r is below 1, i.e.
      r > rho_ref_jk. The shared K-way denominator cancels, so this depends only on r and
      rho_ref_jk.
    - `margin_reversal_possible`: the expected post-thinning gap, computed from the realized
      low_group assignment of all K campaigns (S_raw[m]*p_m(rep) normalised by
      D_post = sum_m S_raw[m]*p_m(rep); this does not cancel), is < -delta. This is the
      necessary condition for a wrong certification at margin delta.

    `naively_wrong`: whether the naive rule (Gamma=1) actually certified the opposite
    direction for this pair in this finite-N replication.
    """
    opposite = {Certification.J_OVER_K: Certification.K_OVER_J, Certification.K_OVER_J: Certification.J_OVER_K}
    p_hi = G2_BASE_P
    p_lo = G2_BASE_P / r
    p_m = np.where(low_group, p_lo, p_hi)
    D_post = float((S_raw * p_m).sum())
    a_post = (S_raw * p_m / D_post) if D_post > 0 else np.zeros_like(S_raw)

    records = []
    for (j, k), cert in R0.items():
        stronger, weaker = (j, k) if cert == Certification.J_OVER_K else (k, j)
        rho_ref = a_hat_full[stronger] / a_hat_full[weaker] if a_hat_full[weaker] > 0 else float("inf")
        risky_assignment = bool(low_group[stronger] and not low_group[weaker])
        sign_reversal_possible = bool(risky_assignment and np.isfinite(rho_ref) and r > rho_ref)
        expected_post_thinning_gap = float(a_post[stronger] - a_post[weaker])
        margin_reversal_possible = bool(expected_post_thinning_gap < -delta)
        naive_cert = R_naive.get((j, k))
        naively_wrong = naive_cert == opposite[cert]
        records.append({
            "pair": f"{j}-{k}", "stronger": stronger, "weaker": weaker,
            "rho_ref_jk": float(rho_ref) if np.isfinite(rho_ref) else None,
            "r": r, "r_over_rho_ref": (r / rho_ref) if np.isfinite(rho_ref) and rho_ref > 0 else None,
            "risky_assignment": risky_assignment,
            "sign_reversal_possible": sign_reversal_possible,
            "expected_post_thinning_gap": expected_post_thinning_gap,
            "margin_reversal_possible": margin_reversal_possible,
            "naively_wrong": bool(naively_wrong),
        })
    return records


def run(
    config_path: str = CONFIG_PATH,
    n_reps: int = N_REPS,
    n_reps_targeted: int = N_REPS_TARGETED,
    seed: int = SEED,
    e6_results_path: str = E6_RESULTS_PATH,
) -> dict:
    cfg = load_config(config_path)
    alpha = cfg["certification"]["alpha"]
    delta = cfg["certification"]["delta"]
    top_20 = cfg["campaigns"]["top_20"]
    K = len(top_20)
    B = 10.0  # linear, A4 cap
    e6_gstar = _load_e6_linear_gamma_star(e6_results_path)
    e6_meta = _load_e6_meta(e6_results_path)
    if e6_meta["cert_method"] != active_cert_method() or e6_meta["boundary"] != MultiCampaignCumulativeShareCS(K, alpha, B).boundary:
        raise ValueError(f"E7 targets pairs by E6's Gamma*; E6 results were produced by {e6_meta} but the active engine is "
                         f"({active_cert_method()}, {MultiCampaignCumulativeShareCS(K, alpha, B).boundary}): regenerate E6 first")

    journeys, _ = build_and_cache_primary_journeys(config_path)
    campaign_idx, n_touches = _campaign_idx_and_n_touches(journeys)
    n_total = len(campaign_idx)
    T_full = n_touches.astype(float)

    Tj_full = np.zeros((n_total, K))
    Tj_full[np.arange(n_total), campaign_idx] = T_full
    mg0 = MultiCampaignCumulativeShareCS(K, alpha, B)
    for i in range(n_total):
        mg0.update(Tj_full[i], T_full[i])
    box0 = snapshot(mg0)
    R0 = _certified_graph(box0, REFERENCE_GAMMA, delta, K)
    a_hat_full = Tj_full.sum(axis=0) / T_full.sum()

    beta = alpha / (K + 1)
    result = {
        "K": K, "alpha": alpha, "delta": delta, "reference_gamma": REFERENCE_GAMMA,
        "naive_gamma": NAIVE_GAMMA, "n_reps": n_reps, "n_journeys": n_total,
        "n_reference_certified": len(R0),
        "cs_method": CS_METHOD, "beta": beta, "alpha_used": beta / 2.0, "cs_running_intersection": False,
        "cert_method": active_cert_method(), "boundary": mg0.boundary,
        "e6_gamma_star_source": {"cert_method": e6_meta.get("cert_method"), "boundary": e6_meta.get("boundary")},
        "G1": {}, "G2": {}, "G3": {},
    }

    rng = np.random.default_rng(seed)

    for rho in G1_RHO_GRID:
        p = 1.0 - rho
        naive_wr, robust_wr, drrs = [], [], []
        for _ in range(n_reps):
            Tj = _stream_uniform(campaign_idx, n_touches, p, rng, K)
            T = Tj.sum(axis=1)
            mg = MultiCampaignCumulativeShareCS(K, alpha, B)
            for i in range(n_total):
                mg.update(Tj[i], T[i])
            box = snapshot(mg)
            R_naive = _certified_graph(box, NAIVE_GAMMA, delta, K)
            R_robust = _certified_graph(box, REFERENCE_GAMMA, delta, K)
            naive_wr.append(_wrong_rate(R0, R_naive))
            robust_wr.append(_wrong_rate(R0, R_robust))
            drrs.append(_drr(R0, R_robust))
        result["G1"][rho] = {
            "naive_wrong_rate_mean": float(np.mean(naive_wr)),
            "robust_wrong_rate_mean": float(np.mean(robust_wr)),
            "drr_mean": float(np.mean(drrs)),
        }

    S_raw = Tj_full.sum(axis=0)

    for r in G2_R_GRID:
        naive_wr, robust_wr, drrs = [], [], []
        cross_check_all_reps = []
        for _ in range(n_reps):
            low_group = rng.random(K) < 0.5
            p_lo = G2_BASE_P / r
            Tj = _stream_g2(campaign_idx, n_touches, low_group, G2_BASE_P, p_lo, rng, K)
            T = Tj.sum(axis=1)
            mg = MultiCampaignCumulativeShareCS(K, alpha, B)
            for i in range(n_total):
                mg.update(Tj[i], T[i])
            box = snapshot(mg)
            R_naive = _certified_graph(box, NAIVE_GAMMA, delta, K)
            R_robust_matched = _certified_graph(box, r, delta, K)
            R_robust_ref = _certified_graph(box, REFERENCE_GAMMA, delta, K)
            naive_wr.append(_wrong_rate(R0, R_naive))
            robust_wr.append(_wrong_rate(R0, R_robust_matched))
            drrs.append(_drr(R0, R_robust_ref))
            cross_check_all_reps.append(_g2_cross_check_records(R0, low_group, r, R_naive, a_hat_full, S_raw, delta))

        n_sign_reversal = sum(1 for rep in cross_check_all_reps for rec in rep if rec["sign_reversal_possible"])
        n_margin_reversal = sum(1 for rep in cross_check_all_reps for rec in rep if rec["margin_reversal_possible"])
        n_margin_and_wrong = sum(1 for rep in cross_check_all_reps for rec in rep if rec["margin_reversal_possible"] and rec["naively_wrong"])
        n_no_margin_but_wrong = sum(1 for rep in cross_check_all_reps for rec in rep if not rec["margin_reversal_possible"] and rec["naively_wrong"])
        adverse_records = [rec for rep in cross_check_all_reps for rec in rep if rec["risky_assignment"]]
        result["G2"][r] = {
            "naive_wrong_rate_mean": float(np.mean(naive_wr)),
            "robust_wrong_rate_mean_gamma_used_eq_r": float(np.mean(robust_wr)),
            "drr_mean_at_reference_gamma": float(np.mean(drrs)),
            "cross_check": {
                "n_reps": n_reps,
                "n_pair_rep_combinations": n_reps * len(R0),
                "n_sign_reversal_possible": n_sign_reversal,
                "n_margin_reversal_possible": n_margin_reversal,
                "n_margin_reversal_possible_and_naively_wrong": n_margin_and_wrong,
                "n_no_margin_reversal_but_naively_wrong": n_no_margin_but_wrong,
                "adverse_assignment_records": adverse_records,
                "note": "(corrects an earlier Gamma*_jk<r diagnostic, which answered a "
                        "different question). sign_reversal_possible: risky_assignment AND r>rho_ref_jk "
                        "(population post-thinning RATIO falls below 1 -- the K-way denominator cancels in "
                        "a pairwise ratio). margin_reversal_possible: the ACTUAL expected post-thinning GAP "
                        "(computed from this rep's full K-campaign group assignment, K-way-normalised, not "
                        "inferred from the ratio) is < -delta -- the real necessary condition for a wrong "
                        "certification to be possible. n_no_margin_reversal_but_naively_wrong should be 0 if "
                        "naive wrong-certification is fully explained by margin_reversal_possible, not chance. "
                        "adverse_assignment_records carries every (pair,rep) where the stronger campaign was "
                        "actually assigned to the low-observability group, with both diagnostics and "
                        "naively_wrong reported individually.",
            },
        }

    for rho in G3_RHO_GRID:
        naive_wr, robust_wr, drrs = [], [], []
        for _ in range(n_reps):
            Tj = _stream_g3(campaign_idx, n_touches, rho, rng, K)
            T = Tj.sum(axis=1)
            mg = MultiCampaignCumulativeShareCS(K, alpha, B)
            for i in range(len(Tj)):
                mg.update(Tj[i], T[i])
            box = snapshot(mg)
            R_naive = _certified_graph(box, NAIVE_GAMMA, delta, K)
            R_robust = _certified_graph(box, REFERENCE_GAMMA, delta, K)
            naive_wr.append(_wrong_rate(R0, R_naive))
            robust_wr.append(_wrong_rate(R0, R_robust))
            drrs.append(_drr(R0, R_robust))
        result["G3"][rho] = {
            "naive_wrong_rate_mean": float(np.mean(naive_wr)),
            "robust_wrong_rate_mean": float(np.mean(robust_wr)),
            "drr_mean": float(np.mean(drrs)),
            "out_of_model": True,
        }

    # G2_targeted: deliberate, non-random, two population-calibrated settings.
    (least_j, least_k), least_gstar = min(e6_gstar.items(), key=lambda kv: kv[1])
    # Stronger/weaker by raw point-estimate share, not certified direction: an early stream can be unresolved, which
    # would swap them and make rho_ref < 1, breaking the r_low/r_high formulas.
    stronger, weaker = (least_j, least_k) if a_hat_full[least_j] >= a_hat_full[least_k] else (least_k, least_j)
    rho_ref_targeted = a_hat_full[stronger] / a_hat_full[weaker] if a_hat_full[weaker] > 0 else float("inf")
    gamma_star_at_delta = box_gamma_star(box0, delta, stronger, weaker, tol=GAMMA_STAR_TOL)

    r_low = G2_TARGETED_R_LOW_MULT * rho_ref_targeted    # predicts NO population sign reversal
    r_high = G2_TARGETED_R_HIGH_MULT * rho_ref_targeted  # predicts a population sign reversal

    targeted_settings = {}
    for label, r_t in [("r_low", r_low), ("r_high", r_high)]:
        low_group = np.zeros(K, dtype=bool)
        low_group[stronger] = True  # only the stronger campaign is disadvantaged; weaker and all others at baseline
        p_lo_t = G2_BASE_P / r_t
        expected_gap = _expected_post_thinning_gap(S_raw, low_group, G2_BASE_P, p_lo_t, stronger, weaker)
        crosses_margin = expected_gap < -delta

        naive_wr_t, robust_wr_t, robust_unresolved_t = [], [], []
        for _ in range(n_reps_targeted):
            Tj = _stream_g2(campaign_idx, n_touches, low_group, G2_BASE_P, p_lo_t, rng, K)
            T = Tj.sum(axis=1)
            mg = MultiCampaignCumulativeShareCS(K, alpha, B)
            for i in range(n_total):
                mg.update(Tj[i], T[i])
            box = snapshot(mg)
            naive_cert = box_rule(box, NAIVE_GAMMA, delta, stronger, weaker)
            robust_cert = box_rule(box, r_t, delta, stronger, weaker)  # correctly specified: Gamma_used=r
            naive_wr_t.append(1.0 if naive_cert == Certification.K_OVER_J else 0.0)
            robust_wr_t.append(1.0 if robust_cert == Certification.K_OVER_J else 0.0)
            robust_unresolved_t.append(1.0 if robust_cert == Certification.UNRESOLVED else 0.0)

        targeted_settings[label] = {
            "r": r_t, "gamma_used": r_t,
            "predicted_sign_reversal": bool(r_t > rho_ref_targeted),
            "expected_post_thinning_gap": expected_gap,
            "crosses_margin_delta": bool(crosses_margin),
            "n_reps": n_reps_targeted,
            "naive_wrong_rate": float(np.mean(naive_wr_t)),
            "robust_wrong_rate": float(np.mean(robust_wr_t)),
            "robust_unresolved_rate": float(np.mean(robust_unresolved_t)),
        }

    result["G2_targeted"] = {
        "pair": f"{stronger}-{weaker}", "stronger": stronger, "weaker": weaker,
        "rho_ref_jk": float(rho_ref_targeted) if np.isfinite(rho_ref_targeted) else None,
        "gamma_star_at_delta": gamma_star_at_delta,
        "settings": targeted_settings,
        "note": "e6's single least-robust LINEAR-certified pair, deliberately placed in "
                "opposite groups at r_low=0.9*rho_ref_jk (predicts no population sign reversal) and "
                "r_high=1.3*rho_ref_jk (predicts a population sign reversal), both using the correctly "
                "specified robust sensitivity parameter Gamma_used=r. gamma_star_at_delta is reported "
                "separately -- it is a margin/resolution-limited certification ceiling (see the paper), not "
                "the population sign-reversal threshold this experiment now targets.",
    }

    # G2_targeted consistency check on the E4 pair (1,10) at r=rho_ref_pair**2 and 4.0, Gamma_used=r, under the G2 mechanism.
    e4_j, e4_k = 1, 10
    e4_stronger, e4_weaker = (e4_j, e4_k) if a_hat_full[e4_j] >= a_hat_full[e4_k] else (e4_k, e4_j)
    e4_rho_ref = a_hat_full[e4_stronger] / a_hat_full[e4_weaker] if a_hat_full[e4_weaker] > 0 else float("inf")
    e4_r_values = [("rho_ref_squared", e4_rho_ref ** 2), ("r_4_0", 4.0)]
    e4_settings = _run_targeted_pair_settings(
        e4_stronger, e4_weaker, e4_r_values, campaign_idx, n_touches, n_total, K, alpha, B, delta,
        S_raw, rng, n_reps_targeted, rho_ref=e4_rho_ref)
    for label, r_t in e4_r_values:
        s = e4_settings[label]

    result["G2_targeted_consistency_check"] = {
        "pair": f"{e4_stronger}-{e4_weaker}", "stronger": e4_stronger, "weaker": e4_weaker,
        "rho_ref_jk": float(e4_rho_ref) if np.isfinite(e4_rho_ref) else None,
        "settings": e4_settings,
        "note": "The E4 Criteo pair (indices 1,10), run under E7's G2 mechanism (NOT E4's "
                "own single-campaign-thinned mechanism) at r=rho_ref_pair**2 and r=4.0, Gamma_used=r "
                "(correctly specified), alongside pair (7,15)'s G2_targeted above for comparison.",
    }

    return result


def write_outputs(result: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e7_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)


if __name__ == "__main__":
    result = run()
    write_outputs(result, "results/e7")
    print(f"R_0 has {result['n_reference_certified']} certified pairs (Gamma={REFERENCE_GAMMA})")
    print("wrote results/e7/e7_results.json")
