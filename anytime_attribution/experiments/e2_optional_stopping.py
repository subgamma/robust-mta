"""E2: optional stopping (synthetic, verification only).

Stop at the first apparent significance and report the wrong-certification rate per method.

Setup: a dedicated population (distinct from E1's) where the compared campaigns
(j=4, k=5) have exactly equal true share (Delta=0) among 8 other campaigns. Under a
tie any certified ordering, in either direction, is a false stop, so a method
that is not anytime-valid will eventually cross a naive threshold from noise alone
as looks accumulate, while an anytime-valid method stays within its alpha budget.
Each method watches the stream; the first look at which it certifies an ordering
(delta=0) is its stopping decision. Wrong-certification rate = P(method ever
certifies an ordering).

Methods:
- bootstrap, fixed_eb, csranks_style: fixed-sample CIs recomputed at each look
  (baselines.py), with no time-uniform correction.
- naive_cs, robust_cs: a pairwise (K_for_beta=2: j, k and the shared denominator)
  `cs.MultiCampaignCumulativeShareCS` streamed incrementally; the first look at
  which `certify.robust_rule` at Gamma=1, delta=0 resolves the pair is the stop.
  naive_cs uses beta=alpha/C(K,2) and robust_cs beta=alpha/K. These are
  multiplicity choices, not a Gamma-robustness distinction: Gamma=1 throughout.
  The class is built with alpha_arg = 3*target_beta so that its internal
  beta = alpha_arg/(K_for_beta+1) equals the target.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..baselines import bootstrap_ci, csranks_style_ci, fixed_eb_ci
from ..certify import Certification
from ..pair_cert import PairStream
from ..estimands import campaign_masses
from ..journeys import apply_cap
from ..synth import generate_synthetic_journeys, true_aggregate_share
from .e1_coverage import MAX_TOUCHES

K = 10
J, KK = 4, 5
WEIGHTS = np.array([1.5, 1.3, 1.1, 0.9, 0.8, 0.8, 0.6, 0.5, 0.4, 0.3])
A_TRUE = true_aggregate_share(WEIGHTS)
B_BOUND = float(MAX_TOUCHES)
ALPHA = 0.05
N_REPS = 200
N_LOOKS = 1000
SEED = 0
N_PAIRS = K * (K - 1) // 2

CHECKPOINTS = tuple(sorted(set(np.geomspace(1, N_LOOKS, 80).astype(int).tolist())))


def _stream_Tj_T(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    journeys = generate_synthetic_journeys(n, WEIGHTS, rng)
    journeys = [apply_cap(j, lookback_days=7.0, max_touches=MAX_TOUCHES) for j in journeys]
    Tj = np.zeros((n, K))
    T = np.zeros(n)
    for i, j in enumerate(journeys):
        m = campaign_masses(j, K)
        Tj[i] = m
        T[i] = m.sum()
    return Tj, T


def _production_cs_first_stop(Tj_j: np.ndarray, Tj_k: np.ndarray, T: np.ndarray, target_beta: float):
    """First look at which the cumulative-share CS's robust rule (Gamma=1, delta=0, pairwise) resolves (j,k).

    Checked at every look. Returns (look, direction), or (None, None) if never resolved within n_looks.
    """
    alpha_arg = 3.0 * target_beta
    ps = PairStream(alpha_arg, B_BOUND, pool="tracked")
    n = len(Tj_j)
    for i in range(n):
        ps.update(Tj_j[i], Tj_k[i], T[i])
        cert = ps.rule(ps.snap(), 1.0, 0.0)
        if cert == Certification.J_OVER_K:
            return i + 1, +1
        if cert == Certification.K_OVER_J:
            return i + 1, -1
    return None, None


def _baseline_first_stop(Tj_j, Tj_k, T, method, checkpoints, rng):
    """First checkpoint (and direction) at which a fixed-sample method excludes 0 for a_j - a_k.

    Approximated by requiring the per-ratio delta-method (or bootstrap) intervals to separate, with no time-uniform correction.
    """
    for L in checkpoints:
        Tj_hist, Tk_hist, T_hist = Tj_j[:L], Tj_k[:L], T[:L]
        D_hist = Tj_hist - Tk_hist
        if method == "bootstrap":
            n = len(D_hist)
            idx = rng.integers(0, n, size=(1000, n))
            resampled = D_hist[idx].mean(axis=1)
            lo, hi = np.quantile(resampled, ALPHA / 2), np.quantile(resampled, 1 - ALPHA / 2)
        elif method == "fixed_eb":
            mean_d = D_hist.mean()
            se = D_hist.std(ddof=1) / np.sqrt(len(D_hist)) if len(D_hist) > 1 else abs(D_hist[0])
            from anytime_attribution.baselines import _normal_quantile

            z = _normal_quantile(1 - ALPHA / 2)
            slack = B_BOUND * np.log(2 / ALPHA) / (3 * max(len(D_hist) - 1, 1))
            lo, hi = mean_d - z * se - slack, mean_d + z * se + slack
        else:
            mean_d = D_hist.mean()
            se = D_hist.std(ddof=1) / np.sqrt(len(D_hist)) if len(D_hist) > 1 else abs(D_hist[0])
            from anytime_attribution.baselines import _normal_quantile

            z = _normal_quantile(1 - ALPHA / (2 * N_PAIRS))
            lo, hi = mean_d - z * se, mean_d + z * se
        if lo > 0:
            return L, +1
        if hi < 0:
            return L, -1
    return None, None


def run(n_reps: int = N_REPS, n_looks: int = N_LOOKS, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    methods = ["bootstrap", "fixed_eb", "csranks_style", "naive_cs", "robust_cs"]
    n_stopped = {m: 0 for m in methods}
    n_wrong = {m: 0 for m in methods}
    stop_looks = {m: [] for m in methods}
    t0 = time.time()

    for rep in range(n_reps):
        Tj, T = _stream_Tj_T(n_looks, rng)

        look, direction = _production_cs_first_stop(Tj[:, J], Tj[:, KK], T, ALPHA / N_PAIRS)
        if look is not None:
            n_stopped["naive_cs"] += 1
            stop_looks["naive_cs"].append(look)
            n_wrong["naive_cs"] += 1

        look, direction = _production_cs_first_stop(Tj[:, J], Tj[:, KK], T, ALPHA / K)
        if look is not None:
            n_stopped["robust_cs"] += 1
            stop_looks["robust_cs"].append(look)
            n_wrong["robust_cs"] += 1

        for method in ["bootstrap", "fixed_eb", "csranks_style"]:
            look, direction = _baseline_first_stop(Tj[:, J], Tj[:, KK], T, method, CHECKPOINTS, rng)
            if look is not None:
                n_stopped[method] += 1
                stop_looks[method].append(look)
                n_wrong[method] += 1


    result = {
        "methods": {
            m: {
                "n_stopped": n_stopped[m],
                "stop_rate": n_stopped[m] / n_reps,
                "wrong_given_stopped": (n_wrong[m] / n_stopped[m]) if n_stopped[m] > 0 else None,
                "wrong_rate_overall": n_wrong[m] / n_reps,
                "median_stop_look": float(np.median(stop_looks[m])) if stop_looks[m] else None,
            }
            for m in methods
        },
        "n_reps": n_reps,
        "n_looks": n_looks,
        "alpha": ALPHA,
        "K": K,
        "n_pairs": N_PAIRS,
        "true_delta": float(A_TRUE[J] - A_TRUE[KK]),
        "seed": seed,
        "wall_time_s": time.time() - t0,
        "cs_method": "howard_eb_ratio",
    }
    return result


def write_outputs(result: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e2_results.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["methods"], indent=2))
    write_outputs(result, "results/e2")
