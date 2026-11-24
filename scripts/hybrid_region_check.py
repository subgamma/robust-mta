"""Read-only diagnostic: where can the clipped legacy bound exceed the direct bound, and does it matter?

Writes results/diagnostics/hybrid_region_check.json.
Part 1: random boxes consistent with a valid CS event (truth inside every box).
Part 2: the primary monitored stream (linear, K=20): one `MultiCampaignCumulativeShareCS` pass, boxes read at
geometric checkpoints, all 190 pairs, Gamma in {1, 1.25, 1.5}, delta=0.005.
Classes: (a) clipped > direct anywhere; (b) the same with clipped > 0; (c) clipped > delta >= direct (a
certification would change); (d) Gamma* would change.
"""
import os
import sys, json, itertools
sys.path.insert(0, ".")
import numpy as np
from anytime_attribution.certify import direct_lower_gap, robust_lower_gap, direct_gamma_star
from anytime_attribution.cs import MultiCampaignCumulativeShareCS
from anytime_attribution.experiments._primary_journeys import build_and_cache_primary_journeys
from anytime_attribution.experiments.e5_certified_fraction import _model_masses

DELTA = 0.005

def comps(LN, UN, L, U, G, j, k):
    d = direct_lower_gap(LN, UN, G, j, k)
    c = robust_lower_gap(L, U, G, j, k) if (U[j] > 0 and U[k] > 0) else -np.inf
    return d, c

rng = np.random.default_rng(0)
tot = a = b = c_ = 0
for _ in range(200000):
    K = int(rng.integers(2, 8))
    m = rng.dirichlet(np.ones(K)) * rng.uniform(0.05, 0.9); dd = m.sum()
    wn = rng.choice([0.02, 0.2, 0.6, 1.5]) * rng.uniform(0, 1, K) * m
    LN, UN = np.maximum(m - wn, 0), np.minimum(m + wn, 1)
    wd = rng.choice([0.02, 0.3, 0.8]) * dd
    LD, UD = max(dd - wd, 1e-9), min(dd + wd, 1)
    L, U = np.clip(LN / UD, 0, 1), np.clip(UN / LD, 0, 1)
    j, k = rng.choice(K, 2, replace=False); G = float(rng.choice([1.0, 1.25, 1.5, 2.0]))
    d, c = comps(LN, UN, L, U, G, j, k)
    tot += 1
    if c > d + 1e-12:
        a += 1; b += c > 0; c_ += (c > DELTA) and (d <= DELTA)
part1 = {"cases": tot, "clipped_gt_direct": a, "of_which_clipped_positive": b, "certification_would_change_at_delta": c_}

journeys, _ = build_and_cache_primary_journeys()
Tj, B = _model_masses(journeys, "linear", 20)
T = Tj.sum(axis=1); K = 20
assert np.allclose(T, Tj.sum(axis=1))
cps = sorted(set(int(x) for x in np.geomspace(20, len(T), 25)))
cs = MultiCampaignCumulativeShareCS(K, 0.05, B)
pairs = list(itertools.combinations(range(K), 2))
res = {"checkpoints": len(cps), "pair_gamma_evals": 0, "clipped_gt_direct": 0, "clipped_gt_direct_and_clipped_positive": 0,
       "certification_changes": 0, "gamma_star_changes": 0, "max_gain_when_clipped_gt_direct": 0.0, "first_t_clipped_positive_gain": None}
ci = 0
for i in range(len(T)):
    cs.update(Tj[i], T[i])
    if ci < len(cps) and i + 1 == cps[ci]:
        ci += 1
        LN, UN = cs.numerator_boxes(); L, U = cs.intervals()
        for (j, k) in pairs:
            for (x, y) in ((j, k), (k, j)):
                for G in (1.0, 1.25, 1.5):
                    d, c = comps(LN, UN, L, U, G, x, y)
                    res["pair_gamma_evals"] += 1
                    if c > d + 1e-12:
                        res["clipped_gt_direct"] += 1
                        res["max_gain_when_clipped_gt_direct"] = max(res["max_gain_when_clipped_gt_direct"], float(c - d))
                        if c > 0:
                            res["clipped_gt_direct_and_clipped_positive"] += 1
                            res["first_t_clipped_positive_gain"] = res["first_t_clipped_positive_gain"] or i + 1
                        if c > DELTA and d <= DELTA:
                            res["certification_changes"] += 1
                if direct_lower_gap(LN, UN, 1.0, x, y) > DELTA or robust_lower_gap(L, U, 1.0, x, y) > DELTA:
                    gd = direct_gamma_star(LN, UN, DELTA, x, y, tol=1e-6)
                    # clipped-legacy gamma* by bisection on its own lower gap
                    f = lambda G: robust_lower_gap(L, U, G, x, y)
                    if f(1.0) > DELTA:
                        lo, hi = 1.0, 1000.0
                        for _ in range(60):
                            mid = (lo + hi) / 2
                            lo, hi = (mid, hi) if f(mid) > DELTA else (lo, mid)
                        gl = lo
                    else:
                        gl = None
                    if gl is not None and (gd is None or gl > gd + 1e-5):
                        res["gamma_star_changes"] += 1
print("part2", res)
os.makedirs("results/diagnostics", exist_ok=True)
json.dump({"part1": part1, "part2": res}, open("results/diagnostics/hybrid_region_check.json", "w"), indent=1, default=int)
