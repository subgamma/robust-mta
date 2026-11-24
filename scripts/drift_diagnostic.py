"""Constant-share drift diagnostic and first-vs-last-fifth movement counts.

Runs `cs.drift_test_cs` (fixed-mean betting sequence, linear masses, primary Criteo stream, level alpha/K per
campaign) and the descriptive movement count. Writes results/diagnostics/drift_diagnostic.json.
"""
import os
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, ".")
import numpy as np
from anytime_attribution.cs import drift_test_cs
from anytime_attribution.experiments._primary_journeys import build_and_cache_primary_journeys
from anytime_attribution.experiments.e5_certified_fraction import _model_masses

ALPHA, K = 0.05, 20
_G = {}

def init():
    journeys, top = build_and_cache_primary_journeys("config/criteo.yaml")
    Tj, B = _model_masses(journeys, "linear", K)
    _G["Tj"], _G["B"] = Tj, B

def one(j):
    Tj, B = _G["Tj"], _G["B"]; T = Tj.sum(1)
    r = drift_test_cs(Tj[:, j].tolist(), T.tolist(), ALPHA / K, B)
    return j, r.status

if __name__ == "__main__":
    t0 = time.time(); init()
    with ProcessPoolExecutor(10, initializer=init) as ex:
        res = dict(ex.map(one, range(K)))
    Tj = _G["Tj"]; n = len(Tj); f = n // 5
    first = Tj[:f].sum(0) / Tj[:f].sum(); last = Tj[-f:].sum(0) / Tj[-f:].sum()
    move = np.abs(last - first) / np.maximum(first, 1e-12)
    out = {"alpha_prime": ALPHA / K, "n_conversions": int(n), "status_by_campaign": {str(k): v for k, v in res.items()},
           "n_reject_constant_share": int(sum(1 for v in res.values() if v == "empty")), "rejecting_campaigns": [k for k, v in res.items() if v == "empty"],
           "n_move_gt10pct_first_vs_last_fifth": int((move > 0.10).sum()), "definition_move": "|share in last fifth - share in first fifth| / share in first fifth, linear aggregate shares",
           "campaign3_share_first_fifth": float(first[3]), "campaign3_share_last_fifth": float(last[3]), "wall_time_s": time.time() - t0}
    os.makedirs("results/diagnostics", exist_ok=True)
    json.dump(out, open("results/diagnostics/drift_diagnostic.json", "w"), indent=1)
    print(out)
