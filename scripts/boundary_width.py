"""Observed width reduction of the exact vs simplified stitched boundary on the Criteo primary stream.

Linear masses, K = 20, alpha = 0.05, shared beta = alpha/21, alpha_used = beta/2; both forms passed explicitly.
"""
import os
import json, sys
sys.path.insert(0, ".")
import numpy as np
from anytime_attribution.cs_vectorized import cumulative_share_boxes
from anytime_attribution.experiments._primary_journeys import build_and_cache_primary_journeys
from anytime_attribution.experiments.e5_certified_fraction import _model_masses
journeys, top = build_and_cache_primary_journeys("config/criteo.yaml"); K = len(top)
out = {"K": K, "alpha": 0.05, "n": len(journeys), "by_model": {}}
for model in ("linear", "time_decay"):
    Tj, B = _model_masses(journeys, model, K); T = Tj.sum(1); au = 0.05 / (K + 1) / 2
    ex = cumulative_share_boxes(Tj, T, B, au, form="exact"); si = cumulative_share_boxes(Tj, T, B, au, form="simplified")
    rows = {}
    for t in (500, 1000, 5000, 20000, 50000, len(T)):
        i = t - 1
        wn_e, wn_s = ex["UN"][i] - ex["LN"][i], si["UN"][i] - si["LN"][i]
        wr_e, wr_s = ex["U"][i] - ex["L"][i], si["U"][i] - si["L"][i]
        ok = wn_s > 0
        rows[str(t)] = {"numerator_halfwidth_ratio_exact_over_simplified_median": float(np.median(wn_e[ok] / wn_s[ok])),
                        "numerator_width_reduction_pct_median": float(100 * (1 - np.median(wn_e[ok] / wn_s[ok]))),
                        "ratio_box_width_reduction_pct_median": float(100 * (1 - np.median(wr_e / np.maximum(wr_s, 1e-12)))),
                        "numerator_width_reduction_pct_min_max": [float(100 * (1 - (wn_e[ok] / wn_s[ok]).max())), float(100 * (1 - (wn_e[ok] / wn_s[ok]).min()))]}
    out["by_model"][model] = rows
    assert np.all(ex["UN"] <= si["UN"] + 1e-15) and np.all(ex["LN"] >= si["LN"] - 1e-15)   # pointwise nesting on the real stream
os.makedirs("results/diagnostics", exist_ok=True)
json.dump(out, open("results/diagnostics/boundary_width_criteo.json", "w"), indent=1)
for m, rows in out["by_model"].items():
    for t, r in rows.items(): print(m, t, "numerator -%.1f%%  ratio box -%.1f%%" % (r["numerator_width_reduction_pct_median"], r["ratio_box_width_reduction_pct_median"]))
