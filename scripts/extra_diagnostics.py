"""Additional diagnostics.

(1) E7 G3 fragmentation severity (same stream, g3_orphan_cut); (2) E10 break-even lambda* split by truth vs
engine; (3) E5 distribution of first-certification times.
Writes results/e7/g3_fragmentation_severity.json, results/e10/lambda_star_decomposition.json and
results/e5/tstar_distribution.json. Needs the configuration A, B, C runs and the 1M-truth B run of E10
(`make ablation`), and the configuration A run of E5.
"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
from anytime_attribution.experiments._primary_journeys import build_and_cache_primary_journeys
from anytime_attribution.experiments.e7_interventions import G3_RHO_GRID, g3_orphan_cut, _campaign_idx_and_n_touches

journeys, _ = build_and_cache_primary_journeys("config/criteo.yaml")
cidx, n = _campaign_idx_and_n_touches(journeys); n = np.asarray(n, dtype=int)
rng = np.random.default_rng(0); out = {"n_conversions": int(len(n)), "linear_mass": int(n.sum()), "reps": 200, "single_touch_fraction": float((n == 1).mean()), "by_rho": {}}
for rho in G3_RHO_GRID:
    rows = []
    for _ in range(200):
        k = g3_orphan_cut(n, rho, rng)
        rows.append([(k > 0).mean(), (k == n).mean(), k.mean(), k.sum() / n.sum(), (k[k > 0] / n[k > 0]).mean() if (k > 0).any() else 0.0, ((k == n) & (n == 1)).sum() / max((n == 1).sum(), 1)])
    r = np.mean(rows, axis=0)
    out["by_rho"][str(rho)] = {"fraction_conversions_cut": float(r[0]), "fraction_conversions_fully_orphaned_(zero observed mass)": float(r[1]), "mean_touches_cut_per_conversion": float(r[2]),
                               "fraction_of_linear_mass_lost": float(r[3]), "mean_fraction_of_touches_removed_among_cut_conversions": float(r[4]), "fraction_of_single_touch_conversions_fully_orphaned": float(r[5]),
                               "rows_emitted_per_conversion": 1.0}
json.dump(out, open("results/e7/g3_fragmentation_severity.json", "w"), indent=1)
print("G3 severity", {k: {a: round(b, 4) for a, b in v.items()} for k, v in out["by_rho"].items()})

def lam(res, m, r):
    mon = res["monitoring"][m][r]; nv = mon["naive(Gamma=1)"]; w = mon[f"wellspec(Gamma={float(r)})"]
    g = lambda d, l: d[f"abstention_adjusted_loss_normalised_lambda={l}"]["mean"]
    L0n, L1n, L0w, L1w = g(nv, 0.0), g(nv, 1.0), g(w, 0.0), g(w, 1.0)
    return {"lambda_star": (L0n - L0w) / ((L1w - L0w) - (L1n - L0n)), "naive_loss_lambda0": L0n, "naive_wrong_per_rep": nv["final_wrong"]["mean"], "robust_unresolved_actionable": w["unresolved_actionable"]["mean"], "robust_certified": w["final_correct"]["mean"], "robust_loss_lambda1": L1w}
srcs = {"A (legacy + simplified, 4M truth)": "results/config_runs/A/e10/e10_results.json", "B1M (hybrid + simplified, 1M truth)": "results/config_runs/B_pop1m/e10/e10_results.json", "B (hybrid + simplified, 4M truth)": "results/config_runs/B/e10/e10_results.json",
        "C (legacy + exact, 4M truth)": "results/config_runs/C/e10/e10_results.json", "D final (hybrid + exact, 4M truth)": "results/e10/e10_results.json"}
dec = {}
for lab, p in srcs.items():
    R = json.load(open(p)); dec[lab] = {f"{m}|{r}": lam(R, m, r) for m in ("linear", "time_decay") for r in ("2.0", "2.5")}
json.dump(dec, open("results/e10/lambda_star_decomposition.json", "w"), indent=1)
for lab, d in dec.items(): print(lab, {k: (round(v["lambda_star"], 3), round(v["naive_wrong_per_rep"], 2), round(v["robust_unresolved_actionable"], 1)) for k, v in d.items()})

e5 = json.load(open("results/e5/e5_results.json")); e5a = json.load(open("results/config_runs/A/e5/e5_results.json"))
ts = {}
for tag, res in (("A_legacy", e5a), ("D_final", e5)):
    for m in ("linear", "time_decay"):
        for G in ("1.0", "1.25", "1.5"):
            v = [x for x in res["models"][m]["gammas"][G]["t_first_certified"].values() if x is not None]
            ts[f"{tag}|{m}|{G}"] = {"n": len(v), "p10": float(np.percentile(v, 10)), "p25": float(np.percentile(v, 25)), "median": float(np.median(v)), "p75": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90))}
json.dump(ts, open("results/e5/tstar_distribution.json", "w"), indent=1)
print({k: (v["n"], v["p25"], v["median"], v["p75"], v["p90"]) for k, v in ts.items() if "|1.0" in k})
