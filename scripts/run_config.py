"""Run an experiment under one of the four controlled configurations.

  A = legacy ratio box + simplified boundary (reference)
  B = hybrid + simplified boundary
  C = legacy ratio box + exact boundary
  D = hybrid + exact boundary (final)
usage: python scripts/run_config.py <A|B|C|D> <e5|e6|...> [out_dir]
D with no out_dir writes to results/<exp>/; the others write to results/config_runs/<X>/<exp>/.
"""
import importlib, json, os, sys, time
sys.path.insert(0, ".")  # run from the repository root; this checkout takes precedence over any installed copy
CFG = {"A": ("legacy_ratio_box", "simplified"), "B": ("hybrid", "simplified"),
       "C": ("legacy_ratio_box", "exact"), "D": ("hybrid", "exact")}
cfg, exp = sys.argv[1], sys.argv[2]
cert, bnd = CFG[cfg]
out = sys.argv[3] if len(sys.argv) > 3 else (f"results/{exp}" if cfg == "D" else f"results/config_runs/{cfg}/{exp}")
mod = importlib.import_module({"e5": "anytime_attribution.experiments.e5_certified_fraction",
                               "e6": "anytime_attribution.experiments.e6_robustness_radius",
                               "e7": "anytime_attribution.experiments.e7_interventions",
                               "e8": "anytime_attribution.experiments.e8_instability",
                               "e9": "anytime_attribution.experiments.e9_sensitivity"}[exp])
t0 = time.time()
from anytime_attribution.certify import cert_method_override
from anytime_attribution.cs import boundary_override
kw = {}
if exp in ("e5", "e6", "e9"):
    kw = dict(cert_method=cert, boundary=bnd)
if exp == "e7":
    base = "results" if cfg == "D" else f"results/config_runs/{cfg}"
    kw = dict(e6_results_path=f"{base}/e6/e6_results.json")
if exp == "e8":
    base = "results" if cfg == "D" else f"results/config_runs/{cfg}"
    e7r = json.load(open(f"{base}/e7/e7_results.json"))
    e7r["G2"] = {float(k): v for k, v in e7r["G2"].items()}   # JSON round-trip stringifies float keys
    kw = dict(e7_result=e7r)
with cert_method_override(cert), boundary_override(bnd):
    res = mod.run(**kw)
res["configuration"] = {"label": cfg, "cert_method": cert, "boundary": bnd}
mod.write_outputs(res, out)
print(f"{exp} config {cfg} ({cert}, {bnd}) -> {out}  [{time.time()-t0:.0f}s]")
