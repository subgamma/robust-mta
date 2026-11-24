"""Regenerate all figures into figures/ from the result files in results/."""
import json, os, sys
sys.path.insert(0, ".")
import numpy as np
from anytime_attribution import plotting as P
OUT = "figures"
J = lambda p: json.load(open(p))
P.fig_e1(J("results/e1/e1_results.json"), f"{OUT}/fig_e1_coverage_width")
P.fig_e2(J("results/e2/e2_results.json"), f"{OUT}/fig_e2_optional_stopping")
P.fig_e3(J("results/e3/e3_results.json"), f"{OUT}/fig_e3_identification")
P.fig_e4(J("results/e4/e4_synthetic.json"), J("results/e4/e4_criteo.json"), f"{OUT}/fig_e4_reversal")
P.fig_e5(J("results/e5/e5_results.json"), J("results/config_runs/A/e5/e5_results.json"), f"{OUT}/fig_e5_certified_fraction")
P.fig_e6(J("results/e6/e6_results.json"), f"{OUT}/fig_e6_robustness_radius")
P.fig_e7(J("results/e7/e7_results.json"), f"{OUT}/fig_e7_stress_tests")
d50 = J("results/e10/e10_decomposition_N50000.json"); d250 = J("results/e10/e10_decomposition_N250000.json")
P.fig_e10_identifiability(d50, f"{OUT}/fig_e10_identifiability")
P.fig_e10_vs_N(d50, d250, f"{OUT}/fig_e10_certification_vs_N")
r50 = J("results/e10/e10_results.json")
P.fig_e10_wrong(r50, f"{OUT}/fig_e10_wrong_certifications")
P.fig_e10_tradeoff(r50, f"{OUT}/fig_e10_abstention_tradeoff")
# descriptive drift figure (linear observed shares; no inferential claim)
DRIFT_SERIES = "results/diagnostics/drift_series.json"
if os.path.exists(DRIFT_SERIES):  # precomputed series (no Criteo data needed)
    d = J(DRIFT_SERIES); cps = np.array(d["checkpoints"]); series = {int(c): v for c, v in d["series"].items()}
else:
    from anytime_attribution.experiments._primary_journeys import build_and_cache_primary_journeys
    from anytime_attribution.experiments.e5_certified_fraction import _model_masses
    journeys, top = build_and_cache_primary_journeys("config/criteo.yaml")
    Tj, _ = _model_masses(journeys, "linear", len(top))
    cs = np.cumsum(Tj, 0); tot = cs.sum(1, keepdims=True)
    cps = np.arange(1000, len(Tj) + 1, 1000)
    series = {c: (cs[cps - 1, c] / tot[cps - 1, 0]).tolist() for c in (0, 1, 3, 4, 14)}
    os.makedirs("results/diagnostics", exist_ok=True)
    json.dump({"checkpoints": cps.tolist(), "series": {str(c): v for c, v in series.items()}}, open(DRIFT_SERIES, "w"))
P.fig_drift(cps, series, {}, f"{OUT}/fig_temporal_drift")
print("figures written to", OUT)
