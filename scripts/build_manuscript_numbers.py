"""Build every number and table quoted in the paper from the experiment artifacts.

Outputs:
  generated_tables/numbers.tex     \\num{key} definitions
  generated_tables/tab_*.tex       generated tables
  results/manuscript_numbers.json  one record per number: key, value, display, source file(s),
                                   field or calculation, configuration, rounding

Everything is read from experiment outputs except design constants (config values and counts
quoted from `docs/criteo_preprocessing.md`). Each constant is registered with its source file and a
`must_contain` string asserted to occur there. Experiment outputs are never modified.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_T = ROOT / "generated_tables"
OUT_T.mkdir(exist_ok=True)
REG: dict[str, dict] = {}


def J(p):
    return json.loads((ROOT / p).read_text())


def txt(p):
    return (ROOT / p).read_text()


def fint(n, ids=False):
    n = int(round(n))
    return str(n) if ids or abs(n) < 1000 else f"{n:,}".replace(",", "{,}")


def fnum(x, d):
    return f"{x:.{d}f}"


def reg(key, value, display, source, field, config="", rounding=""):
    assert key not in REG, key
    REG[key] = {"key": key, "value": value, "display": display, "source": source, "field": field, "configuration": config, "rounding": rounding}


def const(key, value, display, source, must_contain, config="design constant / audited count quoted from the cited file", rounding="exact"):
    assert must_contain in txt(source), f"{key}: '{must_contain}' not found in {source}"
    reg(key, value, display, source, f"constant; verified to contain '{must_contain}'", config, rounding)


E1p, E1b = "results/e1/e1_results.json", "results/e1/e1_simplified_boundary.json"
E2p, E3p, E3b = "results/e2/e2_results.json", "results/e3/e3_results.json", "results/e3/e3_simplified_boundary.json"
E4s, E4c = "results/e4/e4_synthetic.json", "results/e4/e4_criteo.json"
E5p, E6p, E7p, E8p, E9p = ("results/e5/e5_results.json", "results/e6/e6_results.json", "results/e7/e7_results.json", "results/e8/e8_results.json", "results/e9/e9_results.json")
E10p, E10N = "results/e10/e10_results.json", "results/e10/e10_results_N250000.json"
DEC50, DEC250, ORC = "results/e10/e10_decomposition_N50000.json", "results/e10/e10_decomposition_N250000.json", "results/e10/e10_oracle.json"
CMP, BW, DRIFT, TSTAR, G3S, LAMD = ("results/config_runs/comparison.json", "results/diagnostics/boundary_width_criteo.json", "results/diagnostics/drift_diagnostic.json",
                                    "results/e5/tstar_distribution.json", "results/e7/g3_fragmentation_severity.json", "results/e10/lambda_star_decomposition.json")
DOC = "docs/criteo_preprocessing.md"
HYB = "results/diagnostics/hybrid_region_check.json"
e1, e1b, e2, e3, e3b = J(E1p), J(E1b), J(E2p), J(E3p), J(E3b)
e4s, e4c, e5, e6, e7, e8, e9 = J(E4s), J(E4c), J(E5p), J(E6p), J(E7p), J(E8p), J(E9p)
e10, e10n, d50, d250, orc = J(E10p), J(E10N), J(DEC50), J(DEC250), J(ORC)
cmp_, bw, drift, tstar, g3s, lamd = J(CMP), J(BW), J(DRIFT), J(TSTAR), J(G3S), J(LAMD)
last = lambda d: d[max(d, key=lambda x: float(x))]

const("crit.K.primary", 20, "20", "config/criteo.yaml", "top_20")
const("crit.K.secondary", 10, "10", "config/criteo.yaml", "top_10_by_exposure")
const("crit.alpha", 0.05, "0.05", "config/criteo.yaml", "alpha")
const("crit.delta", 0.005, "0.005", "config/criteo.yaml", "delta: 0.005")
const("crit.rows", 16468027, fint(16468027), DOC, "16,468,027")
reg("crit.rows.m", 16.468027, "16.5", DOC, "16,468,027 / 1e6", "", "1 decimal")
const("crit.campaigns", 675, "675", DOC, "675")
const("crit.conv.rows", 806196, fint(806196), DOC, "806,196")
const("crit.journeys", 438730, fint(438730), DOC, "438,730")
const("crit.causalmta", 435810, fint(435810), DOC, "435,810")
const("crit.collide", 2910, fint(2910), DOC, "2,910")
const("crit.warm.events", 65372, fint(65372), DOC, "65,372")
const("crit.mon.events", 274337, fint(274337), DOC, "274,337")
const("crit.warm.top20", 33324, fint(33324), DOC, "33,324")
const("crit.top20.events", 123439, fint(123439), DOC, "123,439")
const("crit.dropped", 14218, fint(14218), DOC, "14,218")
const("crit.processed", 109221, fint(109221), DOC, "109,221")
const("crit.sec.journeys", 20089, fint(20089), DOC, "20,089")
const("crit.sec.multi", 804, "804", DOC, "804")
const("crit.cap.cover", 99.73, "99.73", DOC, "99.73")
reg("crit.top20.pct", 100 * 123439 / 274337, fnum(100 * 123439 / 274337, 1), DOC, "123,439 / 274,337 x 100", "", "1 decimal")
reg("crit.dropped.pct", 100 * 14218 / 123439, fnum(100 * 14218 / 123439, 1), DOC, "14,218 / 123,439 x 100", "", "1 decimal")
reg("crit.processed.pct", 100 * 109221 / 274337, fnum(100 * 109221 / 274337, 1), DOC, "109,221 / 274,337 x 100", "", "1 decimal")
const("crit.beyond7.n", 47723, fint(47723), DOC, "47,723")
reg("crit.beyond7.pct", 100 * 47723 / 208101, fnum(100 * 47723 / 208101, 1), DOC, "47,723 / 208,101 x 100 (touch rows of the 123,439 monitored top-20 journeys)", "", "1 decimal")
assert "208,101" in txt(DOC)
AUD = "results/diagnostics/criteo_selection_audit.json"; aud = J(AUD)
_ov = 100 * aud["monitored_users_also_in_warmup_conversions"] / aud["monitored_users"]
reg("crit.overlap.pct", _ov, fnum(_ov, 1), AUD, "100 x monitored_users_also_in_warmup_conversions / monitored_users", "", "1 decimal")
reg("crit.sec.multi.pct", 100 * 804 / 20089, fnum(100 * 804 / 20089, 1), DOC, "804 / 20,089 x 100", "", "1 decimal")
reg("crit.overlap.top20", aud["primary_overlap_warmup_vs_monitored_top20"], str(aud["primary_overlap_warmup_vs_monitored_top20"]), AUD, "primary_overlap_warmup_vs_monitored_top20", "", "exact")

cfg1 = e1["config"]
reg("e1.K", cfg1["K"], str(cfg1["K"]), E1p, "config.K"); reg("e1.N", cfg1["N"], fint(cfg1["N"]), E1p, "config.N")
reg("e1.reps", cfg1["reps"], str(cfg1["reps"]), E1p, "config.reps"); reg("e1.t0", cfg1["pathwise_t0"], str(cfg1["pathwise_t0"]), E1p, "config.pathwise_t0")
cov = [e1[st][a][m]["pathwise_simultaneous_coverage"] for st in ("stationary", "drift") for a in ("0.05", "0.2", "0.5") for m in ("cs_shared", "cs_single")]
assert min(cov) == 1.0
reg("e1.cs.cov", min(cov), fnum(min(cov), 3), E1p, "min over {stationary,drift} x alpha x {cs_shared,cs_single} of pathwise_simultaneous_coverage", "200 replications", "3 decimals")
for a in ("0.05", "0.2", "0.5"):
    v = e1["stationary"][a]["csranks_style"]["pathwise_simultaneous_coverage"]
    reg(f"e1.cr.path.{a}", v, fnum(v, 3), E1p, f"stationary.{a}.csranks_style.pathwise_simultaneous_coverage", "every t in [500,20000]", "3 decimals")
hw = lambda d, st, a, m: d[st][a][m]["horizons"]["20000"]["halfwidth_median"]
for st, k in (("stationary", "stat"), ("drift", "drift")):
    v = hw(e1, st, "0.05", "cs_shared") / hw(e1, st, "0.05", "csranks_style")
    reg(f"e1.mult.{k}", v, fnum(v, 2), E1p, f"{st}.0.05: halfwidth_median(cs_shared)/halfwidth_median(csranks_style) at t=20000", "", "2 decimals")
    r = 100 * (1 - hw(e1, st, "0.05", "cs_shared") / hw(e1b, st, "0.05", "cs_shared"))
    reg(f"e1.exact.red.{k}", r, fnum(r, 0), f"{E1p}; {E1b}", f"100 x (1 - hw_exact/hw_simplified), {st}, alpha=0.05, t=20000 (simplified = staged boundary)", "", "integer percent")
v = e1["stationary"]["0.05"]["cs_shared"]["horizons"]["20000"]["halfwidth_p90"] / hw(e1, "stationary", "0.05", "cs_shared")
reg("e1.p90.ratio", v, fnum(v, 2), E1p, "stationary.0.05.cs_shared: halfwidth_p90/halfwidth_median at t=20000", "", "2 decimals")
v = hw(e1, "stationary", "0.05", "cs_shared") / hw(e1, "stationary", "0.05", "cs_single")
reg("e1.mult.shared.single", v, fnum(v, 2), E1p, "stationary.0.05: halfwidth_median(cs_shared)/halfwidth_median(cs_single) at t=20000", "", "2 decimals")
nm = [x["numerator_width_reduction_pct_median"] for m in bw["by_model"].values() for x in m.values()]
reg("bw.num.min", min(nm), fnum(min(nm), 0), BW, "min over model x t of numerator_width_reduction_pct_median", "Criteo, K=20, alpha=0.05", "integer percent")
reg("bw.num.max", max(nm), fnum(max(nm), 0), BW, "max over model x t of numerator_width_reduction_pct_median", "Criteo, K=20, alpha=0.05", "integer percent")

reg("e2.looks", e2["n_looks"], fint(e2["n_looks"]), E2p, "n_looks"); reg("e2.reps", e2["n_reps"], str(e2["n_reps"]), E2p, "n_reps")
for k, m in (("boot", "bootstrap"), ("feb", "fixed_eb"), ("cr", "csranks_style")):
    v = 100 * e2["methods"][m]["wrong_rate_overall"]
    reg(f"e2.{k}.pct", v, fnum(v, 0), E2p, f"methods.{m}.wrong_rate_overall x 100", "", "integer percent")
assert e2["methods"]["naive_cs"]["wrong_rate_overall"] == 0 and e2["methods"]["robust_cs"]["wrong_rate_overall"] == 0

A3 = e3["A_exact_LP_verification"]
dev = max(max(v["prop2_max_abs_dev_LP_vs_sharp_closed_form"], v["prop3_max_abs_dev_LP_vs_sharp_closed_form"]) for v in A3.values())
reg("e3.lp.dev", dev, r"$1\times10^{-13}$", E3p, "max over 8 cells of prop2/prop3_max_abs_dev_LP_vs_sharp_closed_form", "", "rounded up to one significant figure"); assert dev <= 1e-13 + 1e-18
reg("e3.lp.cells", len(A3), str(len(A3)), E3p, "number of cells in A_exact_LP_verification")
fl = A3["Gamma=2.0,p_lo=0.8"]["mean_width_reduction_sharp_vs_conservative"]
reg("e3.floor.red", 100 * fl, fnum(100 * fl, 1), E3p, "A_exact_LP_verification[Gamma=2.0,p_lo=0.8].mean_width_reduction_sharp_vs_conservative x 100", "1000 random aggregate-share vectors", "1 decimal")
fd = e3["B_floor_binding_M1"]["pair_ordering"]["fraction"]
reg("e3.floor.diff", 100 * fd, fnum(100 * fd, 1), E3p, "B_floor_binding_M1.pair_ordering.fraction x 100", "90,000 pairs", "1 decimal")
p8 = e3["D_prop8_envelope"]
up = p8["attaining_constructions_n_min_1"][0]["relative_gap_to_upper_bound"]
reg("e3.p8.upper", up, r"$4\times10^{-6}$", E3p, "D_prop8_envelope.attaining_constructions_n_min_1[0].relative_gap_to_upper_bound", "Gamma=2, p_lo=0.2", "rounded up to one significant figure"); assert up <= 4e-6 + 1e-12
reg("e3.p8.n", p8["adversarial_random_search"]["n"], fint(p8["adversarial_random_search"]["n"]), E3p, "D_prop8_envelope.adversarial_random_search.n")
assert p8["adversarial_random_search"]["upper_violations"] == 0 and p8["adversarial_random_search"]["lower_violations"] == 0
C3 = e3["C_finite_sample_composed_coverage"]
reg("e3.cov.n", 20000, fint(20000), E3p, "composed-coverage horizon n"); reg("e3.cov.reps", e3["config"]["fs_reps"], str(e3["config"]["fs_reps"]), E3p, "config.fs_reps")
assert all(C3[k]["coverage_n=20000"] == 1.0 for k in C3 if "negctl" not in k)
reg("e3.w.sharp", C3["linear|interior|sharp_Gamma_eff=1.25"]["median_interval_width_n=20000"], fnum(C3["linear|interior|sharp_Gamma_eff=1.25"]["median_interval_width_n=20000"], 3), E3p, "C...[linear|interior|sharp_Gamma_eff=1.25].median_interval_width_n=20000", "", "3 decimals")
reg("e3.w.cons", C3["linear|interior|conservative_Gamma=2"]["median_interval_width_n=20000"], fnum(C3["linear|interior|conservative_Gamma=2"]["median_interval_width_n=20000"], 3), E3p, "C...[linear|interior|conservative_Gamma=2].median_interval_width_n=20000", "", "3 decimals")
nk = [k for k in C3 if k.startswith("linear|negctl")][0]
reg("e3.neg.cov", C3[nk]["coverage_n=20000"], fnum(C3[nk]["coverage_n=20000"], 3), E3p, f"C...[{nk}].coverage_n=20000", "", "3 decimals")

reg("e4.syn.rho", e4s["A_true"][0] / e4s["A_true"][1], fnum(e4s["A_true"][0] / e4s["A_true"][1], 2), E4s, "A_true[0]/A_true[1]", "", "2 decimals")
reg("e4.syn.N", e4s["n_looks"], fint(e4s["n_looks"]), E4s, "n_looks"); reg("e4.syn.reps", e4s["n_reps"], str(e4s["n_reps"]), E4s, "n_reps")
m3 = e4s["misspecified_r_true_3"]["by_gamma_used"]["1.1"]
reg("e4.mis.N", m3["n_looks"], fint(m3["n_looks"]), E4s, "misspecified_r_true_3.by_gamma_used[1.1].n_looks"); reg("e4.mis.reps", m3["n_reps"], str(m3["n_reps"]), E4s, "misspecified_r_true_3.by_gamma_used[1.1].n_reps")
reg("e4.crit.j", e4c["campaign_j"]["id"], str(e4c["campaign_j"]["id"]), E4c, "campaign_j.id"); reg("e4.crit.k", e4c["campaign_k"]["id"], str(e4c["campaign_k"]["id"]), E4c, "campaign_k.id")
r0 = list(e4c["results"].values())[0]
reg("e4.crit.rho", r0["rho_ref"], fnum(r0["rho_ref"], 2), E4c, "results[*].rho_ref", "", "2 decimals"); reg("e4.crit.rho2", e4c["rho_ref_squared"], fnum(e4c["rho_ref_squared"], 2), E4c, "rho_ref_squared", "", "2 decimals")
reg("e4.crit.reps", e4c["n_reps"], str(e4c["n_reps"]), E4c, "n_reps")
for r, v in e4c["results"].items():
    if float(r) >= 3.0: assert last(v["naive_wrong_rate"]) == 1.0
    assert last(v["robust_wrong_rate"]) == 0.0

G = ("1.0", "1.25", "1.5")
lin = e5["models"]["linear"]["gammas"]
for g in G:
    reg(f"e5.lin.cert.{g}", lin[g]["n_current_certified"], str(lin[g]["n_current_certified"]), E5p, f"models.linear.gammas.{g}.n_current_certified", "hybrid + exact, day 30")
    reg(f"e5.lin.t.{g}", lin[g]["median_t_first_certified"], fint(lin[g]["median_t_first_certified"]), E5p, f"models.linear.gammas.{g}.median_t_first_certified", "", "integer")
v = lin["1.5"]["median_t_first_certified"] / lin["1.0"]["median_t_first_certified"]
reg("e5.lin.t.ratio", v, fnum(v, 2), E5p, "median_t_first_certified(Gamma=1.5)/median_t_first_certified(Gamma=1.0), linear", "", "2 decimals")
reg("e5.lin.unres.1.5", lin["1.5"]["current_unresolved_fraction_day30"], fnum(lin["1.5"]["current_unresolved_fraction_day30"], 3), E5p, "models.linear.gammas.1.5.current_unresolved_fraction_day30", "", "3 decimals")
wd = [g["n_withdrawn"] for m in e5["models"].values() for g in m["gammas"].values()]
reg("e5.wd.min", min(wd), str(min(wd)), E5p, "min over 12 (model, Gamma) cells of n_withdrawn"); reg("e5.wd.max", max(wd), str(max(wd)), E5p, "max over 12 cells of n_withdrawn")
assert sum(g["direction_flips"]["n_flip_events"] for m in e5["models"].values() for g in m["gammas"].values()) == 0
cm = cmp_["e5"]; cc = cm["linear|1.0|certified pairs @day30 (of 190)"]; ct = cm["linear|1.0|median t* (conversions)"]
reg("abl.cert.hyb", cc["hybrid_effect(B-A)"], str(int(cc["hybrid_effect(B-A)"])), CMP, "e5[linear|1.0|certified pairs].hybrid_effect(B-A)", "A/B/C/D identical streams")
reg("abl.cert.bnd", cc["boundary_effect(C-A)"], str(int(cc["boundary_effect(C-A)"])), CMP, "e5[...].boundary_effect(C-A)")
reg("abl.cert.both", cc["total(D-A)"], str(int(cc["total(D-A)"])), CMP, "e5[...].total(D-A)")
reg("abl.t.A", ct["A"], fint(ct["A"]), CMP, "e5[linear|1.0|median t*].A"); reg("abl.t.D", ct["D"], fint(ct["D"]), CMP, "e5[linear|1.0|median t*].D")
gr = cmp_["e6"]["linear"]["mean_ratio_to_A_on_common"]
for k, c in (("hyb", "B"), ("bnd", "C"), ("both", "D")):
    reg(f"abl.g.{k}", 100 * (gr[c] - 1), fnum(100 * (gr[c] - 1), 1), CMP, f"100 x (e6.linear.mean_ratio_to_A_on_common[{c}] - 1)", "pairs defined in all four configurations", "1 decimal")

for k, m in (("lin", "linear"), ("td", "time_decay")):
    g6 = e6["guaranteed"][m]
    reg(f"e6.{k}.n", g6["n_certified"], str(g6["n_certified"]), E6p, f"guaranteed.{m}.n_certified"); reg(f"e6.{k}.med", g6["median"], fnum(g6["median"], 2), E6p, f"guaranteed.{m}.median", "", "2 decimals")
g6 = e6["guaranteed"]["linear"]
reg("e6.lin.q25", g6["q25"], fnum(g6["q25"], 2), E6p, "guaranteed.linear.q25", "", "2 decimals"); reg("e6.lin.q75", g6["q75"], fnum(g6["q75"], 2), E6p, "guaranteed.linear.q75", "", "2 decimals")
reg("e6.lin.max", g6["max"], fnum(g6["max"], 2), E6p, "guaranteed.linear.max", "", "2 decimals")
reg("e6.cap", e6["floor_cap_1_over_p_lo"], fnum(e6["floor_cap_1_over_p_lo"], 0), E6p, "floor_cap_1_over_p_lo", "p_lo = 0.2")
assert all(e6["guaranteed"][m]["n_above_floor_cap"] == 0 for m in ("linear", "time_decay"))
reg("e6.lin.n.A", cmp_["e6"]["linear"]["n_certified"]["A"], str(cmp_["e6"]["linear"]["n_certified"]["A"]), CMP, "e6.linear.n_certified.A")
assert all(cmp_["e6"][m][k] == 0 for m in ("linear", "time_decay") for k in ("violations_B_ge_A", "violations_D_ge_C", "violations_C_ge_A", "violations_D_ge_B"))

reg("e6.lin.min", min(v for v in e6["guaranteed"]["linear"]["gamma_star_by_pair"].values() if v is not None), fnum(min(v for v in e6["guaranteed"]["linear"]["gamma_star_by_pair"].values() if v is not None), 3), E6p, "min over pairs of guaranteed.linear.gamma_star_by_pair", "", "3 decimals")
reg("e6.min.rho", e7["G2_targeted"]["rho_ref_jk"], fnum(e7["G2_targeted"]["rho_ref_jk"], 2), E7p, "G2_targeted.rho_ref_jk (observed share ratio of the least robust pair; E7 targets E6's minimum-radius pair)", "", "2 decimals")
assert e7["G2_targeted"]["pair"] == min(((v, k) for k, v in e6["guaranteed"]["linear"]["gamma_star_by_pair"].items() if v is not None))[1]
hyb = J(HYB)["part2"]
reg("hyb.clip.evals", hyb["pair_gamma_evals"], fint(hyb["pair_gamma_evals"]), HYB, "part2.pair_gamma_evals", "", "exact")
reg("hyb.clip.n", hyb["clipped_gt_direct"], str(hyb["clipped_gt_direct"]), HYB, "part2.clipped_gt_direct", "", "exact")
reg("e7.R0", e7["n_reference_certified"], str(e7["n_reference_certified"]), E7p, "n_reference_certified")
mx = max(v[k] for g in ("G1", "G2", "G3") for v in e7[g].values() for k in v if "wrong" in k and isinstance(v[k], float))
reg("e7.wrong.max", mx, fnum(mx, 3), E7p, "max over G1,G2,G3 and parameters of every *wrong* rate", "", "3 decimals")
reg("e7.g1.drr.0.7", e7["G1"]["0.7"]["drr_mean"], fnum(e7["G1"]["0.7"]["drr_mean"], 2), E7p, "G1[0.7].drr_mean", "", "2 decimals")
for rho in ("0.2", "0.6"):
    reg(f"e7.g3.drr.{rho}", e7["G3"][rho]["drr_mean"], fnum(e7["G3"][rho]["drr_mean"], 2), E7p, f"G3[{rho}].drr_mean", "", "2 decimals")
reg("e7.g3.mass.0.6", 100 * g3s["by_rho"]["0.6"]["fraction_of_linear_mass_lost"], fnum(100 * g3s["by_rho"]["0.6"]["fraction_of_linear_mass_lost"], 0), G3S, "by_rho[0.6].fraction_of_linear_mass_lost x 100", "200 draws of the random cut", "integer percent")
reg("e7.g3.orph.0.6", 100 * g3s["by_rho"]["0.6"]["fraction_conversions_fully_orphaned_(zero observed mass)"], fnum(100 * g3s["by_rho"]["0.6"]["fraction_conversions_fully_orphaned_(zero observed mass)"], 0), G3S, "by_rho[0.6].fraction_conversions_fully_orphaned x 100", "200 draws", "integer percent")
reg("e7.targ.pair", e7["G2_targeted"]["pair"], e7["G2_targeted"]["pair"].replace("-", "--"), E7p, "G2_targeted.pair")
assert all(s["robust_wrong_rate"] == 0 and s["naive_wrong_rate"] == 0 for s in e7["G2_targeted"]["settings"].values())

reg("e8.R0", e8["n_reference_certified"], str(e8["n_reference_certified"]), E8p, "n_reference_certified"); reg("e8.boot.b", e8["sampling"]["bootstrap_b"], str(e8["sampling"]["bootstrap_b"]), E8p, "sampling.bootstrap_b")
reg("e8.boot", e8["sampling"]["reversal_rate"], fnum(e8["sampling"]["reversal_rate"], 4), E8p, "sampling.reversal_rate", "", "4 decimals")
mc = e8["model_choice_primary"]["reversal_rate_by_model"]; assert mc["last_touch"] == mc["logistic_shap"] and mc["time_decay"] == 0
reg("e8.mc.lt", mc["last_touch"], fnum(mc["last_touch"], 4), E8p, "model_choice_primary.reversal_rate_by_model.last_touch (= logistic_shap)", "", "4 decimals")
assert all(v == 0 for v in e8["secondary"]["model_choice_reversal_rate_by_model"].values())
for _m in ("linear", "time_decay"):
    _g = e9["sensitivity"][_m]["grid"]
    for _d in _g:
        _v = [_g[_d][x]["n_certified"] for x in ("1.0", "1.25", "1.5", "2.0")]; assert all(a >= b for a, b in zip(_v, _v[1:]))
    for _x in ("1.0", "1.25", "1.5", "2.0"):
        _v = [_g[d_][_x]["n_certified"] for d_ in ("0.0", "0.0025", "0.005")]; assert all(a >= b for a, b in zip(_v, _v[1:]))
gl = e9["sensitivity"]["linear"]["grid"]
reg("e9.lin.0.1.0", gl["0.0"]["1.0"]["n_certified"], str(gl["0.0"]["1.0"]["n_certified"]), E9p, "sensitivity.linear.grid[0.0][1.0].n_certified")
reg("e9.lin.0.5.2.0", gl["0.005"]["2.0"]["n_certified"], str(gl["0.005"]["2.0"]["n_certified"]), E9p, "sensitivity.linear.grid[0.005][2.0].n_certified")
qa = max(c["abs_diff_q_minus_a"] for c in e9["distortion"]["linear"]["per_campaign"])
reg("e9.qa.lin", qa, fnum(qa, 3), E9p, "max over campaigns distortion.linear.per_campaign.abs_diff_q_minus_a", "", "3 decimals")
reg("drift.n", drift["n_reject_constant_share"], str(drift["n_reject_constant_share"]), DRIFT, "n_reject_constant_share", "drift_test_cs, level alpha/K = 0.0025, linear masses")
reg("drift.move", drift["n_move_gt10pct_first_vs_last_fifth"], str(drift["n_move_gt10pct_first_vs_last_fifth"]), DRIFT, "n_move_gt10pct_first_vs_last_fifth", drift["definition_move"])
reg("drift.c3.first", 100 * drift["campaign3_share_first_fifth"], fnum(100 * drift["campaign3_share_first_fifth"], 1), DRIFT, "campaign3_share_first_fifth x 100", "", "1 decimal")
reg("drift.c3.last", 100 * drift["campaign3_share_last_fifth"], fnum(100 * drift["campaign3_share_last_fifth"], 2), DRIFT, "campaign3_share_last_fifth x 100", "", "2 decimals")

c10 = e10["config"]; comp = e10["population"]["composition"]
reg("e10.K", c10["K"], str(c10["K"]), E10p, "config.K"); reg("e10.npairs", c10["pairs"], str(c10["pairs"]), E10p, "config.pairs")
const("e10.B", 10, "10", "anytime_attribution/experiments/e10_multicampaign.py", "CAP = 10")
reg("e10.delta", c10["delta"], fnum(c10["delta"], 3), E10p, "config.delta", "", "3 decimals"); reg("e10.reps", c10["replications"], str(c10["replications"]), E10p, "config.replications")
reg("e10.N1", 50000, fint(50000), E10p, "config.N_monitored (primary horizon)"); reg("e10.N2", 250000, fint(250000), E10N, "config.N_monitored (nested horizon)")
reg("e10.frac2", 100 * comp["frac_ge2_distinct_campaigns"], fnum(100 * comp["frac_ge2_distinct_campaigns"], 0), E10p, "population.composition.frac_ge2_distinct_campaigns x 100", "1,000,000-conversion composition sample", "integer percent")
reg("e10.frac3", 100 * comp["frac_ge3_distinct_campaigns"], fnum(100 * comp["frac_ge3_distinct_campaigns"], 0), E10p, "population.composition.frac_ge3_distinct_campaigns x 100", "", "integer percent")
reg("e10.effk", comp["effective_K_inverse_simpson"], fnum(comp["effective_K_inverse_simpson"], 1), E10p, "population.composition.effective_K_inverse_simpson", "", "1 decimal")
pr = e10["population"]["pairwise_ranking_disagreement"]
reg("e10.dis.lt", pr["linear|last_touch"]["pairs_changing_order"], str(pr["linear|last_touch"]["pairs_changing_order"]), E10p, "population.pairwise_ranking_disagreement[linear|last_touch].pairs_changing_order")
reg("e10.dis.td", pr["linear|time_decay"]["pairs_changing_order"], str(pr["linear|time_decay"]["pairs_changing_order"]), E10p, "population.pairwise_ranking_disagreement[linear|time_decay].pairs_changing_order")
kt = e10["population"]["kendall_tau_between_model_A"]["linear|last_touch"]
reg("e10.tau.lt", kt, fnum(kt, 2), E10p, "population.kendall_tau_between_model_A[linear|last_touch]", "", "2 decimals")
for r in ("2.0", "2.5"):
    v = e10["population"]["observed_regimes"][r]["natural_reversal_prevalence_population"]["linear"]["of_which_actionable(|Delta|>delta)"]
    reg(f"e10.rev.lin.{r}", v, str(v), E10p, f"population.observed_regimes[{r}].natural_reversal_prevalence_population.linear.of_which_actionable")
reg("e10.orc.n", orc["config"]["n_conversions"], fint(orc["config"]["n_conversions"]), ORC, "config.n_conversions")
lpd = orc["lp_crosscheck_max_abs_deviation"]; reg("e10.lp.dev", lpd, r"$4\times10^{-16}$", ORC, "lp_crosscheck_max_abs_deviation", "", "rounded up to one significant figure"); assert lpd <= 4e-16
th = ("linear", "time_decay")
def drow(d, m, r): return d["rows"][f"{m}|{r}|wellspec"]
def dsum(d, r, f): return sum(drow(d, m, r)[f] for m in th)
for r, tag in (("1.25", "r1.25"), ("2.0", "r2.0"), ("2.5", "r2.5")):
    reg(f"e10.act.{tag}", dsum(d50, r, "n_actionable"), str(dsum(d50, r, "n_actionable")), DEC50, f"sum over linear,time_decay of rows[m|{r}|wellspec].n_actionable")
reg("e10.obs.r2.0", dsum(d50, "2.0", "n_observability_limited"), str(dsum(d50, "2.0", "n_observability_limited")), DEC50, "sum rows[m|2.0|wellspec].n_observability_limited")
reg("e10.orc.r2.0", dsum(d50, "2.0", "n_oracle_certifiable"), str(dsum(d50, "2.0", "n_oracle_certifiable")), DEC50, "sum rows[m|2.0|wellspec].n_oracle_certifiable")
reg("e10.amb.r2.0", dsum(d50, "2.0", "n_boundary_ambiguous_actionable"), str(dsum(d50, "2.0", "n_boundary_ambiguous_actionable")), DEC50, "sum rows[m|2.0|wellspec].n_boundary_ambiguous_actionable")
assert dsum(d50, "2.5", "n_oracle_certifiable") == 0 and dsum(d50, "2.5", "n_observability_limited") == dsum(d50, "2.5", "n_actionable")
reg("e10.orc.r1.25", dsum(d50, "1.25", "n_oracle_certifiable"), str(dsum(d50, "1.25", "n_oracle_certifiable")), DEC50, "sum rows[m|1.25|wellspec].n_oracle_certifiable")
sl = dsum(d50, "1.25", "mean_n_sampling_limited_(oracle-certifiable, uncertified)")
reg("e10.samp.r1.25", sl, fnum(sl, 1), DEC50, "sum rows[m|1.25|wellspec].mean_n_sampling_limited (mean per replication)", "100 replications", "1 decimal")
reg("e10.cert.r1.25.N1", dsum(d50, "1.25", "mean_n_certified_correct_among_oracle_certifiable"), fnum(dsum(d50, "1.25", "mean_n_certified_correct_among_oracle_certifiable"), 1), DEC50, "sum rows[m|1.25|wellspec].mean_n_certified_correct_among_oracle_certifiable", "", "1 decimal")
reg("e10.cert.r1.25.N2", dsum(d250, "1.25", "mean_n_certified_correct_among_oracle_certifiable"), fnum(dsum(d250, "1.25", "mean_n_certified_correct_among_oracle_certifiable"), 1), DEC250, "same at N=250,000", "", "1 decimal")
oc = sum(drow(d250, m, r)["mean_n_certified_correct_among_observability_limited"] for m in th for r in ("1.25", "1.5", "2.0", "2.5"))
reg("e10.obscert.N2", oc, fnum(oc, 2), DEC250, "sum over linear,time_decay x r of rows[...].mean_n_certified_correct_among_observability_limited", "N=250,000", "2 decimals")
assert dsum(d250, "2.0", "mean_n_certified_correct_among_oracle_certifiable") == dsum(d50, "2.0", "n_oracle_certifiable")
assert all(r["mean_total_wrong_certifications_per_rep"] == 0 for d in (d50, d250) for r in d["rows"].values() if r["rule"] == "wellspec")
for H, mon in e10n["monitoring_by_horizon"].items():
    assert all(mon[m][r][f"wellspec(Gamma={float(r)})"]["reps_with_any_wrong_certification_at_any_checkpoint"][0] == 0 for m in mon for r in mon[m])
assert all(e10["monitoring"][m][r][f"wellspec(Gamma={float(r)})"]["reps_with_any_wrong_certification_at_any_checkpoint"][0] == 0 for m in e10["monitoring"] for r in e10["monitoring"][m])
cpu = max(e10["monitoring"][m][r][f"wellspec(Gamma={float(r)})"]["any_wrong_rate_ci95_cp"][1] for m in e10["monitoring"] for r in e10["monitoring"][m])
reg("e10.cp.upper", 100 * cpu, fnum(100 * cpu, 1), E10p, "max over models,r of wellspec.any_wrong_rate_ci95_cp[1] x 100", "Clopper-Pearson 95%, 0 of 100", "1 decimal")
nv = [e10["monitoring"][m][r]["naive(Gamma=1)"]["final_wrong"]["mean"] for m in th for r in ("1.5", "2.0", "2.5")]
reg("e10.naive.min", min(nv), fnum(min(nv), 1), E10p, "min over linear,time_decay x r in {1.5,2,2.5} of naive final_wrong.mean", "", "1 decimal"); reg("e10.naive.max", max(nv), fnum(max(nv), 1), E10p, "max over same", "", "1 decimal")
nl = [e10["monitoring"][m][r]["naive(Gamma=1)"]["final_wrong"]["mean"] for m in ("last_touch", "logistic_shap") for r in ("2.0", "2.5")]
reg("e10.naive.learned.max", max(nl), fnum(max(nl), 1), E10p, "max over last_touch,logistic_shap x r in {2,2.5} of naive final_wrong.mean", "", "1 decimal")
for k, m in (("lin", "linear"), ("td", "time_decay")):
    v = e10["monitoring"][m]["2.5"]["misspec(Gamma=1.25)"]["final_wrong"]["mean"]
    assert e10["monitoring"][m]["2.5"]["misspec(Gamma=1.25)"]["reps_with_any_wrong_certification_at_any_checkpoint"][0] == 100
    reg(f"e10.mis.{k}.2.5", v, fnum(v, 1), E10p, f"monitoring.{m}.2.5.misspec(Gamma=1.25).final_wrong.mean", "", "1 decimal")
lam = {}
for m in th:
    for r in ("2.0", "2.5"):
        v = lamd["D final (hybrid + exact, 4M truth)"][f"{m}|{r}"]["lambda_star"]; lam[(m, r)] = v
reg("e10.lam.min", min(lam.values()), fnum(min(lam.values()), 3), LAMD, "min over linear,time_decay x r in {2,2.5} of lambda_star (D final)", "break-even lambda: L_naive(lam) = L_robust(lam)", "3 decimals")
reg("e10.lam.max", max(lam.values()), fnum(max(lam.values()), 3), LAMD, "max over same", "", "3 decimals")
const("e10.gen.zipf", 1.6, "1.6", "anytime_attribution/experiments/e10_multicampaign.py", "LENGTH_EXP = 1.6"); const("e10.gen.sticky", 0.30, "0.30", "anytime_attribution/experiments/e10_multicampaign.py", "STICKY = 0.30")
const("e10.gen.b0", -2.6, "-2.6", "anytime_attribution/experiments/e10_multicampaign.py", "B0 = -2.6"); const("e10.gen.popen", 0.90, "0.90", "anytime_attribution/experiments/e10_multicampaign.py", "P_OPEN = 0.90")
reg("e10.warm.paths", e10["warmup"]["candidate_paths"], fint(e10["warmup"]["candidate_paths"]), E10p, "warmup.candidate_paths")

def tex_table(name, label, caption, colspec, header, rows, resize=True, notes=""):
    body = "\n".join(" & ".join(str(c) for c in r) + r" \\" for r in rows)
    inner = f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}"
    if resize: inner = "\\resizebox{\\linewidth}{!}{%\n" + inner + "}"
    s = f"\\begin{{table}}[pos=H]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n{inner}\n{notes}\n\\end{{table}}\n"
    (OUT_T / name).write_text(s)

MLAB = {"linear": "linear", "time_decay": "time-decay", "last_touch": "last-touch", "logistic_shap": "logistic+SHAP"}
rows = []
for m, lab, av in (("cs_shared", "Cumulative-share sequence, shared budget $\\alpha/(K+1)$", "yes"), ("cs_single", "Single-campaign sequences, no multiplicity correction", "yes"),
                   ("csranks_style", "Rank-set interval, Bonferroni $\\alpha/K$", "no"), ("fixed_eb_bonf", "Fixed empirical-Bernstein, Bonferroni $\\alpha/K$", "no"),
                   ("bootstrap_bonf", "Percentile bootstrap, Bonferroni $\\alpha/K$", "no"), ("fixed_eb", "Fixed empirical-Bernstein, uncorrected", "no"), ("bootstrap", "Percentile bootstrap, uncorrected", "no")):
    r = [lab, av]
    for a in ("0.05", "0.2", "0.5"):
        e = e1["stationary"][a][m]; h = e["horizons"]["20000"]
        r += [fnum(e["pathwise_simultaneous_coverage"], 3) if "pathwise_simultaneous_coverage" in e else "--", fnum(h["simultaneous_coverage"], 3), fnum(h["halfwidth_median"], 4)]
    rows.append(r)
tex_table("tab_e1.tex", "tab:e1", "E1 (synthetic, stationary conditional shares, $K=10$, 200 replications, $N=20{,}000$): simultaneous coverage of all $K$ campaigns (pathwise at every $t\\in[500,20{,}000]$, and at $t=20{,}000$) and median half-width at $t=20{,}000$, for $\\alpha\\in\\{0.05,0.20,0.50\\}$. Anytime-valid sequences carry a time-uniform guarantee; fixed-sample intervals are valid at one prespecified horizon only (a dash marks a quantity not computed). A coverage of $1.000$ at $\\alpha=0.50$ indicates conservatism, not calibration.",
          "p{4.6cm}lrrrrrrrrr", r"Method & Anytime & \multicolumn{3}{c}{$\alpha=0.05$} & \multicolumn{3}{c}{$\alpha=0.20$} & \multicolumn{3}{c}{$\alpha=0.50$} \\" "\n" r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}" "\n" r" & valid & pathwise & at $20{,}000$ & half-width & pathwise & at $20{,}000$ & half-width & pathwise & at $20{,}000$ & half-width", rows)
rows = []
for k, v in A3.items():
    G_, plo = re.match(r"Gamma=([\d.]+),p_lo=([\de.\-]+)", k).groups()
    rows.append([G_, "none" if float(plo) < 1e-3 else plo, fnum(v["Gamma_eff"], 2), "floor binds ($\\Gamma>1/\\plo$)" if "floor" in v["regime"] else "ratio binds", f"{max(v['prop2_max_abs_dev_LP_vs_sharp_closed_form'], v['prop3_max_abs_dev_LP_vs_sharp_closed_form']):.1e}",
                 v["n_attaining_configs_inadmissible"], fnum(100 * v["mean_width_reduction_sharp_vs_conservative"], 1)])
tex_table("tab_e3.tex", "tab:e3", "E3 (synthetic verification of Theorem~\\ref{thm:ident}): independent linear-programming reference versus the closed-form sharp identified sets (maximum absolute deviation over 1{,}000 identified sets per cell, namely 200 random share vectors with five campaigns each, aggregate shares and pairwise gaps), inadmissible attaining configurations, and the mean width reduction of the sharp set relative to the ratio-only set. Guaranteed (deterministic) identification results.",
          "lllp{3.4cm}rrr", r"$\Gamma$ & $\plo$ & $\Geff$ & regime & max dev.\ & inadmissible & width reduction (\%)", rows)
rows = []
for r, v in e4s["results"].items():
    rows.append(["constructed mechanism (synthetic)", r, fnum(last(v["naive_wrong_rate"]), 3), fnum(last(v["robust_wrong_rate"]), 3), fnum(last(v["robust_correct_rate"]), 3), fnum(last(v["robust_unresolved_rate"]), 3)])
for g, v in e4s["misspecified_r_true_3"]["by_gamma_used"].items():
    rows.append([f"misspecification stream, $r_{{\\mathrm{{true}}}}=3$, $\\Gamma_{{\\mathrm{{used}}}}={g}$" + (" (well specified)" if v["well_specified"] else " (understated)"), "3.0", "--", fnum(last(v["robust_wrong_rate"]), 3), fnum(last(v["robust_correct_rate"]), 3), fnum(last(v["robust_unresolved_rate"]), 3)])
for r, v in e4c["results"].items():
    rows.append(["Criteo targeted worst-case pair", fnum(float(r), 2), fnum(last(v["naive_wrong_rate"]), 3), fnum(last(v["robust_wrong_rate"]), 3), fnum(last(v["robust_correct_rate"]), 3), fnum(last(v["robust_unresolved_rate"]), 3)])
tex_table("tab_e4.tex", "tab:e4", "E4: rates at the final sample size. Naive: $\\Gamma=1$; robust: $\\Gamma=r$ unless stated. The synthetic rows are a constructed mechanism; the Criteo rows are a targeted worst-case illustration for one pair and not a prevalence estimate. Robust wrong-certification control is guaranteed when $\\Gamma\\ge r$ and is not claimed for understated $\\Gamma$.",
          "p{6cm}rrrrr", r"Scenario & $r$ & naive wrong & robust wrong & robust correct & robust \unres", rows)
rows = []
for m in ("linear", "time_decay", "last_touch", "logistic_shap"):
    for g in G:
        x = e5["models"][m]["gammas"][g]
        import numpy as _np
        tv = [t for t in x["t_first_certified"].values() if t is not None]
        rows.append([MLAB[m], g, f"{x['n_current_certified']}/190", f"{x['n_ever_certified']}/190", x["n_withdrawn"], fnum(x["current_unresolved_fraction_day30"], 3), fint(_np.percentile(tv, 25)), fint(x["median_t_first_certified"]), fint(_np.percentile(tv, 75)), fint(_np.percentile(tv, 90)), "guaranteed" if m in ("linear", "time_decay") else "empirical"])
tex_table("tab_e5.tex", "tab:e5", "E5 (real data, Criteo primary construction, tracked pool, $K=20$, $\\delta=0.005$): certification status at day 30 (109{,}221 conversions) under the hybrid rule with the exact boundary. ``Current'' is the headline quantity; ``ever certified'' and ``withdrawn'' are descriptive. The $t^\\star$ columns give the 25th percentile, median, 75th and 90th percentile of the number of conversions to first certification among pairs ever certified. Last-touch and logistic+SHAP are empirical (no guarantee) and coincide on this construction.",
          "llrrrrrrrrl", r"Model & $\Gamma$ & current & ever & withdrawn & \unres\ (current) & $t^\star$ 25th & median & 75th & 90th & status", rows)
rows = [[r["campaign_idx"], fnum(r["min_gamma_star"], 4), r["n_certified_pairs"]] for r in e6["guaranteed"]["linear"]["fragility_ranking"][:5]]
tex_table("tab_e6frag.tex", "tab:e6frag", "E6 (real data, Criteo primary construction): the five least robust campaigns under linear attribution, ranked by the smallest robustness radius $\\Gamma^\\star$ among their certified pairs (hybrid rule, $\\delta=0.005$). Guaranteed radii; sensitivity values, not estimates of retention ratios.", "lrr", r"Campaign index & smallest $\Gamma^\star$ & certified pairs", rows, resize=False)
rows = []
def arow(lab, d, fmt):
    return [lab] + [fmt(d[c]) for c in "ABCD"] + [fmt(d["D"] - d["A"]) if lab else ""]
for m in ("linear", "time_decay"):
    for g in ("1.0",):
        d = cmp_["e5"][f"{m}|{g}|certified pairs @day30 (of 190)"]; rows.append([f"E5 {MLAB[m]}, $\\Gamma={g}$: certified pairs"] + [int(d[c]) for c in "ABCD"])
        d = cmp_["e5"][f"{m}|{g}|median t* (conversions)"]; rows.append([f"E5 {MLAB[m]}, $\\Gamma={g}$: median $t^\\star$"] + [fint(d[c]) for c in "ABCD"])
for m in ("linear", "time_decay"):
    x = cmp_["e6"][m]
    rows.append([f"E6 {MLAB[m]}: pairs with defined $\\Gamma^\\star$"] + [x["n_certified"][c] for c in "ABCD"])
    rows.append([f"E6 {MLAB[m]}: mean $\\Gamma^\\star$ relative to A (common pairs)"] + [fnum(x["mean_ratio_to_A_on_common"][c], 3) for c in "ABCD"])
for r in ("1.25", "1.5"):
    d = cmp_["e10"][f"linear|{r}|certified correct"]; rows.append([f"E10 linear, $r_{{\\mathrm{{true}}}}={r}$: pairs certified correctly per replication"] + [fnum(d[c], 1) for c in "ABCD"])
tex_table("tab_ablate.tex", "tab:ablate", "Controlled comparison on identical streams and seeds. A: ratio-box rule with the simpler boundary; B: hybrid rule with the simpler boundary; C: ratio-box rule with the exact boundary; D: hybrid rule with the exact boundary (final procedure). E5/E6: real data (Criteo primary, $\\delta=0.005$, day 30), theorem-covered fixed-weight rules; E10: synthetic, $N=50{,}000$, 100 replications, well-specified $\\Gamma=r_{\\mathrm{true}}$. The mean robustness radius is expressed relative to configuration A on the pairs defined in all four configurations.",
          "p{7cm}rrrr", r"Quantity & A & B & C & D", rows)
rows = []
for m in ("linear", "time_decay"):
    for g in G:
        x = e5["models"][m]["gammas"][g]; rows.append([MLAB[m], g, len(x["sampling_limited"]), len(x["observability_limited"]), x["n_current_certified"]])
tex_table("tab_e5diag.tex", "tab:e5diag", "E5 (real data): plug-in classification of the pairs not certified at day 30, using the observed shares as point boxes (estimated, descriptive; not a population oracle). Theorem-covered rules only; $\\delta=0.005$.",
          "llrrr", r"Model & $\Gamma$ & sampling-limited & observability-limited & certified", rows)
rows = []
sv = g3s["by_rho"]
for g, lab in (("G1", "G1 uniform loss"), ("G2", "G2 differential loss"), ("G3", "G3 identity fragmentation (out of model)")):
    for k, v in e7[g].items():
        rr = "robust_wrong_rate_mean" if "robust_wrong_rate_mean" in v else "robust_wrong_rate_mean_gamma_used_eq_r"; dr = "drr_mean" if "drr_mean" in v else "drr_mean_at_reference_gamma"
        extra = [fnum(100 * sv[k]["fraction_of_linear_mass_lost"], 1), fnum(100 * sv[k]["fraction_conversions_fully_orphaned_(zero observed mass)"], 1)] if g == "G3" else ["--", "--"]
        rows.append([lab, k, fnum(v["naive_wrong_rate_mean"], 3), fnum(v[rr], 3), fnum(v[dr], 3)] + extra)
tex_table("tab:e7_dummy", "x", "", "l", "x", [], resize=False) if False else None
tex_table("tab_e7.tex", "tab:e7", "E7 (real data with controlled interventions, Criteo primary construction, linear attribution): wrong-certification rates and decision retention rate (DRR) against the reference graph $R_0$ ($\\Gamma=1.25$, $\\delta=0.005$, day 30). Parameter: loss rate $\\rho$ (G1, G3) or retention ratio $r$ (G2). G3 lies outside baseline Assumption~\\ref{a:ret}; Remark~\\ref{r:boundedret} supplies a conservative sufficient condition for certain G3 settings, and no guarantee is claimed uniformly across all G3 settings. Fragmentation severity is the mean percentage of linear mass lost and of conversions reduced to zero observed mass.",
          "p{4.6cm}rrrrrr", r"Intervention & parameter & naive wrong & robust wrong & DRR & mass lost (\%) & zero-mass conversions (\%)", rows)
rows = [["Sampling (primary)", f"bootstrap, $B={e8['sampling']['bootstrap_b']}$", fnum(e8["sampling"]["reversal_rate"], 4)]]
for m, v in e8["model_choice_primary"]["reversal_rate_by_model"].items(): rows.append(["Model choice (primary)", MLAB[m], fnum(v, 4)])
rows.append(["Observability (primary, G2)", "mean over $r$", fnum(e8["observability_primary"]["reversal_rate_mean"], 4)])
for m, v in e8["secondary"]["model_choice_reversal_rate_by_model"].items(): rows.append(["Model choice (secondary)", MLAB.get(m, "DARNN"), fnum(v, 4)])
rows.append(["Training seed (secondary)", "DARNN seeds 1--4 versus seed 0", fnum(e8["secondary"]["darnn_seed_reversal_rate_mean"], 4)])
tex_table("tab_e8.tex", "tab:e8", f"E8 (real data, Criteo): rank-reversal rate by source against the reference graph $R_0$ (linear, $\\Gamma=1.25$, {e8['n_reference_certified']} certified pairs). Descriptive; the single primary-construction model-choice disagreement is a consequence of different mass scales on single-campaign journeys.",
          "llr", r"Source & detail & reversal rate", rows, resize=False)
rows = []
for m in ("linear", "time_decay"):
    for d_, row in e9["sensitivity"][m]["grid"].items():
        rows.append([MLAB[m], d_] + [row[g]["n_certified"] for g in ("1.0", "1.25", "1.5", "2.0")])
tex_table("tab_e9a.tex", "tab:e9a", "E9 (real data, Criteo primary construction): certified-pair count over the $\\delta\\times\\Gamma$ grid at day 30 for theorem-covered rules; the $\\Gamma=1$ column is the naive-rule count.", "lrrrrr", r"Model & $\delta$ & $\Gamma=1$ & $\Gamma=1.25$ & $\Gamma=1.5$ & $\Gamma=2$", rows, resize=False)
rows = [[MLAB[m], fnum(e9["distortion"][m]["sd_S_over_mean_S"], 3), fnum(max(c["abs_diff_q_minus_a"] for c in e9["distortion"][m]["per_campaign"]), 3)] for m in ("linear", "time_decay", "last_touch", "logistic_shap")]
tex_table("tab_e9b.tex", "tab:e9b", "E9 (real data, descriptive): dataset-level $\\mathrm{SD}(S)/\\E[S]$ and the maximum over 20 campaigns of $|q_j-a_j|$, the observed counterpart of the dashboard--aggregate gap of Theorem~\\ref{thm:dashboard}(i). No confidence sequence, $\\Gamma$ or identified-set claim applies.", "lrr", r"Model & $\mathrm{SD}(S)/\E[S]$ & $\max_j|q_j-a_j|$", rows, resize=False)
(OUT_T / "tab_e9.tex").write_text((OUT_T / "tab_e9a.tex").read_text() + "\n" + (OUT_T / "tab_e9b.tex").read_text())
rows = []
for m in th:
    for r in ("1.25", "1.5", "2.0", "2.5"):
        a, b = drow(d50, m, r), drow(d250, m, r)
        rows.append([MLAB[m], r, a["n_actionable"], a["n_observability_limited"], a["n_oracle_certifiable"], a["n_boundary_ambiguous_actionable"], fnum(a["mean_n_certified_correct_among_oracle_certifiable"], 1), fnum(a["mean_n_sampling_limited_(oracle-certifiable, uncertified)"], 1),
                     fnum(b["mean_n_certified_correct_among_oracle_certifiable"], 1), fnum(b["mean_n_sampling_limited_(oracle-certifiable, uncertified)"], 1), fnum(a["mean_total_wrong_certifications_per_rep"] + b["mean_total_wrong_certifications_per_rep"], 1)])
tex_table("tab_e10dec.tex", "tab:e10dec", "E10 (synthetic multi-campaign benchmark): population-oracle decomposition of actionable pairs ($|\\Delta_{jk}|>\\delta=0.005$) for the theorem-covered rules at the well-specified $\\Gamma=r_{\\mathrm{true}}$, over all 45 pairs. Population classification uses a $4{,}000{,}000$-conversion independent population (observability-limited: oracle lower gap $\\le\\delta$; population-certifiable: $>\\delta$; boundary-ambiguous: within three Monte Carlo standard errors of $\\delta$). The last columns are mean pairs per replication (100 replications) on nested streams at $N=50{,}000$ and $N=250{,}000$: population-certifiable pairs certified correctly, and sampling-limited pairs (population-certifiable but not certified). The final column is the total number of wrong certifications over both horizons (all 45 pairs).",
          "llrrrrrrrrr", r"Model & $r_{\mathrm{true}}$ & actionable & obs.-limited & pop.-certifiable & ambiguous & \multicolumn{2}{c}{$N=50{,}000$} & \multicolumn{2}{c}{$N=250{,}000$} & wrong \\" "\n" r" & & & & & & certified & sampl.-lim. & certified & sampl.-lim. & ", rows)
rows = []
for m in ("linear", "time_decay", "last_touch", "logistic_shap"):
    for r in ("1.25", "1.5", "2.0", "2.5"):
        mon = e10["monitoring"][m][r]; nv_, w_ = mon["naive(Gamma=1)"], mon[f"wellspec(Gamma={float(r)})"]; ms = mon.get("misspec(Gamma=1.25)")
        g = lambda d, l: d[f"abstention_adjusted_loss_normalised_lambda={l}"]["mean"]
        L0n, L1n, L0w, L1w = g(nv_, 0.0), g(nv_, 1.0), g(w_, 0.0), g(w_, 1.0)
        lam_ = (L0n - L0w) / ((L1w - L0w) - (L1n - L0n)) if (L1w - L0w) != (L1n - L0n) else None
        lamtxt = "none$^\\dagger$" if (L0n == 0 and L0w == 0) else fnum(lam_, 3)
        rows.append([MLAB[m] + ("" if m in th else " (empirical)"), r, fnum(nv_["final_wrong"]["mean"], 2), fnum(w_["final_wrong"]["mean"], 2), fnum(ms["final_wrong"]["mean"], 2) if ms else "--",
                     fnum(nv_["unresolved_actionable"]["mean"], 1), fnum(w_["unresolved_actionable"]["mean"], 1), lamtxt])
tex_table("tab_e10dv.tex", "tab:e10dv", "E10 (synthetic, $N=50{,}000$, 100 replications, 45 pairs per replication): wrong certifications per replication and unresolved actionable pairs for the naive rule ($\\Gamma=1$), the well-specified robust rule ($\\Gamma=r_{\\mathrm{true}}$) and, for $r_{\\mathrm{true}}\\ge2$, a misspecified robust rule ($\\Gamma=1.25$); and the break-even deferral cost $\\lambda^\\star$ of the normalised ranking loss (for $\\lambda<\\lambda^\\star$ robust abstention has the lower loss, for $\\lambda>\\lambda^\\star$ naive certification). Linear and time-decay are theorem-covered when well specified; last-touch and logistic+SHAP are empirical. ``Wrong'' includes certification of non-actionable pairs. $^\\dagger$No wrong certification on an actionable pair under either rule, so the loss of the robust rule is never smaller for $\\lambda>0$.",
          "llrrrrrr", r"Model & $r_{\mathrm{true}}$ & naive wrong & robust wrong & misspec.\ wrong & naive \unres & robust \unres & $\lambda^\star$", rows)

# Write numbers.tex and manuscript_numbers.json.
lines = ["% generated by scripts/build_manuscript_numbers.py -- do not edit", r"\makeatletter", r"\newcommand{\num}[1]{\csname num@#1\endcsname}"]
for k, v in REG.items():
    lines.append(r"\expandafter\def\csname num@%s\endcsname{%s}" % (k, v["display"]))
lines.append(r"\makeatother")
(OUT_T / "numbers.tex").write_text("\n".join(lines) + "\n")
(ROOT / "results" / "manuscript_numbers.json").write_text(json.dumps({"description": "Every number quoted in the manuscript via \\num{key}; generated by scripts/build_manuscript_numbers.py from final experiment artifacts.", "numbers": REG}, indent=1, default=str))
print(len(REG), "numbers registered;", len(list(OUT_T.glob('tab_*.tex'))), "table files")
