"""CSV summary tables for E1, E3 and the E10 decomposition."""
import csv, json
J = lambda p: json.load(open(p))
e1 = J("results/e1/e1_results.json")
with open("results/e1/table_e1_coverage_width.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["panel", "alpha", "method", "anytime_valid", "pathwise_simultaneous_coverage_every_t", "simultaneous_coverage_at_t20000", "peek4_simultaneous_coverage", "median_halfwidth_t20000", "p90_halfwidth_t20000"])
    for st in ("stationary", "drift"):
        for al in ("0.05", "0.2", "0.5"):
            for m, e in e1[st][al].items():
                h = e["horizons"]["20000"]
                w.writerow([st, al, m, m.startswith("cs_"), e.get("pathwise_simultaneous_coverage", ""), h["simultaneous_coverage"], e["peek4_simultaneous_coverage"], h["halfwidth_median"], h["halfwidth_p90"]])
e3 = J("results/e3/e3_results.json")
with open("results/e3/table_e3_lp_verification.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cell", "Gamma_eff", "regime", "prop2_max_abs_dev_LP", "prop3_max_abs_dev_LP", "inadmissible_attaining_configs", "frac_conservative_strictly_larger", "mean_width_reduction_sharp_vs_conservative"])
    for k, v in e3["A_exact_LP_verification"].items():
        w.writerow([k, v["Gamma_eff"], v["regime"], v["prop2_max_abs_dev_LP_vs_sharp_closed_form"], v["prop3_max_abs_dev_LP_vs_sharp_closed_form"], v["n_attaining_configs_inadmissible"], v["frac_sets_where_conservative_strictly_larger_than_sharp"], v["mean_width_reduction_sharp_vs_conservative"]])
with open("results/e3/table_e3_composed_coverage.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["condition", "coverage_n5000", "coverage_n20000", "median_width_n20000"])
    for k, v in e3["C_finite_sample_composed_coverage"].items():
        w.writerow([k, v.get("coverage_n=5000"), v.get("coverage_n=20000"), v.get("median_interval_width_n=20000")])
for H in (50000, 250000):
    d = J(f"results/e10/e10_decomposition_N{H}.json")
    with open(f"results/e10/table_e10_decomposition_N{H}.csv", "w", newline="") as f:
        cols = ["model", "r_true", "rule", "Gamma", "Gamma_eff", "n_actionable", "n_observability_limited", "n_oracle_certifiable", "n_boundary_ambiguous_actionable", "mean_n_currently_certified_correct_actionable",
                "mean_n_sampling_limited_(oracle-certifiable, uncertified)", "mean_wrong_direction_certifications_actionable", "mean_certifications_of_nonactionable_pairs", "mean_finite_sample_shortfall_oracle_minus_hybrid_gap_(oracle-certifiable)"]
        w = csv.writer(f); w.writerow(cols)
        for r in d["rows"].values(): w.writerow([r[c] for c in cols])
print("tables written")
