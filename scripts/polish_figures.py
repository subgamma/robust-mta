"""Regenerate the E1, E3 and E5 figures with final layout.

Reads the final result JSON files and overwrites figures/fig_e1_coverage_width, fig_e3_identification and
fig_e5_certified_fraction. Style constants come from anytime_attribution.plotting.
"""
import json, sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib.pyplot as plt
from anytime_attribution.plotting import OI, MODEL_LABEL, GUAR, _save
J = lambda p: json.load(open(p))
OUT = "figures"

e1 = J("results/e1/e1_results.json")
fig, ax = plt.subplots(1, 3, figsize=(12, 4.0))
alphas = ["0.05", "0.2", "0.5"]; x = np.arange(3)
meths = [("cs_shared", "Anytime-valid sequence (shared budget)", OI["blue"]), ("cs_single", "Anytime-valid sequences (single campaign)", OI["sky"]),
         ("csranks_style", "Rank-set interval, Bonferroni (recomputed at every $t$)", OI["orange"]), ("bootstrap_bonf", "Bootstrap, Bonferroni (recomputed at every $t$)", OI["yellow"])]
for i, (m, lab, c) in enumerate(meths):
    v = [e1["stationary"][a][m].get("pathwise_simultaneous_coverage", np.nan) for a in alphas]
    ax[0].bar(x + (i - 1.5) * 0.2, v, 0.2, color=c, label=lab)
for xi, a in zip(x, alphas): ax[0].plot([xi - 0.45, xi + 0.45], [1 - float(a)] * 2, color="k", lw=1, ls=":")
ax[0].set_xticks(x, [f"$\\alpha$ = {a}" for a in alphas]); ax[0].set_ylabel("Simultaneous coverage at every $t \\in [500, 20{,}000]$"); ax[0].set_title("(a) Pathwise coverage (dotted: nominal $1-\\alpha$)", fontsize=9); ax[0].set_ylim(0, 1.05)
for i, (m, lab, c) in enumerate([("cs_shared", "Anytime-valid sequence (shared budget)", OI["blue"]), ("csranks_style", "Rank-set interval, Bonferroni (one horizon)", OI["orange"]), ("bootstrap_bonf", "Bootstrap, Bonferroni (one horizon)", OI["yellow"])]):
    ax[1].bar(x + (i - 1) * 0.27, [e1["stationary"][a][m]["horizons"]["20000"]["halfwidth_median"] for a in alphas], 0.27, color=c)
ax[1].set_xticks(x, [f"$\\alpha$ = {a}" for a in alphas]); ax[1].set_ylabel("Median half-width at $t = 20{,}000$"); ax[1].set_title("(b) Width cost of anytime validity", fontsize=9)
for m, lab, c, ls in [("cs_shared", "", OI["blue"], "-"), ("csranks_style", "", OI["orange"], "--")]:
    d = e1["stationary"]["0.05"][m]["halfwidth_median_vs_t"]; ts = sorted(int(t) for t in d); ax[2].loglog(ts, [d[str(t)] for t in ts], color=c, ls=ls)
ax[2].set_xlabel("Conversions $t$"); ax[2].set_ylabel("Median half-width ($\\alpha$ = 0.05)"); ax[2].set_title("(c) Width versus $t$", fontsize=9)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.12))
fig.suptitle("Synthetic coverage study ($K$ = 10, 200 replications): validity of the anytime sequences and their conservatism", fontsize=10, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97]); _save(fig, f"{OUT}/fig_e1_coverage_width")

e3 = J("results/e3/e3_results.json")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
s = e3["B_floor_binding_M1"]["sets"]; xs = np.arange(len(s))
ax[0].bar(xs - 0.19, [q["width_conservative"] for q in s], 0.38, color=OI["grey"], label="Ratio-only set ($\\Gamma$ = 2)")
ax[0].bar(xs + 0.19, [q["width_sharp"] for q in s], 0.38, color=OI["blue"], label="Sharp set ($\\Gamma_{\\rm eff}$ = 1.25, $p_{\\rm lo}$ = 0.8)")
ax[0].set_xticks(xs, [f"$a$ = {q['a_obs']:.2f}" for q in s]); ax[0].set_ylabel("Width of identified interval for $A_j$"); ax[0].set_ylim(0, 0.42); ax[0].legend(fontsize=8, loc="upper right")
ax[0].set_title("(a) Floor-binding case $\\Gamma$ = 2, $p_{\\rm lo}$ = 0.8", fontsize=9)
C = e3["C_finite_sample_composed_coverage"]
lab = {"interior": "interior shares", "vertex_A_low": "attaining configuration (lower end)", "vertex_A_high": "attaining configuration (upper end)"}
rows = []
for k, v in C.items():
    if k.startswith("linear|") and "negctl" not in k:
        sub, kind = k.split("|")[1], k.split("|")[2]
        rows.append((lab[sub], "Sharp set" if kind.startswith("sharp") else "Ratio-only set", v["coverage_n=20000"]))
neg = [v for k, v in C.items() if k.startswith("linear|negctl")][0]
yl, yy, cols = [], [], []
y = 0
for sub in ("interior shares", "attaining configuration (lower end)", "attaining configuration (upper end)"):
    for kind, c in (("Sharp set", OI["blue"]), ("Ratio-only set", OI["grey"])):
        v = [r_[2] for r_ in rows if r_[0] == sub and r_[1] == kind][0]; ax[1].barh(y, v, color=c, label=kind if sub == "interior shares" else None); yl.append(f"{sub}, {kind.lower()}"); y += 1
ax[1].barh(y, neg["coverage_n=20000"], color=OI["orange"], label="Negative control ($\\Gamma_{\\rm used}$ = 1, true ratio 2)"); yl.append("negative control"); y += 1
ax[1].set_yticks(range(len(yl)), yl, fontsize=7); ax[1].invert_yaxis(); ax[1].axvline(0.95, color="k", ls=":", lw=1); ax[1].set_xlim(0, 1.05)
ax[1].set_xlabel("Coverage of the true $A_j$ at $n$ = 20,000"); ax[1].set_title("(b) Finite-sample composed intervals (linear)", fontsize=9)
ax[1].text(0.02, y - 1, "coverage 0.000", va="center", fontsize=8, color=OI["orange"])
h, l = ax[1].get_legend_handles_labels(); fig.legend(h, l, ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.04))
fig.suptitle("Identified-set verification (synthetic): sharp versus ratio-only intervals", fontsize=10); fig.tight_layout(rect=[0, 0.05, 1, 0.95]); _save(fig, f"{OUT}/fig_e3_identification")

e5 = J("results/e5/e5_results.json"); ref = J("results/config_runs/A/e5/e5_results.json")
fig, ax = plt.subplots(1, 3, figsize=(12, 4.0), sharey=True)
cols = {"linear": OI["blue"], "time_decay": OI["green"], "last_touch": OI["orange"], "logistic_shap": OI["purple"]}
for a, G in zip(ax, ["1.0", "1.25", "1.5"]):
    for m, res in e5["models"].items():
        f = res["gammas"][G]; xs_ = sorted(int(c) for c in f["current_certified_fraction"])
        if m == "last_touch": continue            # coincides with logistic+SHAP on this construction
        a.plot(xs_, [f["current_certified_fraction"][str(c)] for c in xs_], color=cols[m], ls="-" if m in GUAR else "--", lw=1.8,
               label=("Linear (guaranteed)" if m == "linear" else "Time decay (guaranteed)" if m == "time_decay" else "Last touch and logistic + SHAP (empirical; coincide)"))
    f = ref["models"]["linear"]["gammas"][G]; xs_ = sorted(int(c) for c in f["current_certified_fraction"])
    a.plot(xs_, [f["current_certified_fraction"][str(c)] for c in xs_], color="#999999", lw=1.2, label="Linear, ratio-box rule with simpler boundary (reference)")
    a.set_title(f"$\\Gamma$ = {G}", fontsize=10); a.set_xlabel("Conversions"); a.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax[0].set_ylabel("Fraction of the 190 campaign pairs certified")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.1))
fig.suptitle("Certified fraction over the monitored window (Criteo, tracked 20-campaign pool, $\\delta$ = 0.005; certificates re-evaluated at every checkpoint)", fontsize=9.5, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.96]); _save(fig, f"{OUT}/fig_e5_certified_fraction")
print("polished figures written")
