"""Figures for the paper: colour-blind-safe (Okabe-Ito) palette; guaranteed (fixed-weight, theorem-covered) results
solid, empirical dashed; synthetic studies (E1, E3, E10) labelled as such.
Each function reads saved result JSON and writes PDF and PNG.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OI = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73", "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#E69F00", "grey": "#7F7F7F", "black": "#000000"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#E3E3E3", "grid.linewidth": 0.6,
                     "figure.dpi": 120, "savefig.bbox": "tight", "legend.frameon": False})
MODEL_LABEL = {"linear": "Linear", "time_decay": "Time decay", "last_touch": "Last touch", "logistic_shap": "Logistic + SHAP"}
GUAR = {"linear", "time_decay"}


def _save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf"); fig.savefig(f"{out}.png", dpi=170); plt.close(fig)


def _last(d):
    k = max(d, key=lambda x: float(x)); return d[k]


def _J(p):
    return json.load(open(p))


def fig_e1(e1: dict, out: str):
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6))
    alphas = ["0.05", "0.2", "0.5"]; x = np.arange(3)
    meths = [("cs_shared", "Anytime-valid CS (shared)", OI["blue"]), ("cs_single", "Anytime-valid CS (single, no multiplicity)", OI["sky"]),
             ("csranks_style", "Fixed-sample, Bonferroni (peeked at every t)", OI["orange"]), ("bootstrap_bonf", "Bootstrap, Bonferroni (peeked)", OI["yellow"])]
    for i, (m, lab, c) in enumerate(meths):
        v = [e1["stationary"][a][m].get("pathwise_simultaneous_coverage", np.nan) for a in alphas]
        ax[0].bar(x + (i - 1.5) * 0.2, v, 0.2, color=c, label=lab)
    for xi, a in zip(x, alphas):
        ax[0].plot([xi - 0.45, xi + 0.45], [1 - float(a)] * 2, color="k", lw=1, ls=":")
    ax[0].set_xticks(x, [f"α={a}" for a in alphas]); ax[0].set_ylabel("Simultaneous coverage at every t ∈ [500, 20,000]"); ax[0].set_title("(a) Pathwise coverage (dotted: nominal 1−α)"); ax[0].set_ylim(0, 1.05)
    ax[0].legend(fontsize=6.5, loc="lower left")
    for i, (m, lab, c) in enumerate([("cs_shared", "Anytime CS (shared)", OI["blue"]), ("csranks_style", "Fixed-sample (Bonferroni, one horizon)", OI["orange"]), ("bootstrap_bonf", "Bootstrap (Bonferroni)", OI["yellow"])]):
        v = [e1["stationary"][a][m]["horizons"]["20000"]["halfwidth_median"] for a in alphas]
        ax[1].bar(x + (i - 1) * 0.27, v, 0.27, color=c, label=lab)
    ax[1].set_xticks(x, [f"α={a}" for a in alphas]); ax[1].set_ylabel("Median interval half-width at t = 20,000"); ax[1].set_title("(b) Width cost of anytime validity"); ax[1].legend(fontsize=7)
    for m, lab, c, ls in [("cs_shared", "Anytime CS (shared)", OI["blue"], "-"), ("csranks_style", "Fixed-sample (Bonferroni)", OI["orange"], "--")]:
        d = e1["stationary"]["0.05"][m]["halfwidth_median_vs_t"] if "halfwidth_median_vs_t" in e1["stationary"]["0.05"][m] else None
        if d:
            ts = sorted(int(t) for t in d); ax[2].loglog(ts, [d[str(t)] for t in ts], color=c, ls=ls, label=lab)
    ax[2].set_xlabel("Conversions t"); ax[2].set_ylabel("Median half-width (α = 0.05)"); ax[2].set_title("(c) Width versus t"); ax[2].legend(fontsize=7)
    fig.suptitle("Synthetic coverage study (200 replications, K = 10): validity of the anytime sequences, and their conservatism", fontsize=9.5, y=1.02)
    _save(fig, out)


def fig_e2(e2: dict, out: str):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = [("bootstrap", "Bootstrap"), ("fixed_eb", "Fixed empirical-Bernstein"), ("csranks_style", "Rank-CI (Bonferroni)"), ("naive_cs", "Anytime CS (pair budget)"), ("robust_cs", "Anytime CS (campaign budget)")]
    v = [e2["methods" if "methods" in e2 else list(e2)[0]][k]["wrong_rate_overall"] if "methods" in e2 else e2[k]["wrong_rate_overall"] for k, _ in names]
    cols = [OI["orange"]] * 3 + [OI["blue"]] * 2
    ax.bar(range(5), v, color=cols); ax.axhline(0.05, color="k", ls=":", lw=1)
    ax.set_xticks(range(5), [n for _, n in names], rotation=18, ha="right"); ax.set_ylabel("Probability of ever certifying an ordering\n(true tie; every certification is wrong)")
    ax.set_title("Continuous monitoring under a true tie (synthetic)\nfixed-sample intervals peeked at every look (orange) vs anytime-valid sequences (blue); dotted: α = 0.05")
    _save(fig, out)


def fig_e3(e3: dict, out: str):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    s = e3["B_floor_binding_M1"]["sets"]
    lab = [f"a = {x['a_obs']:.2f}" for x in s]; xs = np.arange(len(s))
    ax[0].bar(xs - 0.18, [x["width_conservative"] for x in s], 0.36, color=OI["grey"], label="Ratio-only set (Γ = 2)")
    ax[0].bar(xs + 0.18, [x["width_sharp"] for x in s], 0.36, color=OI["blue"], label="Sharp set (Γ_eff = 1.25; p_lo = 0.8)")
    ax[0].set_xticks(xs, lab); ax[0].set_ylabel("Width of identified interval for A_j"); ax[0].legend(fontsize=7)
    ax[0].set_title("(a) Floor-binding case Γ = 2, p_lo = 0.8")
    C = e3["C_finite_sample_composed_coverage"]
    keys = [k for k in C if k.startswith("linear|") and "sharp" in k]
    for k in keys:
        n = [kk for kk in C[k] if kk.startswith("coverage_n=")]
    names = []; vals = []
    for k in C:
        if k.startswith("linear|"):
            names.append(k.replace("linear|", "").replace("_Gamma", " Γ").replace("sharp", "sharp").replace("conservative", "ratio-only")); vals.append(C[k]["coverage_n=20000"])
    ax[1].barh(range(len(vals)), vals, color=[OI["blue"] if "sharp" in n else OI["grey"] for n in names]); ax[1].set_yticks(range(len(vals)), names, fontsize=7)
    ax[1].axvline(0.95, color="k", ls=":", lw=1); ax[1].set_xlim(0.9, 1.005); ax[1].set_xlabel("Coverage of the true A_j at n = 20,000"); ax[1].set_title("(b) Finite-sample composed intervals (linear)")
    fig.suptitle("Identified-set verification (synthetic): sharp versus ratio-only intervals", fontsize=9.5, y=1.02)
    _save(fig, out)


def fig_e4(syn: dict, cri: dict, out: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    rs = [float(r) for r in syn["results"]]
    ax[0].plot([r - 0.02 for r in rs], [_last(syn["results"][str(r)]["naive_wrong_rate"]) for r in syn["results"]], "o-", color=OI["orange"], label="Naive (Γ = 1)")
    ax[0].plot([r + 0.02 for r in rs], [_last(syn["results"][str(r)]["robust_wrong_rate"]) for r in syn["results"]], "s-", color=OI["blue"], label="Robust (Γ = r)")
    ms = syn["misspecified_r_true_3"]["by_gamma_used"]; g = sorted(ms, key=float)
    ax[0].plot([3.0] * len(g), [_last(ms[k]["robust_wrong_rate"]) for k in g], "^", color=OI["purple"], label="Robust, Γ < r (misspecified)")
    ax[0].set_title("(a) Constructed mechanism (synthetic)"); ax[0].set_xlabel("True retention ratio r"); ax[0].set_ylabel("Probability of a wrong certification at final N"); ax[0].legend(fontsize=7)
    rc = [float(r) for r in cri["results"]]
    ax[1].plot([r - 0.02 for r in rc], [_last(cri["results"][str(r)]["naive_wrong_rate"]) for r in cri["results"]], "o-", color=OI["orange"], label="Naive (Γ = 1)")
    ax[1].plot([r + 0.02 for r in rc], [_last(cri["results"][str(r)]["robust_wrong_rate"]) for r in cri["results"]], "s-", color=OI["blue"], label="Robust (Γ = r)")
    ax[1].set_title("(b) Targeted worst-case Criteo pair (not a prevalence estimate)"); ax[1].set_xlabel("Imposed retention ratio r (campaign j thinned)"); ax[1].legend(fontsize=7)
    for a in ax: a.set_ylim(-0.03, 1.03)
    _save(fig, out)


def fig_e5(e5: dict, e5_ref: dict | None, out: str):
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    cols = {"linear": OI["blue"], "time_decay": OI["green"], "last_touch": OI["orange"], "logistic_shap": OI["purple"]}
    for a, G in zip(ax, ["1.0", "1.25", "1.5"]):
        for m, res in e5["models"].items():
            f = res["gammas"].get(G)
            if f is None: continue
            xs = sorted(int(c) for c in f["current_certified_fraction"]); ys = [f["current_certified_fraction"][str(c)] for c in xs]
            a.plot(xs, ys, color=cols[m], ls="-" if m in GUAR else "--", lw=1.8, label=f"{MODEL_LABEL[m]} ({'guaranteed' if m in GUAR else 'empirical'})")
        if e5_ref:
            f = e5_ref["models"]["linear"]["gammas"][G]; xs = sorted(int(c) for c in f["current_certified_fraction"])
            a.plot(xs, [f["current_certified_fraction"][str(c)] for c in xs], color="#999999", lw=1.2, ls="-", label="Linear, legacy ratio-box rule + simplified boundary (reference)")
        a.set_title(f"Γ = {G}"); a.set_xlabel("Conversions")
    ax[0].set_ylabel("Fraction of the 190 campaign pairs certified"); ax[0].legend(fontsize=6.5, loc="upper left")
    ax[1].text(0.98, 0.04, "Last touch and logistic + SHAP coincide on single-campaign journeys (one curve).", transform=ax[1].transAxes, ha="right", fontsize=6.5)
    fig.suptitle("Certified fraction over the monitored window (Criteo, tracked 20-campaign pool, δ = 0.005; certifications re-evaluated at every checkpoint)", fontsize=9, y=1.02)
    _save(fig, out)


def fig_e6(e6: dict, out: str):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for m, c in [("linear", OI["blue"]), ("time_decay", OI["green"])]:
        v = [g for g in e6["guaranteed"][m]["gamma_star_by_pair"].values() if g is not None]
        ax[0].hist(v, bins=np.linspace(1, 5, 33), alpha=0.55, color=c, label=f"{MODEL_LABEL[m]} (n = {len(v)} of 190)")
    ax[0].axvline(1 / e6["p_lo"], color="k", ls=":", lw=1); ax[0].text(1 / e6["p_lo"], ax[0].get_ylim()[1] * 0.92, " 1/p_lo", fontsize=7)
    ax[0].set_xlabel("Robustness radius Γ*"); ax[0].set_ylabel("Pairs"); ax[0].set_title("(a) Guaranteed radius (hybrid rule)"); ax[0].legend(fontsize=7)
    sens = e6["empirical_primary"]["gamma_hat_sens_by_pair"]
    for G, c in [("1.25", OI["sky"]), ("1.5", OI["orange"])]:
        v = [x[G] for x in sens.values() if G in x]
        ax[1].hist(v, bins=30, alpha=0.55, color=c, label=f"re-thinning ratio {G}")
    ax[1].set_xlabel("|shift in credit difference| under re-thinning"); ax[1].set_title("(b) Empirical sensitivity, last touch (no guarantee)"); ax[1].legend(fontsize=7)
    _save(fig, out)


def fig_e7(e7: dict, out: str):
    fig, ax = plt.subplots(1, 4, figsize=(14, 3.5))
    spec = [("G1", "Uniform loss ρ", "naive_wrong_rate_mean", "robust_wrong_rate_mean", "drr_mean", "Fraction of touches lost"),
            ("G2", "Differential observability r", "naive_wrong_rate_mean", "robust_wrong_rate_mean_gamma_used_eq_r", "drr_mean_at_reference_gamma", "Retention ratio r"),
            ("G3", "Identity fragmentation (out of model)", "naive_wrong_rate_mean", "robust_wrong_rate_mean", "drr_mean", "Fragmentation probability ρ")]
    for a, (g, title, kn, kr, kd, xl) in zip(ax[:3], spec):
        xs = sorted(e7[g], key=float); xf = [float(x) for x in xs]
        a.plot(xf, [e7[g][x][kn] for x in xs], "o-", color=OI["orange"], label="Naive wrong-certification rate")
        a.plot(xf, [e7[g][x][kr] for x in xs], "s-", color=OI["blue"], label="Robust wrong-certification rate")
        a.plot(xf, [e7[g][x][kd] for x in xs], "^--", color=OI["green"], label="Decision-retention rate (DRR)")
        a.set_title(title, fontsize=9); a.set_xlabel(xl); a.set_ylim(-0.03, 1.03)
    ax[0].set_ylabel("Rate"); ax[0].legend(fontsize=6.5, loc="center left")
    gt = e7["G2_targeted"]["settings"]; labs = list(gt)
    xx = np.arange(len(labs))
    ax[3].bar(xx - 0.27, [gt[l]["naive_wrong_rate"] for l in labs], 0.27, color=OI["orange"], label="Naive: wrong")
    ax[3].bar(xx, [gt[l]["robust_wrong_rate"] for l in labs], 0.27, color=OI["blue"], label="Robust: wrong")
    ax[3].bar(xx + 0.27, [gt[l]["robust_unresolved_rate"] for l in labs], 0.27, color=OI["grey"], label="Robust: unresolved")
    ax[3].set_xticks(xx, [f"r = {gt[l]['r']:.2f}\n({'below' if l == 'r_low' else 'above'} observed ratio)" for l in labs], fontsize=7)
    ax[3].set_title(f"Targeted worst-case pair {e7['G2_targeted']['pair']}", fontsize=9); ax[3].set_ylim(0, 1.03); ax[3].legend(fontsize=7)
    fig.suptitle("Criteo-based stress tests of the certified graph (G3 is outside the retention model: no coverage guarantee is claimed)", fontsize=9.5, y=1.03)
    _save(fig, out)


def fig_drift(checkpoints, traj, ids, out):
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    for i, (c, ys) in enumerate(traj.items()):
        ax.plot(checkpoints, ys, lw=1.8, color=list(OI.values())[i % 7], label=f"Campaign {c}")
    ax.set_xlabel("Conversions (chronological order)"); ax.set_ylabel("Cumulative observed share (linear)")
    ax.set_title("Cumulative observed share over the monitored window (Criteo): several campaigns are not stationary", fontsize=9); ax.legend(fontsize=7)
    _save(fig, out)


def fig_e10_identifiability(dec: dict, out: str):
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for a, m in zip(ax, ("linear", "time_decay")):
        rs = ["1.25", "1.5", "2.0", "2.5"]; xs = np.arange(4)
        rows = [dec["rows"][f"{m}|{r}|wellspec"] for r in rs]
        obs = np.array([r["n_observability_limited"] for r in rows]); cert = np.array([r["mean_n_certified_correct_among_oracle_certifiable"] for r in rows])
        samp = np.array([r["mean_n_sampling_limited_(oracle-certifiable, uncertified)"] for r in rows]); amb = np.array([r["n_boundary_ambiguous_actionable"] for r in rows])
        a.bar(xs, obs, color=OI["grey"], label="Observability-limited (population gap ≤ δ)")
        a.bar(xs, cert, bottom=obs, color=OI["blue"], label="Population-certifiable, certified at N = 50,000")
        a.bar(xs, samp, bottom=obs + cert, color=OI["sky"], label="Population-certifiable, not yet certified (sampling-limited)")
        a.bar(xs, amb, bottom=obs + cert + samp, color=OI["yellow"], label="Boundary-ambiguous (±3 SE)")
        a.set_xticks(xs, [f"r = {r}" for r in rs]); a.set_title(f"{MODEL_LABEL[m]} (well-specified Γ = r)")
    ax[0].set_ylabel("Actionable pairs (of 45)")
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=2, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Synthetic multi-campaign benchmark: why pairs stay unresolved — population identification vs finite-sample power", fontsize=9.5, y=1.02)
    _save(fig, out)


def fig_e10_vs_N(d50: dict, d250: dict, out: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for a, m in zip(ax, ("linear", "time_decay")):
        rs = ["1.25", "1.5", "2.0", "2.5"]; w = 0.18; xs = np.arange(4)
        for i, (d, lab, hatch) in enumerate([(d50, "N = 50,000", ""), (d250, "N = 250,000", "//")]):
            pc = []; po = []
            for r in rs:
                x = d["rows"][f"{m}|{r}|wellspec"]
                pc.append(x["mean_n_certified_correct_among_oracle_certifiable"] / x["n_oracle_certifiable"] if x["n_oracle_certifiable"] else np.nan)
                po.append(x["mean_n_certified_correct_among_observability_limited"] / x["n_observability_limited"] if x["n_observability_limited"] else np.nan)
            a.bar(xs + (2 * i - 1.5) * w, pc, w, color=OI["blue"], hatch=hatch, edgecolor="white", label=f"Population-certifiable, {lab}")
            a.bar(xs + (2 * i - 0.5) * w, po, w, color=OI["grey"], hatch=hatch, edgecolor="white", label=f"Observability-limited, {lab}")
            for xi, v in zip(xs + (2 * i - 0.5) * w, po):
                if v == 0: a.text(xi, 0.015, "0", ha="center", fontsize=7, color=OI["grey"])
        a.set_xticks(xs, [f"r = {r}" for r in rs]); a.set_title(MODEL_LABEL[m]); a.set_ylim(0, 1.05)
    ax[0].set_ylabel("Fraction of actionable pairs certified correctly")
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=2, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, -0.14))
    fig.suptitle("Matched-stream certification versus N (well-specified Γ = r): more conversions help only population-certifiable pairs", fontsize=9.5, y=1.02)
    _save(fig, out)


def fig_e10_wrong(res: dict, out: str):
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    rs = ["1.25", "1.5", "2.0", "2.5"]
    for a, m in zip(ax, ("linear", "time_decay", "last_touch", "logistic_shap")):
        mon = res["monitoring"][m]
        a.plot(rs, [mon[r]["naive(Gamma=1)"]["final_wrong"]["mean"] for r in rs], "o-", color=OI["orange"], label="Naive (Γ = 1)")
        a.plot(rs, [mon[r][f"wellspec(Gamma={float(r)})"]["final_wrong"]["mean"] for r in rs], "s-", color=OI["blue"], label="Robust, Γ = r")
        mis = [(r, mon[r]["misspec(Gamma=1.25)"]["final_wrong"]["mean"]) for r in rs if "misspec(Gamma=1.25)" in mon[r]]
        a.plot([x for x, _ in mis], [y for _, y in mis], "^--", color=OI["purple"], label="Robust, Γ = 1.25 < r")
        a.set_title(f"{MODEL_LABEL[m]}\n({'theorem-covered' if m in GUAR else 'empirical'})", fontsize=9); a.set_xlabel("True retention ratio r")
    ax[0].set_ylabel("Wrong certifications per replication (of 45 pairs)"); ax[0].legend(fontsize=7)
    fig.suptitle("Synthetic benchmark, N = 50,000: wrong certifications by rule (100 replications)", fontsize=9.5, y=1.03)
    _save(fig, out)


def fig_e10_tradeoff(res: dict, out: str):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    lams = np.linspace(0, 0.3, 61)
    for a, (m, r) in zip(ax, [("linear", "2.0"), ("linear", "2.5"), ("time_decay", "2.0")]):
        mon = res["monitoring"][m][r]; g = lambda d, l: d[f"abstention_adjusted_loss_normalised_lambda={l}"]["mean"]
        for name, key, c in [("Naive (Γ = 1)", "naive(Gamma=1)", OI["orange"]), (f"Robust (Γ = {r})", f"wellspec(Gamma={float(r)})", OI["blue"])]:
            L0, L1 = g(mon[key], 0.0), g(mon[key], 1.0)
            a.plot(lams, L0 + lams * (L1 - L0), color=c, label=name)
        n, w = mon["naive(Gamma=1)"], mon[f"wellspec(Gamma={float(r)})"]
        L0n, L1n, L0w, L1w = g(n, 0.0), g(n, 1.0), g(w, 0.0), g(w, 1.0)
        lam = (L0n - L0w) / ((L1w - L0w) - (L1n - L0n)); a.axvline(lam, color="k", ls=":", lw=1); a.text(lam, a.get_ylim()[1] * 0.9, f" break-even λ* = {lam:.3f}", fontsize=7)
        a.set_title(f"{MODEL_LABEL[m]}, r = {r}"); a.set_xlabel("Deferral cost λ (per unit |Δ| left unresolved)")
    ax[0].set_ylabel("Normalised abstention-adjusted loss"); ax[0].legend(fontsize=7)
    fig.suptitle("Cost of abstention: ranking-loss tradeoff in the synthetic benchmark (not causal budget regret)", fontsize=9.5, y=1.03)
    _save(fig, out)
