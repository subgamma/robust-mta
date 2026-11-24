"""Controlled A/B/C/D comparison on identical streams and seeds.

  A legacy ratio box + simplified boundary (reference) | B hybrid + simplified boundary |
  C legacy ratio box + exact boundary | D hybrid + exact boundary (final)
For a metric f: rule effect = f(B)-f(A); boundary effect = f(C)-f(A);
interaction = f(D)-f(A)-(f(B)-f(A))-(f(C)-f(A)).
Reads results/config_runs/{A,B,C}/<exp>/ and the D outputs in results/<exp>/ (also in config_runs/D for e5/e6).
Writes results/config_runs/comparison.json and a markdown table to stdout.
"""
import json, os, sys
import numpy as np
C = "ABCD"
def path(c, exp, fn):
    p = f"results/config_runs/{c}/{exp}/{fn}"
    if c == "D" and not os.path.exists(p): p = f"results/{exp}/{fn}"
    return p
def load(c, exp, fn): 
    p = path(c, exp, fn)
    return json.load(open(p)) if os.path.exists(p) else None
out = {}
def dec(v):  # v: dict config->value
    a, b, c_, d = v["A"], v["B"], v["C"], v["D"]
    return {"A": a, "B": b, "C": c_, "D": d, "hybrid_effect(B-A)": b - a, "boundary_effect(C-A)": c_ - a, "interaction": d - a - (b - a) - (c_ - a), "total(D-A)": d - a}

md = ["### E5 (Criteo primary, tracked pool, K=20, delta=0.005): guaranteed models", "| model | Gamma | metric | A | B | C | D | hybrid (B-A) | boundary (C-A) | interaction | total (D-A) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
e5 = {c: load(c, "e5", "e5_results.json") for c in C}
out["e5"] = {}
for m in ("linear", "time_decay"):
    for G in ("1.0", "1.25", "1.5"):
        for name, f in (("certified pairs @day30 (of 190)", lambda g: g["n_current_certified"]), ("median t* (conversions)", lambda g: g["median_t_first_certified"]),
                        ("withdrawn pairs", lambda g: g["n_withdrawn"])):
            vals = {c: f(e5[c]["models"][m]["gammas"][G]) for c in C}
            if any(v is None for v in vals.values()): continue
            d = dec(vals); out["e5"][f"{m}|{G}|{name}"] = d
            md.append(f"| {m} | {G} | {name} | " + " | ".join(f"{d[k]:.0f}" if "median" not in name else f"{d[k]:.0f}" for k in ("A", "B", "C", "D")) + " | " + " | ".join(f"{d[k]:+.0f}" for k in ("hybrid_effect(B-A)", "boundary_effect(C-A)", "interaction", "total(D-A)")) + " |")
md += ["", "### E6 (Gamma*, day-30, linear / time-decay, delta=0.005)", "| model | metric | A | B | C | D |", "|---|---|---|---|---|---|"]
e6 = {c: load(c, "e6", "e6_results.json") for c in C}
out["e6"] = {}
for m in ("linear", "time_decay"):
    g = {c: e6[c]["guaranteed"][m]["gamma_star_by_pair"] for c in C}
    cert = {c: [k for k, v in g[c].items() if v is not None] for c in C}
    common = [k for k in g["A"] if all(g[c][k] is not None for c in C)]
    row = {"n_certified": {c: len(cert[c]) for c in C}, "median_all_certified": {c: float(np.median([g[c][k] for k in cert[c]])) for c in C},
           "n_common_pairs": len(common), "median_common": {c: float(np.median([g[c][k] for k in common])) for c in C},
           "mean_ratio_to_A_on_common": {c: float(np.mean([g[c][k] / g["A"][k] for k in common])) for c in C},
           "max": {c: float(max(g[c][k] for k in cert[c])) for c in C},
           "violations_B_ge_A": int(sum(1 for k in common if g["B"][k] < g["A"][k] - 2e-4)), "violations_D_ge_C": int(sum(1 for k in common if g["D"][k] < g["C"][k] - 2e-4)),
           "violations_C_ge_A": int(sum(1 for k in common if g["C"][k] < g["A"][k] - 2e-4)), "violations_D_ge_B": int(sum(1 for k in common if g["D"][k] < g["B"][k] - 2e-4))}
    out["e6"][m] = row
    md.append(f"| {m} | pairs with Gamma* defined (of 190) | " + " | ".join(str(row["n_certified"][c]) for c in C) + " |")
    md.append(f"| {m} | median Gamma* (all defined) | " + " | ".join(f"{row['median_all_certified'][c]:.3f}" for c in C) + " |")
    md.append(f"| {m} | median Gamma* ({row['n_common_pairs']} pairs defined in all four) | " + " | ".join(f"{row['median_common'][c]:.3f}" for c in C) + " |")
    md.append(f"| {m} | mean Gamma*/Gamma*_A on common pairs | " + " | ".join(f"{row['mean_ratio_to_A_on_common'][c]:.3f}" for c in C) + " |")
    md.append(f"| {m} | max Gamma* | " + " | ".join(f"{row['max'][c]:.2f}" for c in C) + " |")
md += ["", "### E10 (synthetic, N=50,000, 100 reps; same streams): certified-correct and wrong certifications per replication, well-specified rule", "| model | r_true | metric | A | B | C | D |", "|---|---|---|---|---|---|---|"]
e10 = {c: load(c, "e10", "e10_results.json") for c in C}
out["e10"] = {}
for m in ("linear", "time_decay"):
    for r in ("1.25", "1.5", "2.0", "2.5"):
        rn = f"wellspec(Gamma={float(r)})"
        for name, f in (("certified correct", lambda d: d["final_correct"]["mean"]), ("wrong certifications", lambda d: d["final_wrong"]["mean"])):
            vals = {c: f(e10[c]["monitoring"][m][r][rn]) for c in C}
            out["e10"][f"{m}|{r}|{name}"] = dec(vals)
            md.append(f"| {m} | {r} | {name} | " + " | ".join(f"{vals[c]:.2f}" for c in C) + " |")
e7 = {c: load(c, "e7", "e7_results.json") for c in C}
if all(v is not None for v in e7.values()):
    md += ["", "### E7 (Criteo-based stress tests; same seeds; G3 = realistic random-cut, OUT-OF-MODEL)", "| scenario | metric | A | B | C | D |", "|---|---|---|---|---|---|"]
    out["e7"] = {}
    for g in ("G1", "G2", "G3"):
        for k, v in e7["D"][g].items():
            for name in ("naive_wrong_rate_mean", "robust_wrong_rate_mean", "robust_wrong_rate_mean_gamma_used_eq_r", "drr_mean", "drr_mean_at_reference_gamma"):
                if name in v:
                    vals = {c: e7[c][g][k][name] for c in C}
                    out["e7"][f"{g}|{k}|{name}"] = vals
                    md.append(f"| {g} {k} | {name} | " + " | ".join(f"{vals[c]:.4f}" for c in C) + " |")
os.makedirs("results/config_runs", exist_ok=True)
json.dump(out, open("results/config_runs/comparison.json", "w"), indent=1, default=float)
print("\n".join(md))
