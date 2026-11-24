"""Consistency check of the generated numbers and tables.

1. Regenerates every number and table with scripts/build_manuscript_numbers.py and checks that the artifacts
   in results/ and generated_tables/ are identical.
2. Cross-checks registered numbers against a second, independently written artifact (CSV tables or a separate
   JSON), not the JSON the builder read.
Writes results/manuscript_table_audit.csv (one row per checked item) and prints the pass count.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GEN = ROOT / "generated_tables"
rows: list[dict] = []


def add(category, item, expected, found, status, source="", note=""):
    rows.append({"category": category, "item": item, "expected": expected, "found": found, "status": status, "source": source, "note": note})


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
before = {p.name: sha(p) for p in list(GEN.glob("*.tex"))} | {"manuscript_numbers.json": sha(ROOT / "results/manuscript_numbers.json")}
subprocess.run([sys.executable, str(ROOT / "scripts/build_manuscript_numbers.py")], check=True, cwd=ROOT, capture_output=True)
after = {p.name: sha(p) for p in list(GEN.glob("*.tex"))} | {"manuscript_numbers.json": sha(ROOT / "results/manuscript_numbers.json")}
for k in sorted(set(before) | set(after)):
    add("artifact regeneration", k, before.get(k, "absent")[:12], after.get(k, "absent")[:12], "MATCH" if before.get(k) == after.get(k) else "MISMATCH", "scripts/build_manuscript_numbers.py", "committed artifact equals regenerated artifact")
REG = json.loads((ROOT / "results/manuscript_numbers.json").read_text())["numbers"]

def csvrows(p):
    with open(ROOT / p) as f:
        return [r for r in csv.DictReader(l for l in f if not l.startswith("#"))]
def close(a, b, tol): return abs(float(a) - float(b)) <= tol
checks = []
t3 = csvrows("results/e5/table3.csv")
for g in ("1.0", "1.25", "1.5"):
    r = [x for x in t3 if x["model"] == "linear" and float(x["Gamma"]) == float(g)][0]
    checks += [(f"e5.lin.cert.{g}", r["n_current_certified"], 0, "results/e5/table3.csv"), (f"e5.lin.t.{g}", r["median_t_first_certified"], 0.5, "results/e5/table3.csv")]
t6 = csvrows("results/e9/table6_sensitivity.csv")
r = [x for x in t6 if x["model"] == "linear" and float(x["delta"]) == 0.0 and float(x["Gamma"]) == 1.0][0]; checks.append(("e9.lin.0.1.0", r["n_certified"], 0, "results/e9/table6_sensitivity.csv"))
r = [x for x in t6 if x["model"] == "linear" and float(x["delta"]) == 0.005 and float(x["Gamma"]) == 2.0][0]; checks.append(("e9.lin.0.5.2.0", r["n_certified"], 0, "results/e9/table6_sensitivity.csv"))
d6 = csvrows("results/e9/table6_distortion.csv")
checks.append(("e9.qa.lin", max(float(x["abs_diff_q_minus_a"]) for x in d6 if x["model"] == "linear"), 0.0005, "results/e9/table6_distortion.csv"))
t5 = csvrows("results/e8/table5.csv") if False else None
e1c = csvrows("results/e1/table_e1_coverage_width.csv")
r = [x for x in e1c if x["panel"] == "stationary" and x["alpha"] == "0.05" and x["method"] == "csranks_style"][0]; checks.append(("e1.cr.path.0.05", r["pathwise_simultaneous_coverage_every_t"], 0.0005, "results/e1/table_e1_coverage_width.csv"))
r2 = [x for x in e1c if x["panel"] == "stationary" and x["alpha"] == "0.05" and x["method"] == "cs_shared"][0]
checks.append(("e1.mult.stat", float(r2["median_halfwidth_t20000"]) / float([x for x in e1c if x["panel"] == "stationary" and x["alpha"] == "0.05" and x["method"] == "csranks_style"][0]["median_halfwidth_t20000"]), 0.005, "results/e1/table_e1_coverage_width.csv (ratio)"))
d50 = csvrows("results/e10/table_e10_decomposition_N50000.csv")
def s50(r, f): return sum(float(x[f]) for x in d50 if x["rule"] == "wellspec" and float(x["r_true"]) == float(r) and x["model"] in ("linear", "time_decay"))
checks += [("e10.obs.r2.0", s50("2.0", "n_observability_limited"), 0, "results/e10/table_e10_decomposition_N50000.csv"), ("e10.orc.r2.0", s50("2.0", "n_oracle_certifiable"), 0, "results/e10/table_e10_decomposition_N50000.csv"),
           ("e10.act.r2.0", s50("2.0", "n_actionable"), 0, "results/e10/table_e10_decomposition_N50000.csv"), ("e10.act.r2.5", s50("2.5", "n_actionable"), 0, "results/e10/table_e10_decomposition_N50000.csv")]
t4 = csvrows("results/e6/table4_fragility.csv")
e3t = csvrows("results/e3/table_e3_lp_verification.csv")
checks.append(("e3.floor.red", 100 * float([x for x in e3t if x["cell"] == "Gamma=2.0,p_lo=0.8"][0]["mean_width_reduction_sharp_vs_conservative"]), 0.05, "results/e3/table_e3_lp_verification.csv"))
ntot = json.loads((ROOT / "results/e6/e6_results.json").read_text())
checks.append(("e6.lin.n", sum(1 for k, v in ntot["guaranteed"]["linear"]["gamma_star_by_pair"].items() if v is not None), 0, "results/e6/e6_results.json (recount over gamma_star_by_pair)"))
checks.append(("e6.lin.med", sorted(v for v in ntot["guaranteed"]["linear"]["gamma_star_by_pair"].values() if v is not None)[len([1 for v in ntot["guaranteed"]["linear"]["gamma_star_by_pair"].values() if v is not None]) // 2] if False else __import__("numpy").median([v for v in ntot["guaranteed"]["linear"]["gamma_star_by_pair"].values() if v is not None]), 0.005, "results/e6/e6_results.json (recomputed median)"))
e7r = json.loads((ROOT / "results/e7/e7_results.json").read_text())
checks.append(("e7.R0", e7r["n_reference_certified"], 0, "results/e7/e7_results.json"))
for key, val, tol, src in checks:
    exp = REG[key]["value"]
    ok = close(exp, val, tol)
    add("cross-check (independent artifact)", key, REG[key]["display"], val, "MATCH" if ok else "MISMATCH", src, REG[key]["source"] + " :: " + REG[key]["field"])
from anytime_attribution.identification import prop2_interval
lo, hi = prop2_interval(0.4, 2.0)
add("analytic worked example", "Theta_1(2) for a=0.4", "[0.25, 0.571]", f"[{lo:.3f}, {hi:.3f}]", "MATCH" if (round(lo, 3), round(hi, 3)) == (0.25, 0.571) else "MISMATCH", "anytime_attribution/identification.py")
lo, hi = prop2_interval(0.4, 1.25)
add("analytic worked example", "feasible set for A_1, Gamma_eff=1.25", "[0.348, 0.455]", f"[{lo:.3f}, {hi:.3f}]", "MATCH" if (round(lo, 3), round(hi, 3)) == (0.348, 0.455) else "MISMATCH", "anytime_attribution/identification.py")
add("analytic worked example", "weighted-rule ratio (omega=100, p=0.5)", "34", f"{(1 + 100 * 0.5) / (2 - 0.5):.0f}", "MATCH" if round((1 + 100 * 0.5) / 1.5) == 34 else "MISMATCH", "closed form of Theorem 2(iii)")

with open(ROOT / "results/manuscript_table_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["category", "item", "expected", "found", "status", "source", "note"]); w.writeheader(); w.writerows(rows)
from collections import Counter
c = Counter(r["status"] for r in rows)
checked = [r for r in rows if r["status"] in ("MATCH", "MISMATCH", "UNREGISTERED", "DESIGN")]
fails = [r for r in rows if r["status"] in ("MISMATCH", "UNREGISTERED")]
print(dict(c)); print(f"items audited: {len(rows)}; mismatches/unregistered: {len(fails)}")
for r in fails[:40]: print("  FAIL", r["category"], r["item"], r["expected"], r["found"])
sys.exit(1 if fails else 0)
