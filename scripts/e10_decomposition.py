"""Join the E10 population oracle with the finite-sample monitoring outcomes at horizon H.

Final certifications per replication and pair are classified as observability-limited (population sharp lower
gap <= delta), sampling-limited (population gap > delta but not certified at H) or certified, with wrong
certifications and boundary-ambiguous pairs counted separately. Linear and time_decay only.

usage: python scripts/e10_decomposition.py [H=50000] [npz] [out.json]
"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
from anytime_attribution.experiments import e10_multicampaign as E

H = int(sys.argv[1]) if len(sys.argv) > 1 else 50_000
npz = np.load(sys.argv[2] if len(sys.argv) > 2 else "results/e10/e10_results_pair_outcomes.npz")
out_path = sys.argv[3] if len(sys.argv) > 3 else f"results/e10/e10_decomposition_N{H}.json"
orc = json.load(open("results/e10/e10_oracle.json")); DELTA = orc["config"]["delta"]
PAIRS = E.pair_list()
rows_out = {}
for key, c in orc["conditions"].items():
    m, r, rn = key.split("|")
    base = f"{m}|{float(r)}|{'naive(Gamma=1)' if rn == 'naive' else ('misspec(Gamma=1.25)' if rn.startswith('misspec') else f'wellspec(Gamma={float(r)})')}|{H}"
    codes = npz[base + "|codes"]; gjk = npz[base + "|gap_jk"]; gkj = npz[base + "|gap_kj"]
    R = codes.shape[0]
    pr = c["pairs"]
    act = np.array([x["actionable"] for x in pr]); amb = np.array([x["ambiguous"] for x in pr])
    ocert = np.array([x["oracle_certifiable"] for x in pr]); og = np.array([x["oracle_gap"] for x in pr])
    sgn = np.array([1 if (x["oriented"][0] == x["pair"][0]) else -1 for x in pr])           # +1: true direction is j>k
    correct_freq = np.mean(codes == sgn[None, :], axis=0)                                      # certified in the TRUE direction
    wrong_freq = np.mean(codes == -sgn[None, :], axis=0)                                       # certified in the opposite direction
    anywrong_nonact = np.mean((codes != 0) & (~act)[None, :], axis=0)                          # any certification of a non-actionable pair (decision semantics: wrong)
    tg = np.where(sgn[None, :] > 0, gjk, gkj)                                                  # finite-sample hybrid lower gap in the true direction
    clear = act & ~amb
    obs_lim = clear & ~ocert; cert_ok = clear & ocert
    samp_lim_cell = cert_ok & (correct_freq < 0.5)
    d = {"model": m, "r_true": float(r), "rule": rn, "Gamma": c["Gamma"], "Gamma_eff": c["Gamma_eff"], "H": H, "reps": int(R),
         "n_pairs": 45, "n_actionable": int(act.sum()), "n_boundary_ambiguous_actionable": int((act & amb).sum()),
         "n_observability_limited": int(obs_lim.sum()), "frac_observability_limited": float(obs_lim.sum() / max(act.sum(), 1)),
         "n_oracle_certifiable": int(cert_ok.sum()), "frac_oracle_certifiable": float(cert_ok.sum() / max(act.sum(), 1)),
         "mean_n_currently_certified_correct_actionable": float(correct_freq[act].sum()), "frac_currently_certified_correct_actionable": float(correct_freq[act].mean()),
         "mean_n_certified_correct_among_oracle_certifiable": float(correct_freq[cert_ok].sum()),
         "mean_n_certified_correct_among_observability_limited": float(correct_freq[obs_lim].sum()),
         "mean_n_sampling_limited_(oracle-certifiable, uncertified)": float((1 - correct_freq[cert_ok]).sum()),
         "n_cells_sampling_limited_(oracle-cert, certified in <50% reps)": int(samp_lim_cell.sum()),
         "mean_wrong_direction_certifications_actionable": float(wrong_freq[act].sum()),
         "mean_certifications_of_nonactionable_pairs": float(anywrong_nonact.sum()),
         "mean_total_wrong_certifications_per_rep": float(wrong_freq[act].sum() + anywrong_nonact[~act].sum()),
         "oracle_margin_(gap-delta)_actionable_quantiles": {q: float(np.quantile(og[act] - DELTA, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)} if act.any() else {},
         "mean_finite_sample_shortfall_oracle_minus_hybrid_gap_(oracle-certifiable)": float(np.mean(og[cert_ok] - tg.mean(0)[cert_ok])) if cert_ok.any() else None,
         "mean_finite_sample_shortfall_all_actionable": float(np.mean(og[act] - tg.mean(0)[act])) if act.any() else None}
    rows_out[key] = d
tot = {}
for r in E.R_TRUE_GRID:
    for m in ("linear", "time_decay"):
        tot[f"{m}|{r}"] = rows_out[f"{m}|{r}|wellspec"]["n_cells_sampling_limited_(oracle-cert, certified in <50% reps)"]
n_samp = int(sum(tot.values()))
res = {"H": H, "delta": DELTA, "rows": rows_out, "decision_rule": {"threshold_cells": 10, "n_sampling_limited_cells_wellspec": n_samp, "by_model_r": tot, "run_250k": bool(n_samp >= 10),
       "rule_file": "decision rule fixed before any final E10 outcome was inspected"}}
json.dump(res, open(out_path, "w"), indent=1)
print(f"H={H}: sampling-limited well-specified cells = {n_samp} (threshold 10) -> run_250k = {n_samp >= 10}")
for key, d in rows_out.items():
    if key.endswith("wellspec"):
        print(f"{key:22s} act={d['n_actionable']} obs-lim={d['n_observability_limited']} oracle-cert={d['n_oracle_certifiable']} amb={d['n_boundary_ambiguous_actionable']} "
              f"cert@H={d['mean_n_currently_certified_correct_actionable']:.2f} samp-lim={d['mean_n_sampling_limited_(oracle-certifiable, uncertified)']:.2f} wrong/rep={d['mean_total_wrong_certifications_per_rep']:.2f}")
