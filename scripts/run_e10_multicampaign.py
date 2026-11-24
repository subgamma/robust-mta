"""E10: controlled multi-campaign synthetic benchmark runner.

See anytime_attribution/experiments/e10_multicampaign.py for the generator. Not observational evidence; shares
are model credit shares, not causal effects.
Output: results/e10/e10_results.json (and a pilot variant).
Run: python scripts/run_e10_multicampaign.py [--pilot] [--n-mon 50000] [--reps 40]
Certification: certify_pair(method="hybrid") on one shared state (K+1 processes, beta = alpha/(K+1)),
alpha = 0.05, delta = 0.005, for all 45 pairs at geometric checkpoints.
"""
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from scipy.stats import beta as _beta

sys.path.insert(0, ".")
from anytime_attribution.certify import BoxSet, Certification, box_lower_gap
from anytime_attribution.cs_vectorized import cumulative_share_boxes
from anytime_attribution.experiments import e10_multicampaign as E
from anytime_attribution.identification import prop7_envelope

ALPHA, DELTA, SEED = 0.05, 0.005, 20263
K = E.K
PAIRS = E.pair_list()
WARM_M = 600_000
POP_N = 1_000_000
CFG = {"A": ("legacy_ratio_box", "simplified"), "B": ("hybrid", "simplified"), "C": ("legacy_ratio_box", "exact"), "D": ("hybrid", "exact")}
BASE_HORIZON = 50_000


class BoxesCS:
    """Adapter exposing the two methods `certify_pair` reads from a `MultiCampaignCumulativeShareCS` (numerator_boxes,
    intervals).

    Backed by the vectorised boxes of one shared state (K numerators + denominator, beta = alpha/(K+1)) at time t.
    Equality with the production class is tested in tests/test_e10_multicampaign.py.
    """
    untracked_mass_seen = False
    def __init__(self, bx, t): self.LN, self.UN, self.L, self.U = bx["LN"][t - 1], bx["UN"][t - 1], bx["L"][t - 1], bx["U"][t - 1]
    def numerator_boxes(self): return self.LN, self.UN
    def intervals(self): return self.L, self.U


def rules_for(r):
    rules = {"naive(Gamma=1)": 1.0, f"wellspec(Gamma={r})": r}
    if r in (2.0, 2.5):
        rules["misspec(Gamma=1.25)"] = 1.25
    return rules


def cp(k, n):
    lo = 0.0 if k == 0 else _beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else _beta.ppf(0.975, k + 1, n - k)
    return [float(lo), float(hi)]


def delta_dict(A):
    return {p: float(A[p[0]] - A[p[1]]) for p in PAIRS}


def mc_summary(trip, N, p, shap_model):
    """Population-level (large latent sample) A, Q (latent) and a, q (observed) for every model."""
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 7]))
    keep = E.thin(trip, p, rng)
    out = {}
    for m in E.MODELS:
        S, _ = E.model_masses(m, trip, None, N, shap_model); T, _ = E.model_masses(m, trip, keep, N, shap_model)
        ST = S.sum(1); TT = T.sum(1)
        A = S.sum(0) / ST.sum(); a = T.sum(0) / TT.sum()
        Q = (S[ST > 0] / ST[ST > 0, None]).mean(0); q = (T[TT > 0] / TT[TT > 0, None]).mean(0)
        out[m] = {"A": A, "a": a, "Q": Q, "q": q, "frac_zero_mass_observed": float((TT == 0).mean()), "S": S if m == "linear" else None}
    return out


def checkpoints_for(N, horizons):
    """Nested-stream checkpoints: the 40 geometric points to 50,000 (plus the horizons) do not depend on N, so the N =
    250,000 run read at t = 50,000 is the N = 50,000 run.
    """
    base = set(np.geomspace(250, BASE_HORIZON, 40).astype(int).tolist()) | {BASE_HORIZON}
    if N > BASE_HORIZON:
        base |= set(np.geomspace(BASE_HORIZON, N, 12).astype(int).tolist())
    return sorted(t for t in (base | set(horizons) | {N}) if t <= N)


def one_rep(args):
    rep, N, A_pop, shap_state, horizons, cfg = args
    cert_method, boundary = CFG[cfg]
    shap_model = E.LinearSHAPModel(); shap_model.beta, shap_model.mu = shap_state
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3, rep]))
    trip, rate = E.sample_converting(N, rng)
    cps = checkpoints_for(N, horizons)
    res = {}
    for r in E.R_TRUE_GRID:
        p = E.retention_vector(r)
        keep = E.thin(trip, p, np.random.default_rng(np.random.SeedSequence([SEED, 4, rep, int(r * 100)])))
        for m in E.MODELS:
            Tj, B = E.model_masses(m, trip, keep, N, shap_model)
            T = Tj.sum(1)
            bx = cumulative_share_boxes(Tj, T, B, ALPHA / (K + 1) / 2, form=boundary)
            Delta = delta_dict(A_pop[m])
            for rname, G in rules_for(r).items():
                first = {}; ever_wrong = False; by_h = {}
                for t in cps:
                    bs = BoxSet(bx["L"][t - 1], bx["U"][t - 1], bx["LN"][t - 1], bx["UN"][t - 1])
                    gjk = np.empty(len(PAIRS)); gkj = np.empty(len(PAIRS)); cert = {}
                    for i, (j, k) in enumerate(PAIRS):
                        gjk[i] = box_lower_gap(bs, G, j, k, method=cert_method, delta=DELTA)
                        gkj[i] = box_lower_gap(bs, G, k, j, method=cert_method, delta=DELTA)
                        c = "j>k" if gjk[i] > DELTA else ("k>j" if gkj[i] > DELTA else "unres")
                        cert[(j, k)] = c
                        if c != "unres":
                            first.setdefault((j, k), t)
                            d = Delta[(j, k)]
                            if (c == "j>k" and d <= DELTA) or (c == "k>j" and d >= -DELTA): ever_wrong = True
                    if t in horizons:
                        absD, act, corr, wrong, unres = E.decision_outcomes(cert, Delta, DELTA)
                        code = np.array([1 if cert[p_] == "j>k" else (-1 if cert[p_] == "k>j" else 0) for p_ in PAIRS], dtype=np.int8)
                        by_h[t] = {"loss": E.loss_summary(absD, act, corr, wrong, unres), "n_correct": int(corr.sum()), "n_wrong": int(wrong.sum()), "n_unres": int(unres.sum()),
                                   "ever_wrong": bool(ever_wrong), "first_cert": {p_: first[p_] for p_ in first}, "first_cert_correct_median": None,
                                   "codes": code.tolist(), "gap_jk": gjk.tolist(), "gap_kj": gkj.tolist()}
                res[(m, r, rname)] = by_h
    return rep, rate, res


def population_block(shap_model):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2]))
    trip, rate = E.sample_converting(POP_N, rng)
    N = trip["N"]; n = trip["n"]; jid, camp = trip["jid"], trip["camp"]
    u = np.unique(jid * K + camp); nd = np.bincount(u // K, minlength=N)
    X = E.counts_matrix(jid, camp, None, N); A_lin = X.sum(0) / X.sum()
    comp = {"n_converting_journeys_pop_sample": N, "candidate_conversion_rate": rate, "frac_ge2_distinct_campaigns": float((nd >= 2).mean()), "frac_ge3_distinct_campaigns": float((nd >= 3).mean()),
            "journey_length_quantiles(latent, capped at 10)": {q: float(np.quantile(n, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.99)}, "journey_length_mean": float(n.mean()), "frac_length_1": float((n == 1).mean()),
            "linear_share_A": A_lin.tolist(), "effective_K_inverse_simpson": float(1 / (A_lin ** 2).sum()), "effective_K_exp_entropy": float(np.exp(-(A_lin * np.log(A_lin)).sum())),
            "top2_share": float(np.sort(A_lin)[-2:].sum()), "top5_share": float(np.sort(A_lin)[-5:].sum()), "top10_share": float(A_lin.sum()),
            "role_shares_of_touches": {E.ROLE_NAMES[i]: float(A_lin[E.ROLE_OF == i].sum()) for i in range(3)}}
    # supported pairs: >= 100 co-occurring converting journeys in a 50,000-conversion stream (scaled from the
    # population sample)
    pres = np.zeros((N, K), dtype=bool); pres[jid, camp] = True
    co = pres.T.astype(np.float64) @ pres.astype(np.float64)
    scale = 50_000 / N
    comp["supported_pairs_ge100_cooccurring_journeys_per_50k"] = int(sum(co[j, k] * scale >= 100 for (j, k) in PAIRS))
    comp["min_pair_cooccurrence_per_50k"] = float(min(co[j, k] for (j, k) in PAIRS) * scale)
    comp["median_pair_cooccurrence_per_50k"] = float(np.median([co[j, k] for (j, k) in PAIRS]) * scale)
    S = {m: E.model_masses(m, trip, None, N, shap_model)[0] for m in E.MODELS}
    A_pop = {m: S[m].sum(0) / S[m].sum() for m in E.MODELS}
    Qlat = {m: (S[m] / S[m].sum(1, keepdims=True)).mean(0) for m in E.MODELS}
    # Q_j - A_j = -Cov(S, Z_j)/E[S] (linear).
    Sl = S["linear"].sum(1); Z = S["linear"] / Sl[:, None]
    cov = ((Sl - Sl.mean())[:, None] * (Z - Z.mean(0))).mean(0)
    prop6 = {"max_abs_error": float(np.max(np.abs((Qlat["linear"] - A_pop["linear"]) - (-cov / Sl.mean())))), "Q_minus_A_linear": (Qlat["linear"] - A_pop["linear"]).tolist()}
    abs_QA = {m: {"per_campaign": np.abs(Qlat[m] - A_pop[m]).tolist(), "max": float(np.max(np.abs(Qlat[m] - A_pop[m]))), "mean": float(np.mean(np.abs(Qlat[m] - A_pop[m])))} for m in E.MODELS}
    sub = slice(0, 200_000)
    Zs = {m: S[m][sub] / S[m][sub].sum(1, keepdims=True) for m in E.MODELS}
    l1 = {f"{a}|{b}": float(np.abs(Zs[a] - Zs[b]).sum(1).mean()) for a, b in __import__("itertools").combinations(E.MODELS, 2)}
    rank_dis = {}
    for a, b in __import__("itertools").combinations(E.MODELS, 2):
        d = [np.sign(A_pop[a][j] - A_pop[a][k]) != np.sign(A_pop[b][j] - A_pop[b][k]) for (j, k) in PAIRS]
        rank_dis[f"{a}|{b}"] = {"pairs_changing_order": int(np.sum(d)), "fraction": float(np.mean(d))}
    from scipy.stats import kendalltau
    kt = {f"{a}|{b}": float(kendalltau(A_pop[a], A_pop[b]).statistic) for a, b in __import__("itertools").combinations(E.MODELS, 2)}
    agg_diff = {f"{a}|{b}": {"L1": float(np.abs(A_pop[a] - A_pop[b]).sum()), "max_abs": float(np.abs(A_pop[a] - A_pop[b]).max())} for a, b in __import__("itertools").combinations(E.MODELS, 2)}
    obs = {}
    for r in E.R_TRUE_GRID:
        p = E.retention_vector(r); chk = E.check_retention(p, r, p.min())
        mc = mc_summary(trip, N, p, shap_model)
        d = {"retention_vector": p.round(4).tolist(), "retention_checks": chk, "p_lo_declared(=min p)": float(p.min())}
        # Linear, w=1, independent retention, n_min=1: Q_j in [(s_low/Gamma_eff) q_j, q_j/p_lo] with s_low = p_lo.
        plo = float(p.min()); lo_hi = [prop7_envelope(float(mc["linear"]["q"][j]), r, plo, n_min=1) for j in range(K)]
        inside = [lo_hi[j][0] - 1e-9 <= Qlat["linear"][j] <= lo_hi[j][1] + 1e-9 for j in range(K)]
        d["prop8_linear_dashboard_envelope"] = {"Q_inside_envelope_all_campaigns": bool(all(inside)), "n_inside": int(sum(inside)),
            "envelope_relative_width_median(hi-lo)/Q": float(np.median([(lo_hi[j][1] - lo_hi[j][0]) / Qlat["linear"][j] for j in range(K)])),
            "abs_q_minus_Q_max": float(np.max(np.abs(mc["linear"]["q"] - Qlat["linear"]))), "abs_a_minus_A_max": float(np.max(np.abs(mc["linear"]["a"] - A_pop["linear"]))),
            "label": "theorem-covered (in-model, w=1)"}
        d["prop7_within_journey_distortion_q_over_Q"] = {m: {"min": float(np.min(mc[m]["q"] / Qlat[m])), "median": float(np.median(mc[m]["q"] / Qlat[m])), "max": float(np.max(mc[m]["q"] / Qlat[m]))} for m in E.MODELS}
        d["prop7_label"] = "no (Gamma,p_lo)-only bound for weighted/last-touch/learned rules (Prop 7 of the paper); these ratios are descriptive, out-of-theorem"
        rev = {}
        for m in E.MODELS:
            Dl = np.array([A_pop[m][j] - A_pop[m][k] for (j, k) in PAIRS]); Do = np.array([mc[m]["a"][j] - mc[m]["a"][k] for (j, k) in PAIRS])
            act = np.abs(Dl) > DELTA
            rev[m] = {"pairs_where_observed_order_differs_from_latent": int(np.sum(np.sign(Dl) != np.sign(Do))), "of_which_actionable(|Delta|>delta)": int(np.sum((np.sign(Dl) != np.sign(Do)) & act)),
                      "n_actionable": int(act.sum()), "label": GUARANTEED_LABEL(m)}
        d["natural_reversal_prevalence_population"] = rev
        d["fraction_conversions_with_zero_observed_mass"] = {m: mc[m]["frac_zero_mass_observed"] for m in E.MODELS}
        d["observed_share_a"] = {m: mc[m]["a"].round(4).tolist() for m in E.MODELS}
        obs[str(r)] = d
    return {"composition": comp, "A_latent": {m: A_pop[m].tolist() for m in E.MODELS}, "Q_latent": {m: Qlat[m].tolist() for m in E.MODELS}, "abs_Q_minus_A": abs_QA, "prop6_covariance_identity_linear": prop6,
            "per_journey_credit_L1_between_models": l1, "aggregate_share_difference_between_models": agg_diff, "pairwise_ranking_disagreement": rank_dis, "kendall_tau_between_model_A": kt,
            "observed_regimes": obs}, A_pop


def GUARANTEED_LABEL(m):
    return E.GUARANTEED[m]


def aggregate(outs, H):
    agg = {}
    for key in outs[0][2]:
        m, r, rn = key; ent = [o[2][key][H] for o in outs]
        def ms(f): v = np.array([f(e) for e in ent], float); return {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None}
        ew = sum(e["ever_wrong"] for e in ent)
        fc = []
        for e in ent:
            for p, t in e["first_cert"].items(): fc.append(t)
        d = {"label": GUARANTEED_LABEL(m), "reps": len(ent), "final_correct": ms(lambda e: e["n_correct"]), "final_wrong": ms(lambda e: e["n_wrong"]), "final_unresolved": ms(lambda e: e["n_unres"]),
             "certification_fraction": ms(lambda e: (e["n_correct"] + e["n_wrong"]) / 45), "wrong_certification_fraction_of_certified": ms(lambda e: e["n_wrong"] / max(e["n_correct"] + e["n_wrong"], 1)),
             "wrong_fraction_of_45_pairs": ms(lambda e: e["n_wrong"] / 45),
             "reps_with_any_wrong_certification_at_any_checkpoint": [int(ew), len(ent)], "any_wrong_rate_ci95_cp": cp(int(ew), len(ent)),
             "first_certification_time_conversions": {"n_pair_reps_certified": len(fc), "median": float(np.median(fc)) if fc else None, "p90": float(np.quantile(fc, .9)) if fc else None},
             "resolved_opportunity_fraction": ms(lambda e: e["loss"]["resolved_opportunity_fraction"]), "n_actionable": ms(lambda e: e["loss"]["n_actionable"]),
             "wrong_actionable": ms(lambda e: e["loss"]["wrong_actionable"]), "wrong_nonactionable": ms(lambda e: e["loss"]["wrong_nonactionable"]), "unresolved_actionable": ms(lambda e: e["loss"]["unresolved_actionable"]),
             "directional_error_loss": ms(lambda e: e["loss"]["directional_error_loss"]), "directional_error_loss_normalised": ms(lambda e: e["loss"]["directional_error_loss_normalised"])}
        for lam in E.LAMBDAS:
            d[f"abstention_adjusted_loss_normalised_lambda={lam}"] = ms(lambda e: e["loss"][f"abstention_adjusted_loss_normalised_lambda={lam}"])
            d[f"abstention_adjusted_loss_lambda={lam}"] = ms(lambda e: e["loss"][f"abstention_adjusted_loss_lambda={lam}"])
        agg.setdefault(m, {}).setdefault(str(r), {})[rn] = d
    return agg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true"); ap.add_argument("--n-mon", type=int, default=50_000); ap.add_argument("--reps", type=int, default=100); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--config", default="D", choices=list(CFG)); ap.add_argument("--horizons", default=None, help="comma list; default = N")
    ap.add_argument("--truth", default="oracle", choices=["oracle", "pop1m"]); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    global POP_N, WARM_M
    N, R = (5000, 2) if a.pilot else (a.n_mon, a.reps)
    if a.pilot: POP_N, WARM_M = 200_000, 150_000
    horizons = tuple(int(h) for h in a.horizons.split(",")) if a.horizons else (N,)
    assert max(horizons) <= N
    t0 = time.time()
    warm = E.sample_warmup(WARM_M, np.random.default_rng(np.random.SeedSequence([SEED, 1])))
    shap_model = E.LinearSHAPModel().fit(warm, seed=0)
    warm_info = {"candidate_paths": WARM_M, "n_converting": int(warm["y"].sum()), "n_nonconverting": int((~warm["y"]).sum()), "conversion_rate": float(warm["y"].mean()), "model": "LogisticRegression(C=1) on per-campaign touch counts, FULL latent warm-up journeys, trained once and frozen",
                 "coef": shap_model.beta.round(4).tolist(), "background_mean_counts": shap_model.mu.round(4).tolist(), "seed_stream": "SeedSequence([SEED,1]) disjoint from monitoring streams SeedSequence([SEED,3,rep]) and population SeedSequence([SEED,2])"}
    pop, A_pop = population_block(shap_model)
    truth_info = {"decision_truth": "1,000,000-conversion latent sample (earlier convention)"}
    if a.truth == "oracle" and not a.pilot:
        orc = json.load(open("results/e10/e10_oracle.json"))
        A_1m = A_pop
        A_pop = {m: np.array(orc["A_latent"][m]) for m in E.MODELS}
        n_changed = {m: int(sum((abs(A_1m[m][j] - A_1m[m][k]) > DELTA) != (abs(A_pop[m][j] - A_pop[m][k]) > DELTA) for (j, k) in PAIRS)) for m in E.MODELS}
        truth_info = {"decision_truth": f"{orc['config']['n_conversions']}-conversion latent oracle population (results/e10/e10_oracle.json, seed stream [SEED,5,chunk])",
                      "max_abs_A_diff_vs_1M_sample": {m: float(np.abs(A_1m[m] - A_pop[m]).max()) for m in E.MODELS}, "n_pairs_whose_actionability_differs_vs_1M_sample": n_changed}
    state = (shap_model.beta, shap_model.mu)
    cfg = a.config
    with ProcessPoolExecutor(a.workers) as ex:
        outs = list(ex.map(one_rep, [(i, N, A_pop, state, horizons, cfg) for i in range(R)]))
    rates = [o[1] for o in outs]
    by_h = {str(H): aggregate(outs, H) for H in horizons}
    main_H = BASE_HORIZON if BASE_HORIZON in horizons else horizons[-1]
    cert_method, boundary = CFG[cfg]
    result = {"config": {"K": K, "pairs": len(PAIRS), "alpha": ALPHA, "delta": DELTA, "N_monitored": N, "horizons": list(horizons), "replications": R, "configuration": cfg, "cert_method": cert_method, "boundary": boundary,
                         "budget": "shared alpha/(K+1) (K numerators + denominator); production rule hybrid (tracked pool, K=10 = whole pool)",
                         "rules": "naive Gamma=1; well-specified Gamma=r_true; misspecified Gamma=1.25 for r_true in {2.0,2.5}", "checkpoints": "40 geometric points 250..50,000 (+ horizons; coarser 50,000..N when N > 50,000, nested)", "seed": SEED,
                         "decision_semantics": "certify(j>k) INCORRECT if Delta_jk <= delta (tex); actionable pair |Delta|>delta", **truth_info},
              "warmup": warm_info, "population": pop, "monitoring": by_h[str(main_H)], "monitoring_by_horizon": by_h, "monitoring_main_horizon": main_H,
              "candidate_conversion_rate_in_replications": float(np.mean(rates)), "wall_time_s": time.time() - t0}
    outdir = Path(a.out or ("results/e10" if cfg == "D" else f"results/config_runs/{cfg}/e10")); outdir.mkdir(parents=True, exist_ok=True)
    stem = "e10_results_pilot" if a.pilot else ("e10_results" if max(horizons) == BASE_HORIZON or len(horizons) == 1 and N == BASE_HORIZON else f"e10_results_N{N}")
    json.dump(result, open(outdir / f"{stem}.json", "w"), indent=1, default=float)
    arrs = {}
    for (m, r, rn) in outs[0][2]:
        for H in horizons:
            arrs[f"{m}|{r}|{rn}|{H}|codes"] = np.array([o[2][(m, r, rn)][H]["codes"] for o in outs], dtype=np.int8)
            arrs[f"{m}|{r}|{rn}|{H}|gap_jk"] = np.array([o[2][(m, r, rn)][H]["gap_jk"] for o in outs]); arrs[f"{m}|{r}|{rn}|{H}|gap_kj"] = np.array([o[2][(m, r, rn)][H]["gap_kj"] for o in outs])
    np.savez_compressed(outdir / f"{stem}_pair_outcomes.npz", **arrs)
    print("wrote", outdir / f"{stem}.json", "wall", round(result["wall_time_s"], 1))


if __name__ == "__main__":
    main()
