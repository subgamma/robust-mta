"""Independent population oracle for E10 (synthetic; not observational evidence).

Latent population: 10 independent chunks x 400,000 converting latent journeys = 4,000,000 conversions (seed
stream SeedSequence([SEED, 5, chunk]), disjoint from warm-up [SEED,1], the 1M composition sample [SEED,2],
monitoring [SEED,3,rep] and thinning [SEED,4,...]). Per chunk: latent credit totals S_m(j) for every model.
 * truth: A_j = S_j / sum S (full observation); Delta_jk = A_j - A_k; actionable iff |Delta_jk| > delta.
 * observed: independent Bernoulli retention with fixed weights gives E[T_j] = p_j E[S_j] exactly, so no thinning
   Monte Carlo is needed; the only Monte Carlo is the latent E[S_j].
 * oracle gap: Delta^oracle_jk(Gamma) = direct_lower_gap(m, m, Gamma, j, k, p_lo) with m_j = p_j S_j (point boxes:
   the sharp lower end of the identified set for A_j - A_k at the population observed shares), cross-checked
   against the independent LP (run_e3.lp_extreme) for every pair and condition.
 * SE: batch means over the 10 chunks (gap per chunk; SE = sd/sqrt(10)).
 * boundary-ambiguous: |Delta_true| or the oracle gap within 3 SE of delta.
Only linear and time_decay carry the fixed-weight identification theorem; last_touch and logistic_shap get truth
only. The oracle uses no confidence sequence.
Output: results/e10/e10_oracle.json
"""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

sys.path.insert(0, ".")
from anytime_attribution.certify import direct_lower_gap
from anytime_attribution.experiments import e10_multicampaign as E

SEED, DELTA, CHUNKS, CHUNK_N, WARM_M = 20263, 0.005, 10, 400_000, 600_000
PAIRS = E.pair_list(); K = E.K
THEOREM_MODELS = ("linear", "time_decay")


def rules_for(r):
    d = {"naive": 1.0, "wellspec": float(r)}
    if r in (2.0, 2.5):
        d["misspec1.25"] = 1.25
    return d


def chunk_totals(args):
    c, shap_state = args
    sm = E.LinearSHAPModel(); sm.beta, sm.mu = shap_state
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 5, c]))
    trip, rate = E.sample_converting(CHUNK_N, rng)
    out = {m: E.model_masses(m, trip, None, CHUNK_N, sm)[0].sum(0) for m in E.MODELS}
    return out, rate


def oracle_gap(S, r, Gamma, j, k):
    p = E.retention_vector(r); m = p * S
    return direct_lower_gap(m, m, Gamma, j, k, p_lo=float(p.min()), delta=DELTA)


def lp_gap(S, r, Gamma, j, k):
    from scripts.run_e3 import lp_extreme   # independent LP reference
    p = E.retention_vector(r); m = p * S; plo = float(p.min())
    return lp_extreme(m / m.sum(), j, k, Gamma, plo, "min", "gap")[0]


def main():
    t0 = time.time()
    warm = E.sample_warmup(WARM_M, np.random.default_rng(np.random.SeedSequence([SEED, 1])))
    shap = E.LinearSHAPModel().fit(warm, seed=0); state = (shap.beta, shap.mu)
    with ProcessPoolExecutor(CHUNKS) as ex:
        res = list(ex.map(chunk_totals, [(c, state) for c in range(CHUNKS)]))
    rates = [x[1] for x in res]
    tot = {m: np.sum([x[0][m] for x in res], axis=0) for m in E.MODELS}
    A = {m: tot[m] / tot[m].sum() for m in E.MODELS}
    out = {"config": {"n_conversions": CHUNKS * CHUNK_N, "chunks": CHUNKS, "seed_stream": "SeedSequence([20263,5,chunk])", "delta": DELTA,
                      "candidate_conversion_rate": float(np.mean(rates)), "nonlinear_gap_se": "batch means over chunks"},
           "A_latent": {m: A[m].tolist() for m in E.MODELS}, "conditions": {}}
    max_lp_dev = 0.0
    for m in THEOREM_MODELS:
        for r in E.R_TRUE_GRID:
            p = E.retention_vector(r)
            for rn, G in rules_for(r).items():
                rows = []
                for (j, k) in PAIRS:
                    d_true = A[m][j] - A[m][k]
                    sgn = 1 if d_true >= 0 else -1
                    a_, b_ = (j, k) if sgn > 0 else (k, j)             # oriented in the TRUE direction
                    g = oracle_gap(tot[m], r, G, a_, b_)
                    # batch means
                    gb = np.array([oracle_gap(x[0][m], r, G, a_, b_) for x in res])
                    db = np.array([(x[0][m][a_] - x[0][m][b_]) / x[0][m].sum() for x in res])
                    se_g = float(gb.std(ddof=1) / np.sqrt(CHUNKS)); se_d = float(db.std(ddof=1) / np.sqrt(CHUNKS))
                    lp = lp_gap(tot[m], r, G, a_, b_); max_lp_dev = max(max_lp_dev, abs(lp - g))
                    absd = abs(d_true)
                    actionable = absd > DELTA
                    amb_true = abs(absd - DELTA) <= 3 * se_d
                    amb_gap = abs(g - DELTA) <= 3 * se_g
                    rows.append({"pair": [j, k], "oriented": [a_, b_], "delta_true_abs": float(absd), "se_delta_true": se_d, "oracle_gap": float(g), "se_oracle_gap": se_g,
                                 "actionable": bool(actionable), "ambiguous": bool(actionable and (amb_true or amb_gap)) or bool((not actionable) and amb_true),
                                 "oracle_certifiable": bool(g > DELTA), "lp_gap": float(lp)})
                out["conditions"][f"{m}|{r}|{rn}"] = {"model": m, "r_true": r, "rule": rn, "Gamma": G, "p_lo_declared": float(p.min()),
                                                      "Gamma_eff": float(min(G, 1 / p.min())), "pairs": rows}
    out["lp_crosscheck_max_abs_deviation"] = max_lp_dev
    out["wall_time_s"] = time.time() - t0
    Path("results/e10").mkdir(parents=True, exist_ok=True)
    json.dump(out, open("results/e10/e10_oracle.json", "w"), indent=1, default=float)
    print("wrote results/e10/e10_oracle.json; LP max dev", max_lp_dev, "wall", round(out["wall_time_s"], 1))


if __name__ == "__main__":
    main()
