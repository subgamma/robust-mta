"""E3: identification-boundary verification and sharpness (synthetic, verification only).

Output: results/e3/<name> (see --out and --name; --boundary simplified writes the looser-boundary comparison).

A. Exact endpoint verification (no sampling). For a grid of (Gamma, p_lo) cells and random observed share vectors
   a, the identified sets of A_j and A_j - A_k over the admissible retention polytope
   P(Gamma, p_lo) = {p in [p_lo,1]^K: p_i <= Gamma p_k} are solved independently of the proofs by linear
   programming (Charnes-Cooper on u = 1/p) and compared with the closed forms at Gamma_eff = min(Gamma, 1/p_lo)
   and with the conservative ratio-only version. The LP optimiser is returned as the attaining configuration and
   re-checked for admissibility. The ordering rule (identified iff a_j/a_k > Gamma_eff) is tested on random pairs
   and on pairs within 1e-6 of the threshold.
B. Floor-binding cell Gamma=2, p_lo=0.8 (Gamma_eff=1.25): sharp (floored) vs conservative (ratio-only) endpoints,
   widths, width reduction, differing ordering conclusions, and empirical coverage.
C. Finite-sample composed coverage (CS box x identified-set map, `identification.robust_A_interval`) at n in
   {5000, 20000, 50000}, 200 replications per cell, linear and time-decay weights, under admissible retention:
   interior random, lower-endpoint vertex, upper-endpoint vertex; Gamma_used = Gamma_eff (sharp) and Gamma
   (conservative); plus a negative control (Gamma_used = 1 with true retention ratio 2) showing the test can
   detect undercoverage. 100% coverage is not treated as calibration evidence.
D. Dashboard envelope: exact q_j/Q_j computations, attainment of both endpoints at n_min = 1 (converging as N
   grows, pi -> 0), and an adversarial random search over two- and three-type mixtures that must never exceed
   the bound; n_min > 1 is informational (sharpness not claimed).

Run: python scripts/run_e3.py [--reps 200] [--pilot]
"""
from __future__ import annotations

import argparse, itertools, json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta as _beta, binom

sys.path.insert(0, ".")
from anytime_attribution.cs import _check_form, boundary_override
from anytime_attribution import cs as _cs
from anytime_attribution.cs_vectorized import cumulative_share_boxes
from anytime_attribution.identification import (gamma_eff, ordering_identified, prop2_interval, prop3_gap_interval,
                                                 prop7_envelope, robust_A_interval)
from anytime_attribution.synth import LENGTH_DIST_BINS, sample_journey_lengths, true_aggregate_share

RATIO_ONLY_PLO = 1e-6            # effectively no floor
CELLS = [(1.25, RATIO_ONLY_PLO), (2.0, RATIO_ONLY_PLO), (1.25, 0.8), (1.5, 0.8), (2.0, 0.8), (2.0, 0.5), (5.0, 0.2), (3.0, 0.2)]
B_BOUND = 10.0


def lp_extreme(a, j, k, Gamma, p_lo, sense, mode):
    """Optimise A_j (mode="A", k ignored) or A_j - A_k (mode="gap") over admissible u = 1/p.

    Variables (y_1..y_K, t) with y = u t, t = 1/sum_m a_m u_m (Charnes-Cooper). Returns (value, p*).
    """
    K = len(a); umax = 1.0 / p_lo
    c = np.zeros(K + 1)
    if mode == "A":
        c[j] = a[j]
    else:
        c[j] = a[j]; c[k] = -a[k]
    if sense == "min":
        obj = c
    else:
        obj = -c
    A_eq = [np.append(a, 0.0)]; b_eq = [1.0]
    A_ub, b_ub = [], []
    for i in range(K):
        r = np.zeros(K + 1); r[i] = -1; r[K] = 1; A_ub.append(r); b_ub.append(0.0)           # t <= y_i (u_i >= 1)
        r = np.zeros(K + 1); r[i] = 1; r[K] = -umax; A_ub.append(r); b_ub.append(0.0)        # y_i <= umax t
        for m in range(K):
            if m != i:
                r = np.zeros(K + 1); r[i] = 1; r[m] = -Gamma; A_ub.append(r); b_ub.append(0.0)   # u_i <= Gamma u_m
    res = linprog(obj, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq, bounds=[(0, None)] * (K + 1), method="highs")
    if res.status != 0:
        raise RuntimeError(f"LP failed: {res.message}")
    y, t = res.x[:K], res.x[K]
    u = y / t; p = 1.0 / u
    val = (a[j] * y[j]) if mode == "A" else (a[j] * y[j] - a[k] * y[k])
    return float(val), p


def admissible(p, Gamma, p_lo, tol=1e-7):
    return bool(np.all(p >= p_lo - tol) and np.all(p <= 1 + tol) and p.max() / p.min() <= Gamma + tol)


def part_A(rng, n_a=200, K=5):
    out = {}
    for (G, plo) in CELLS:
        Ge = gamma_eff(G, plo)
        dev_sharp = dev_cons = dev_gap = 0.0; n_bad_adm = 0; n_cons_not_sharp = 0; n_sets = 0
        wid_sharp, wid_cons = [], []
        ord_checks = ord_ok = 0; near_ok = near_n = 0
        examples = None
        for _ in range(n_a):
            a = rng.dirichlet(np.ones(K))
            for j in range(K):
                lo_lp, p_lo_cfg = lp_extreme(a, j, 0, G, plo, "min", "A"); hi_lp, p_hi_cfg = lp_extreme(a, j, 0, G, plo, "max", "A")
                lo_lp, hi_lp = lo_lp, hi_lp  # LP value is A_j (since sum a u t = 1 -> objective a_j y_j = A_j)
                slo, shi = prop2_interval(a[j], G, plo); clo, chi = prop2_interval(a[j], G)
                dev_sharp = max(dev_sharp, abs(lo_lp - slo), abs(hi_lp - shi))
                # conservative set must CONTAIN the sharp one; flag when it is strictly larger (not sharp)
                if clo < slo - 1e-9 or chi > shi + 1e-9:
                    n_cons_not_sharp += 1
                dev_cons = max(dev_cons, abs(lo_lp - clo) if G <= 1 / plo + 1e-12 else 0.0)
                n_bad_adm += (not admissible(p_lo_cfg, G, plo)) + (not admissible(p_hi_cfg, G, plo))
                # Attained: the configuration reproduces the endpoint exactly when plugged into the closed form.
                u = 1.0 / p_lo_cfg; A_at = a[j] * u[j] / (a * u).sum(); dev_sharp = max(dev_sharp, abs(A_at - slo))
                wid_sharp.append(shi - slo); wid_cons.append(chi - clo); n_sets += 1
                if examples is None and j == 0:
                    examples = {"a": a.round(4).tolist(), "j": 0, "sharp_set": [slo, shi], "conservative_set": [clo, chi],
                                "lower_endpoint_attaining_p": p_lo_cfg.round(4).tolist(), "upper_endpoint_attaining_p": p_hi_cfg.round(4).tolist()}
            for (j, k) in itertools.combinations(range(K), 2):
                glo_lp, _ = lp_extreme(a, j, k, G, plo, "min", "gap"); ghi_lp, _ = lp_extreme(a, j, k, G, plo, "max", "gap")
                glo, ghi = prop3_gap_interval(a, G, j, k, plo)
                dev_gap = max(dev_gap, abs(glo_lp - glo), abs(ghi_lp - ghi))
                # Ordering j>k is identified iff the LP-min gap > 0.
                r = a[j] / a[k]
                ord_checks += 1; ord_ok += ((glo_lp > 1e-9) == ordering_identified(a[j], a[k], G, plo))
        for _ in range(60):
            a = rng.dirichlet(np.ones(K)); a[1] = a[0] / (Ge * (1 + rng.choice([-1, 1]) * 1e-6))
            a = a / a.sum()
            glo_lp, _ = lp_extreme(a, 0, 1, G, plo, "min", "gap")
            near_n += 1; near_ok += ((glo_lp > 0) == ordering_identified(a[0], a[1], G, plo))
        out[f"Gamma={G},p_lo={plo}"] = {
            "Gamma_eff": Ge, "regime": "floor binding (Gamma > 1/p_lo)" if G > 1 / plo + 1e-12 else "ratio constraint binding",
            "prop2_max_abs_dev_LP_vs_sharp_closed_form": dev_sharp, "prop3_max_abs_dev_LP_vs_sharp_closed_form": dev_gap,
            "n_attaining_configs_inadmissible": int(n_bad_adm), "n_sets": n_sets,
            "frac_sets_where_conservative_strictly_larger_than_sharp": n_cons_not_sharp / n_sets,
            "mean_width_sharp": float(np.mean(wid_sharp)), "mean_width_conservative": float(np.mean(wid_cons)),
            "mean_width_reduction_sharp_vs_conservative": float(1 - np.mean(wid_sharp) / np.mean(wid_cons)),
            "corollary2_agreement": [int(ord_ok), int(ord_checks)], "corollary2_threshold_pairs_agreement": [int(near_ok), int(near_n)],
            "example_attaining_configuration": examples}
    return out


def part_B(rng, K=10, n_a=2000):
    G, plo = 2.0, 0.8; Ge = gamma_eff(G, plo)
    a0 = np.array([0.4, 0.3, 0.3])
    tab = []
    for j in range(3):
        slo, shi = prop2_interval(a0[j], G, plo); clo, chi = prop2_interval(a0[j], G)
        tab.append({"campaign": j, "a_obs": a0[j], "sharp_floored_Gamma_eff_1.25": [slo, shi], "conservative_ratio_only_Gamma_2": [clo, chi],
                    "width_sharp": shi - slo, "width_conservative": chi - clo, "width_reduction": 1 - (shi - slo) / (chi - clo)})
    diff = tot = 0
    ex = None
    for _ in range(n_a):
        a = rng.dirichlet(np.ones(K))
        for (j, k) in itertools.combinations(range(K), 2):
            r = max(a[j] / a[k], a[k] / a[j]); tot += 1
            s_id = r > Ge; c_id = r > G
            if s_id != c_id:
                diff += 1
                if ex is None: ex = {"a_j": float(a[j]), "a_k": float(a[k]), "ratio": float(a[j] / a[k]), "identified_sharp(Gamma_eff=1.25)": bool(s_id), "identified_conservative(Gamma=2)": bool(c_id)}
    # interior + vertex draws coverage (population level, exact): true A always inside the SHARP set
    viol = 0; draws = 0; closeness = []
    for _ in range(5000):
        a = rng.dirichlet(np.ones(5)); j = int(rng.integers(5))
        pmin = rng.uniform(plo, 1.0); p = np.minimum(1.0, np.maximum(plo, pmin * rng.uniform(1.0, Ge, size=5)))
        if p.max() / p.min() > Ge + 1e-12: continue
        A = (a / p)[j] / (a / p).sum(); lo, hi = prop2_interval(a[j], G, plo); draws += 1
        viol += not (lo - 1e-12 <= A <= hi + 1e-12); closeness.append(min(A - lo, hi - A) / (hi - lo))
    return {"example_a": a0.tolist(), "sets": tab, "pair_ordering": {"pairs_evaluated": tot, "pairs_where_sharp_and_conservative_conclusions_differ": diff, "fraction": diff / tot,
            "example": ex, "note": "pairs with observed share ratio in (1.25, 2] are identified under the sharp floored set but not under the conservative ratio-only set"},
            "population_coverage_of_sharp_set_under_random_admissible_p": {"draws": draws, "violations": int(viol)}}


W_FS = np.linspace(1.8, 0.4, 5); A_FS = true_aggregate_share(W_FS); K_FS = 5


def latent_counts_and_decay(n, rng):
    """Capped latent journeys: per-touch arrays, so thinning can be applied touch-wise.

    Cap = last 10 touches (A4, as apply_cap with a 7-day lookback); tau = 7 - U(0,7), sorted.
    """
    lengths = sample_journey_lengths(n, rng)
    jid = np.repeat(np.arange(n), lengths)
    camp = rng.choice(K_FS, size=len(jid), p=W_FS / W_FS.sum())
    tau = 7.0 - rng.uniform(0, 7.0, size=len(jid))          # time to conversion in days
    order = np.lexsort((tau, jid)); jid, camp, tau = jid[order], camp[order], tau[order]   # within journey: ascending tau (most recent first)
    start = np.concatenate([[0], np.cumsum(lengths)[:-1]]); rank = np.arange(len(jid)) - np.repeat(start, lengths)
    keep = rank < 10                                         # keep the 10 most recent touches
    return jid[keep], camp[keep], tau[keep]


def observed_streams(jid, camp, tau, n, p, rng, kind):
    keep = rng.random(len(jid)) < p[camp]
    w = np.ones(len(jid)) if kind == "linear" else 2.0 ** (-tau / 7.0)
    Tj = np.zeros((n, K_FS)); np.add.at(Tj, (jid[keep], camp[keep]), w[keep])
    return Tj, Tj.sum(1)


def retention_config(name, G, plo, rng):
    Ge = gamma_eff(G, plo)
    j = 2   # the evaluated campaign (middle of K=5, fixed)
    if name == "interior":
        while True:
            pmin = rng.uniform(plo, 1.0); p = np.minimum(1.0, np.maximum(plo, pmin * rng.uniform(1.0, Ge, size=K_FS)))
            if p.max() / p.min() <= Ge + 1e-12: return p
    p = np.full(K_FS, plo)
    if name == "vertex_A_low":      # campaign j best observed -> A_j at its LOWER endpoint
        p[j] = min(1.0, Ge * plo)
    elif name == "vertex_A_high":   # campaign j worst observed, the others best -> A_j at its UPPER endpoint
        p = np.full(K_FS, min(1.0, Ge * plo)); p[j] = plo
    return p


def fs_cell(args):
    kind, cfg, G, plo, n_max, rep, G_used, seed = args
    rng = np.random.default_rng(np.random.SeedSequence([20261, seed, rep]))
    p = retention_config(cfg, G, plo, rng) if cfg != "negctl_vertex" else None
    if cfg == "negctl_vertex":
        p = np.full(K_FS, 0.2); p[2] = 0.4                       # true retention ratio 2 (floor 0.2)
    jid, camp, tau = latent_counts_and_decay(n_max, rng)
    Tj, T = observed_streams(jid, camp, tau, n_max, p, rng, kind)
    # true full-observation aggregate share for the rule (latent, before thinning), same capped journeys
    w = np.ones(len(jid)) if kind == "linear" else 2.0 ** (-tau / 7.0)
    S = np.zeros(K_FS); np.add.at(S, camp, w); A_true = S / S.sum()
    j = 2
    res = {}
    bx = cumulative_share_boxes(Tj, T, B_BOUND, 0.05 / 2 / 2)   # single-campaign-style CS at alpha=0.05 (beta = alpha/2, boundary alpha/4)
    L, U = bx["L"][:, j], bx["U"][:, j]
    lo_map = np.vectorize(lambda x: prop2_interval(x, G_used, None)[0]); hi_map = np.vectorize(lambda x: prop2_interval(x, G_used, None)[1])
    lo, hi = lo_map(L), hi_map(U)
    cov = (lo - 1e-12 <= A_true[j]) & (A_true[j] <= hi + 1e-12)
    for n in (5000, 20000, 50000):
        if n <= n_max: res[n] = (bool(cov[n - 1]), float(hi[n - 1] - lo[n - 1]))
    res["pathwise"] = bool(cov[499:].all())
    res["ratio_true"] = float(p.max() / p.min())
    return res


def cp_ci(k, n):
    lo = 0.0 if k == 0 else _beta.ppf(0.025, k, n - k + 1); hi = 1.0 if k == n else _beta.ppf(0.975, k + 1, n - k)
    return [float(lo), float(hi)]


def part_C(reps, workers):
    cells = []
    G, plo = 2.0, 0.8; Ge = gamma_eff(G, plo)
    specs = []
    for kind in ("linear", "time_decay"):
        for cfg in ("interior", "vertex_A_low", "vertex_A_high"):
            for tag, Gu in (("sharp_Gamma_eff=1.25", Ge), ("conservative_Gamma=2", G)):
                specs.append((kind, cfg, G, plo, Gu, tag))
    for kind in ("linear", "time_decay"):
        specs.append((kind, "negctl_vertex", 2.0, 0.2, 1.0, "NEGATIVE CONTROL Gamma_used=1 (true ratio 2)"))
    out = {}
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(_check_form(None),)) as ex:
        for (kind, cfg, Gg, pl, Gu, tag) in specs:
            r = list(ex.map(fs_cell, [(kind, cfg, Gg, pl, 50000, rep, Gu, 0) for rep in range(reps)]))
            key = f"{kind}|{cfg}|{tag}"
            ent = {"Gamma_true_model": Gg, "p_lo": pl, "Gamma_used": Gu, "reps": reps, "true_retention_ratio_example": r[0]["ratio_true"]}
            for n in (5000, 20000, 50000):
                k = sum(x[n][0] for x in r)
                ent[f"coverage_n={n}"] = k / reps; ent[f"coverage_ci95_cp_n={n}"] = cp_ci(k, reps)
                ent[f"median_interval_width_n={n}"] = float(np.median([x[n][1] for x in r]))
            kp = sum(x["pathwise"] for x in r); ent["pathwise_coverage_t>=500"] = kp / reps; ent["pathwise_ci95_cp"] = cp_ci(kp, reps)
            out[key] = ent
    return out


def exact_q_Q(types, pj, pk):
    """types = [(prob, n_j, n_k)]. Exact q_j = E[T_j/T 1{T>0}]/P(T>0) and Q_j under independent retention (w=1)."""
    num = den = Qj = 0.0
    for pr, nj, nk in types:
        a = np.arange(nj + 1); b = np.arange(nk + 1)
        pa = binom.pmf(a, nj, pj); pb = binom.pmf(b, nk, pk)
        tot = a[:, None] + b[None, :]
        P = pa[:, None] * pb[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(tot > 0, a[:, None] / np.maximum(tot, 1), 0.0)
        num += pr * float((P * frac).sum()); den += pr * float(P[tot > 0].sum()); Qj += pr * nj / (nj + nk)
    return num / den, Qj


def part_D(rng):
    out = {"attaining_constructions_n_min_1": []}
    for (G, plo) in ((2.0, 0.2), (5.0, 0.2), (2.0, 0.5), (1.5, 0.8)):
        Ge = gamma_eff(G, plo); rows = []
        for N in (10, 100, 1000, 5000):
            for pi in (1e-2, 1e-4, 1e-6):
                # lower: single j touch (prob pi) vs N k-touches, all retention p_lo
                q, Q = exact_q_Q([(pi, 1, 0), (1 - pi, 0, N)], plo, plo)
                # upper: p_j = Gamma_eff p_lo, p_k = p_lo; type A: 1 on j + N on k (pi); type B: 1 on k (1-pi)
                q2, Q2 = exact_q_Q([(pi, 1, N), (1 - pi, 0, 1)], min(1.0, Ge * plo), plo)
                rows.append({"N": N, "pi": pi, "q_over_Q_lower_construction": q / Q, "q_over_Q_upper_construction": q2 / Q2})
        lo_lim, up_lim = plo, Ge / plo
        out["attaining_constructions_n_min_1"].append({"Gamma": G, "p_lo": plo, "Gamma_eff": Ge, "bound_lower(p_lo)": lo_lim, "bound_upper(Gamma_eff/p_lo; s_low=p_lo at n_min=1)": up_lim,
            "best_ratio_lower_construction": min(r["q_over_Q_lower_construction"] for r in rows), "best_ratio_upper_construction": max(r["q_over_Q_upper_construction"] for r in rows),
            "relative_gap_to_upper_bound": 1 - max(r["q_over_Q_upper_construction"] for r in rows) / up_lim,
            "relative_gap_to_lower_bound": min(r["q_over_Q_lower_construction"] for r in rows) / lo_lim - 1, "grid": rows})
    # adversarial random search: never exceed the bound
    worst = {"upper_violations": 0, "lower_violations": 0, "n": 0, "max_ratio_over_bound": 0.0, "min_ratio_over_plo": 1e9}
    for _ in range(20000):
        G = float(rng.choice([1.25, 2.0, 5.0])); plo = float(rng.choice([0.2, 0.5, 0.8])); Ge = gamma_eff(G, plo)
        nmin = int(rng.choice([1, 2, 3]))
        ntypes = int(rng.integers(2, 4)); pr = rng.dirichlet(np.ones(ntypes))
        types = []
        for t in range(ntypes):
            nk = int(rng.integers(0, 15)); nj = int(rng.integers(0, 15))
            if nj + nk < nmin: nk = nmin
            types.append((pr[t], nj, nk))
        if rng.random() < 0.5:
            pj, pk = (min(1.0, Ge * plo), plo) if rng.random() < 0.5 else (plo, min(1.0, Ge * plo))
        else:
            pmin = rng.uniform(plo, 1.0); pk = pmin; pj = min(1.0, pmin * rng.uniform(1.0, Ge)) if rng.random() < 0.5 else pmin
            if rng.random() < 0.5: pj, pk = pk, pj
        if max(pj, pk) / min(pj, pk) > Ge + 1e-9: continue
        nmin_act = min(nj + nk for (_, nj, nk) in types); s_low = 1 - (1 - plo) ** nmin_act
        q, Q = exact_q_Q(types, pj, pk)
        if Q <= 0: continue
        worst["n"] += 1
        worst["max_ratio_over_bound"] = max(worst["max_ratio_over_bound"], (q / Q) / (Ge / s_low)); worst["min_ratio_over_plo"] = min(worst["min_ratio_over_plo"], (q / Q) / plo)
        worst["upper_violations"] += (q / Q) > Ge / s_low + 1e-9; worst["lower_violations"] += (q / Q) < plo - 1e-9
    out["adversarial_random_search"] = worst
    return out


def _init_worker(form):
    """Spawned workers do not inherit the boundary override; set it explicitly."""
    _cs._ACTIVE_BOUNDARY[0] = form


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=200); ap.add_argument("--pilot", action="store_true"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="results/e3"); ap.add_argument("--name", default=None, help="output file name (default: e3_results.json, or e3_simplified_boundary.json with --boundary simplified)")
    ap.add_argument("--boundary", default="exact", choices=["exact", "simplified"], help="stitched-boundary form of the confidence sequences; 'exact' is the production form, 'simplified' is the looser upper bound used only for the width comparison")
    a = ap.parse_args(); reps = 8 if a.pilot else a.reps
    a.name = a.name or ("e3_results.json" if a.boundary == "exact" else "e3_simplified_boundary.json")
    out_dir = Path(a.out + ("_pilot" if a.pilot else "")); out_dir.mkdir(parents=True, exist_ok=True)
    with boundary_override(a.boundary):
        t0 = time.time(); rng = np.random.default_rng(20260)
        res = {"config": {"seed": 20260, "cells": CELLS, "fs_reps": reps, "boundary": _check_form(None), "cert_method": "N/A (identified-set verification; composed with CS boxes in part C)", "ratio_only_p_lo_proxy": RATIO_ONLY_PLO}}
        res["A_exact_LP_verification"] = part_A(rng, n_a=20 if a.pilot else 200)
        res["B_floor_binding_M1"] = part_B(rng)
        res["C_finite_sample_composed_coverage"] = part_C(reps, a.workers)
        res["D_prop8_envelope"] = part_D(rng)
        res["wall_time_s"] = time.time() - t0
        json.dump(res, open(out_dir / a.name, "w"), indent=1, default=float)
        print("wrote", out_dir / a.name, round(res["wall_time_s"], 1))


if __name__ == "__main__":
    main()
