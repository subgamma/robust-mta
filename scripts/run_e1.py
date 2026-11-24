"""E1: calibration, width and multiplicity-matched baseline fairness (synthetic, verification only).

Output: results/e1/<name> (see --out and --name; --boundary simplified writes the looser-boundary comparison).

Methods (K = 10 campaigns, B = 10, N = 20,000 conversions per replication):
 Anytime-valid (Howard et al. 2021 Thm 4 ratio CS; `cs_vectorized` equals `cs.py`, m = 1):
   cs_shared: one object, K numerators + shared denominator, beta = alpha/(K+1) (primary)
   cs_single: K independent single-campaign CS (each beta = alpha/2): anytime, marginal only,
              no multiplicity correction
 Fixed-sample (valid at one prespecified horizon only):
   bootstrap, fixed_eb, bootstrap_bonf, fixed_eb_bonf, csranks_style (Bonf = alpha/K; csranks-style is Bonf by design)
Bonferroni addresses multiplicity only; it does not make a fixed-sample interval anytime-valid. The budget
differences are deliberate: fixed-sample Bonf uses alpha/K, cs_shared uses alpha/(K+1) (its denominator process
takes the extra share and serves the dashboard intervals on the same event).

Coverage notions, kept separate:
 horizon:  covered at one prespecified t in {1000, 5000, 10000, 20000} (marginal = fraction of the K campaigns;
           simultaneous = all K at once)
 peek4:    covered at all four prespecified horizons (a four-look analyst; all methods)
 pathwise: covered at every t in [t0, N] (anytime methods and, for contrast, fixed-sample methods peeked at every
           t >= 500); marginal per campaign and simultaneous over K
Truth: stationary panel A_j = w_j/sum(w); drift panel with exact cumulative share a_{j,t} = (1/t) sum_{i<=t} p_{i,j}.
The iid assumptions of fixed-sample methods are violated in the drift panel, which only shows how they behave.

Run: python scripts/run_e1.py [--reps 200] [--pilot]
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import beta as _beta, norm

sys.path.insert(0, ".")
from anytime_attribution.cs import _check_form, boundary_override
from anytime_attribution import cs as _cs
from anytime_attribution.cs_vectorized import cumulative_share_boxes
from anytime_attribution.experiments.e1_coverage import (
    A_TRUE, B_BOUND, K, DRIFT_P_END, DRIFT_P_START, _stream_Tj_T, MAX_TOUCHES)

N = 20_000
HORIZONS = (1000, 5000, 10000, 20000)
ALPHAS = (0.05, 0.20, 0.50)
T0_PATH = 500            # pathwise window start (existing E1 n0)
WIDTH_GRID = np.unique(np.geomspace(100, N, 40).astype(int))
BOOT_B = 2000        # large enough to resolve the alpha/(2K) = 0.25% tails of the Bonferroni percentile interval
SEED0 = 20260


def gen_drift(rng, n=N):
    frac = np.linspace(0, 1, n)[:, None]
    p_t = DRIFT_P_START[None, :] * (1 - frac) + DRIFT_P_END[None, :] * frac
    p_t = p_t / p_t.sum(1, keepdims=True)
    u = rng.random(n)
    camp = (u[:, None] > np.cumsum(p_t, 1)).sum(1).clip(0, K - 1)
    w = rng.integers(1, MAX_TOUCHES + 1, size=n).astype(float)
    Tj = np.zeros((n, K)); Tj[np.arange(n), camp] = w
    return Tj, w, np.cumsum(p_t, 0) / np.arange(1, n + 1)[:, None]


def delta_method_all_t(Tj, T):
    """ratio_hat_t and delta-method SE_t for every t (vectorised cumulative sums); equal to baselines._delta_method_se at
    each t (tested).
    """
    n = len(T); t = np.arange(1, n + 1, dtype=float)
    cT, cTj = np.cumsum(T), np.cumsum(Tj, 0)
    r = cTj / cT[:, None]
    s_jj = np.cumsum(Tj ** 2, 0); s_jt = np.cumsum(Tj * T[:, None], 0); s_tt = np.cumsum(T ** 2)
    ss = s_jj - 2 * r * s_jt + r ** 2 * s_tt[:, None]          # sum of squared residuals around 0
    mean_res = (cTj - r * cT[:, None]) / t[:, None]             # = 0 up to rounding (r minimises)
    var = (ss - t[:, None] * mean_res ** 2) / np.maximum(t[:, None] - 1, 1)
    se = np.sqrt(np.maximum(var, 0) / t[:, None]) / (cT / t)[:, None]
    return r, se, t


def fixed_hw(se, t, alpha, kind):
    if kind == "fixed_eb":
        return norm.ppf(1 - alpha / 2) * se + B_BOUND * math.log(2 / alpha) / (3 * np.maximum(t - 1, 1))[:, None]
    if kind == "fixed_eb_bonf":
        a = alpha / K
        return norm.ppf(1 - a / 2) * se + B_BOUND * math.log(2 / a) / (3 * np.maximum(t - 1, 1))[:, None]
    if kind == "csranks_style":
        return norm.ppf(1 - alpha / (2 * K)) * se
    raise ValueError(kind)


def one_rep(args):
    setting, rep, seed = args
    rng = np.random.default_rng(np.random.SeedSequence([SEED0, seed, rep]))
    if setting == "stationary":
        Tj, T = _stream_Tj_T(N, rng); truth = np.broadcast_to(A_TRUE, (N, K))
    else:
        Tj, T, truth = gen_drift(rng)
    out = {}
    r_all, se_all, t_all = delta_method_all_t(Tj, T)
    boot = {}
    for h in HORIZONS:
        num = np.zeros((BOOT_B, K)); den = np.zeros(BOOT_B)
        for c0 in range(0, BOOT_B, 50):
            ii = rng.integers(0, h, size=(50, h))
            num[c0:c0 + 50] = Tj[:h][ii].sum(1); den[c0:c0 + 50] = T[:h][ii].sum(1)
        boot[h] = num / den[:, None]
    for alpha in ALPHAS:
        res = {}
        for name, a_used in (("cs_shared", alpha / (K + 1) / 2), ("cs_single", alpha / 2 / 2)):
            bx = cumulative_share_boxes(Tj, T, B_BOUND, a_used)
            cov = (bx["L"] <= truth) & (truth <= bx["U"])
            res[name] = {"cov_H": cov[[h - 1 for h in HORIZONS]], "path": cov[T0_PATH - 1:],
                         "hw_H": ((bx["U"] - bx["L"]) / 2)[[h - 1 for h in HORIZONS]],
                         "hw_grid": ((bx["U"] - bx["L"]) / 2)[WIDTH_GRID - 1]}
        for kind in ("fixed_eb", "fixed_eb_bonf", "csranks_style"):
            hw = fixed_hw(se_all, t_all, alpha, kind)
            cov = (r_all - hw <= truth) & (truth <= r_all + hw)
            res[kind] = {"cov_H": cov[[h - 1 for h in HORIZONS]], "path": cov[T0_PATH - 1:],
                         "hw_H": hw[[h - 1 for h in HORIZONS]], "hw_grid": hw[WIDTH_GRID - 1]}
        for kind, a in (("bootstrap", alpha), ("bootstrap_bonf", alpha / K)):
            cov_l, hw_l = [], []
            for hi, h in enumerate(HORIZONS):
                lo, up = np.quantile(boot[h], a / 2, axis=0), np.quantile(boot[h], 1 - a / 2, axis=0)
                cov_l.append((lo <= truth[h - 1]) & (truth[h - 1] <= up)); hw_l.append((up - lo) / 2)
            res[kind] = {"cov_H": np.array(cov_l), "hw_H": np.array(hw_l)}
        out[alpha] = res
    return out


def cp_ci(k, n, conf=0.95):
    lo = 0.0 if k == 0 else _beta.ppf((1 - conf) / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta.ppf(1 - (1 - conf) / 2, k + 1, n - k)
    return float(lo), float(hi)


def summarize(setting, reps_out):
    R = len(reps_out); table = {}
    methods = list(reps_out[0][ALPHAS[0]].keys())
    for alpha in ALPHAS:
        table[str(alpha)] = {}
        for m in methods:
            cov_H = np.stack([r[alpha][m]["cov_H"] for r in reps_out])
            hw_H = np.stack([r[alpha][m]["hw_H"] for r in reps_out])
            e = {"horizons": {}}
            for hi, h in enumerate(HORIZONS):
                marg_rep = cov_H[:, hi].mean(1)                                # per-rep marginal coverage
                sim = cov_H[:, hi].all(1)
                bs = np.random.default_rng(1).choice(R, size=(2000, R)); mm = marg_rep[bs].mean(1)
                e["horizons"][h] = {
                    "marginal_coverage": float(marg_rep.mean()), "marginal_ci95_cluster_bootstrap": [float(np.quantile(mm, .025)), float(np.quantile(mm, .975))],
                    "simultaneous_coverage": float(sim.mean()), "simultaneous_ci95_clopper_pearson": cp_ci(int(sim.sum()), R),
                    "halfwidth_median": float(np.median(hw_H[:, hi])), "halfwidth_p90": float(np.quantile(hw_H[:, hi], 0.9))}
            peek = cov_H.all(axis=(1, 2)); e["peek4_simultaneous_coverage"] = float(peek.mean()); e["peek4_ci95_cp"] = cp_ci(int(peek.sum()), R)
            e["peek4_marginal_coverage"] = float(cov_H.all(1).mean())
            if "path" in reps_out[0][alpha][m]:
                path = np.stack([r[alpha][m]["path"] for r in reps_out])
                pm = path.all(1); ps = path.all((1, 2))
                e["pathwise_marginal_coverage"] = float(pm.mean()); e["pathwise_simultaneous_coverage"] = float(ps.mean())
                e["pathwise_simultaneous_ci95_cp"] = cp_ci(int(ps.sum()), R)
                e["pathwise_scope"] = f"all t in [{T0_PATH}, {N}], every t"
                e["pathwise_note"] = "valid use (anytime)" if m.startswith("cs_") else "CONTRAST ONLY: fixed-sample interval recomputed at every t (invalid use)"
                hg = np.stack([r[alpha][m]["hw_grid"] for r in reps_out])
                e["halfwidth_median_vs_t"] = {int(t): float(np.median(hg[:, i])) for i, t in enumerate(WIDTH_GRID)}
            table[str(alpha)][m] = e
    return table


def _init_worker(form):
    """Spawned workers do not inherit the boundary override; set it explicitly."""
    _cs._ACTIVE_BOUNDARY[0] = form


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=200); ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", default="results/e1"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--name", default=None, help="output file name (default: e1_results.json, or e1_simplified_boundary.json with --boundary simplified)")
    ap.add_argument("--boundary", default="exact", choices=["exact", "simplified"], help="stitched-boundary form of the confidence sequences; 'exact' is the production form, 'simplified' is the looser upper bound used only for the width comparison")
    a = ap.parse_args()
    a.name = a.name or ("e1_results.json" if a.boundary == "exact" else "e1_simplified_boundary.json")
    reps = 6 if a.pilot else a.reps
    out_dir = Path(a.out + ("_pilot" if a.pilot else "")); out_dir.mkdir(parents=True, exist_ok=True)
    with boundary_override(a.boundary):
        t0 = time.time(); result = {"config": {"K": K, "N": N, "B": B_BOUND, "horizons": HORIZONS, "alphas": ALPHAS, "reps": reps, "boot_B": BOOT_B,
                                              "seed0": SEED0, "boundary": _check_form(None), "cert_method": "N/A (coverage/width study)", "pathwise_t0": T0_PATH, "budget": {"cs_shared": "alpha/(K+1) per process", "cs_single": "alpha/2 per process (K=1, no multiplicity)",
                                              "fixed_*_bonf": "alpha/K", "csranks_style": "alpha/K", "bootstrap/fixed_eb": "alpha (uncorrected)"}}}
        for setting in ("stationary", "drift"):
            with ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(_check_form(None),)) as ex:
                reps_out = list(ex.map(one_rep, [(setting, r, 0 if setting == "stationary" else 1) for r in range(reps)]))
            result[setting] = summarize(setting, reps_out)
        result["wall_time_s"] = time.time() - t0
        json.dump(result, open(out_dir / a.name, "w"), indent=1)
        print("wrote", out_dir / a.name, "wall", round(result["wall_time_s"], 1))


if __name__ == "__main__":
    main()
