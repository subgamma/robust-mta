"""E1: simultaneous coverage under continuous monitoring (synthetic, verification only).

One synthetic stream of N_FULL=20,000 conversions (K=10) is generated per
replication. `L` is the number of looks during that period. For L in
REPORTED_LOOKS={1,10,100,1000}, look times are L log-spaced points between N0=500
and N_FULL; a fixed FINEST_L=1000-point grid is computed once and smaller L index-subsample
it. L=1 is a single look at N_FULL, the classic fixed-sample setting.

Two tables:
- MARGINAL: coverage of one campaign's interval, held across all L look times, averaged over the K campaigns and replications.
- SIMULTANEOUS: whether all K campaigns' intervals hold across all L look times.

Multiplicity correction per method:
- bootstrap, fixed_eb: none (alpha per campaign).
- csranks_style: alpha/K (see baselines.py).
- naive_cs: none; K independent single-campaign `cs.MultiCampaignCumulativeShareCS(K_for_beta=1, ALPHA, B)`.
- robust_cs: beta = alpha/(K+1) and time-uniform; one joint `MultiCampaignCumulativeShareCS(K, ALPHA, B)`.
  Simultaneous coverage should stay >= 0.95 at every L.

naive_cs and robust_cs use the production cumulative-share CS. Coverage is checked
at every finest-schedule checkpoint, since the radius is a closed-form function of
(t, V_hat_t) and coverage at N_FULL does not imply coverage at earlier looks.

Panel (ii), drift, reruns naive_cs and robust_cs on a population whose per-campaign
share drifts over the stream (matched to the Criteo diagnostic: most campaigns move
>= 10% relative, one decays toward 0) and checks coverage of the exact time-varying
cumulative share a_{j,t}. This exercises Assumption A3 (no constancy of shares).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..baselines import csranks_style_ci, fixed_eb_ci
from ..cs import MultiCampaignCumulativeShareCS
from ..estimands import campaign_masses
from ..journeys import apply_cap
from ..synth import generate_synthetic_journeys, true_aggregate_share

K = 10
# Campaign weights with 3:1 max:min heterogeneity.
WEIGHTS = np.linspace(1.5, 0.5, K)
A_TRUE = true_aggregate_share(WEIGHTS)
MAX_TOUCHES = 10
B_BOUND = float(MAX_TOUCHES)
ALPHA = 0.05
N0 = 500
N_FULL = 20_000
FINEST_L = 1000
REPORTED_LOOKS = (1, 10, 100, 1000)
SEED = 0

# N_REPS and BOOTSTRAP_B are set together for tractability: the 1000-look schedule needs 1000 bootstrap evaluations
# per replication.
N_REPS = 60
BOOTSTRAP_B = 100

METHODS = ["bootstrap", "fixed_eb", "csranks_style", "naive_cs", "robust_cs"]
METHOD_BONFERRONI = {
    "bootstrap": "no",
    "fixed_eb": "no",
    "csranks_style": "yes (alpha/K)",
    "naive_cs": "no",
    "robust_cs": "yes (beta=alpha/(K+1), Prop 9)",
}


def _finest_schedule(n0: int = N0, n_full: int = N_FULL, num: int = FINEST_L) -> np.ndarray:
    return np.array(sorted(set(np.geomspace(n0, n_full, num).astype(int).tolist())))


def _subschedule_indices(finest: np.ndarray, L: int) -> np.ndarray:
    """Indices into `finest` for an L-point subsample, always including the last point (N_FULL)."""
    n = len(finest)
    if L == 1:
        return np.array([n - 1])
    idx = np.unique(np.linspace(0, n - 1, L).round().astype(int))
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _bootstrap_ci_batched(
    Tj_hist: np.ndarray, T_hist: np.ndarray, alpha: float, B: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap CI for all K campaigns at once.

    One journey-level resample index matrix is drawn and shared by every campaign
    column, so each campaign's CI remains a valid marginal percentile-bootstrap CI
    while avoiding K independent resample draws (about 10x cheaper).
    """
    n = len(T_hist)
    idx = rng.integers(0, n, size=(B, n))
    Tj_r_sum = Tj_hist[idx].sum(axis=1)
    T_r_sum = T_hist[idx].sum(axis=1)
    ratios = Tj_r_sum / T_r_sum[:, None]
    lo = np.quantile(ratios, alpha / 2, axis=0)
    hi = np.quantile(ratios, 1 - alpha / 2, axis=0)
    return lo, hi


def _stream_Tj_T(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Journeys capped at MAX_TOUCHES (A4, as in the Criteo analysis) so B_BOUND reflects the capped increment scale."""
    journeys = generate_synthetic_journeys(n, WEIGHTS, rng)
    journeys = [apply_cap(j, lookback_days=7.0, max_touches=MAX_TOUCHES) for j in journeys]
    Tj = np.zeros((n, K))
    T = np.zeros(n)
    for i, j in enumerate(journeys):
        m = campaign_masses(j, K)
        Tj[i] = m
        T[i] = m.sum()
    return Tj, T


def run(
    n_reps: int = N_REPS,
    seed: int = SEED,
    reported_looks: tuple[int, ...] = REPORTED_LOOKS,
    n0: int = N0,
    n_full: int = N_FULL,
    finest_l: int = FINEST_L,
    bootstrap_b: int = BOOTSTRAP_B,
) -> dict:
    rng = np.random.default_rng(seed)
    finest = _finest_schedule(n0, n_full, finest_l)
    sub_idx = {L: _subschedule_indices(finest, L) for L in reported_looks}

    baseline_methods = ["bootstrap", "fixed_eb", "csranks_style"]
    cs_methods = ["naive_cs", "robust_cs"]

    marginal_sum = {m: {L: 0.0 for L in reported_looks} for m in METHODS}
    simultaneous_sum = {m: {L: 0 for L in reported_looks} for m in METHODS}
    t0 = time.time()

    for rep in range(n_reps):
        Tj, T = _stream_Tj_T(n_full, rng)

        covered = {m: np.zeros((K, len(finest)), dtype=bool) for m in baseline_methods}
        for ci, n in enumerate(finest):
            Tj_hist = Tj[:n]
            T_hist = T[:n]
            lo_b, hi_b = _bootstrap_ci_batched(Tj_hist, T_hist, ALPHA, bootstrap_b, rng)
            covered["bootstrap"][:, ci] = (lo_b <= A_TRUE) & (A_TRUE <= hi_b)
            for j in range(K):
                lo_f, hi_f = fixed_eb_ci(Tj_hist[:, j], T_hist, ALPHA, bound=B_BOUND)
                covered["fixed_eb"][j, ci] = lo_f <= A_TRUE[j] <= hi_f
                lo_c, hi_c = csranks_style_ci(Tj_hist[:, j], T_hist, ALPHA, n_campaigns=K)
                covered["csranks_style"][j, ci] = lo_c <= A_TRUE[j] <= hi_c

        for method in baseline_methods:
            for L in reported_looks:
                sticky = covered[method][:, sub_idx[L]].all(axis=1)  # sticky through schedule_L, per campaign
                marginal_sum[method][L] += sticky.mean()
                simultaneous_sum[method][L] += int(sticky.all())

        mg_robust = MultiCampaignCumulativeShareCS(K, ALPHA, B_BOUND)
        mg_naive = [MultiCampaignCumulativeShareCS(1, ALPHA, B_BOUND) for _ in range(K)]
        covered_robust = np.zeros((K, len(finest)), dtype=bool)
        covered_naive = np.zeros((K, len(finest)), dtype=bool)
        fi = 0
        finest_set = set(finest.tolist())
        for i in range(n_full):
            mg_robust.update(Tj[i], T[i])
            for j in range(K):
                mg_naive[j].update(Tj[i, j:j + 1], T[i])
            if (i + 1) in finest_set:
                Lr, Ur = mg_robust.intervals()
                covered_robust[:, fi] = (Lr <= A_TRUE) & (A_TRUE <= Ur)
                for j in range(K):
                    Ln, Un = mg_naive[j].intervals()
                    covered_naive[j, fi] = (Ln[0] <= A_TRUE[j]) & (A_TRUE[j] <= Un[0])
                fi += 1

        for method, covered_arr in [("naive_cs", covered_naive), ("robust_cs", covered_robust)]:
            for L in reported_looks:
                sticky = covered_arr[:, sub_idx[L]].all(axis=1)
                marginal_sum[method][L] += sticky.mean()
                simultaneous_sum[method][L] += int(sticky.all())


    marginal_table = {m: {L: marginal_sum[m][L] / n_reps for L in reported_looks} for m in METHODS}
    simultaneous_table = {m: {L: simultaneous_sum[m][L] / n_reps for L in reported_looks} for m in METHODS}
    return {
        "marginal_table": marginal_table,
        "simultaneous_table": simultaneous_table,
        "bonferroni_status": METHOD_BONFERRONI,
        "n_reps": n_reps,
        "n0": n0,
        "n_full": n_full,
        "finest_l": finest_l,
        "bootstrap_b": bootstrap_b,
        "alpha": ALPHA,
        "K": K,
        "seed": seed,
        "wall_time_s": time.time() - t0,
    }


DRIFT_P_START = np.array([0.30, 0.13, 0.13, 0.10, 0.10, 0.08, 0.06, 0.04, 0.03, 0.03])
DRIFT_P_END = np.array([0.30, 0.17, 0.09, 0.13, 0.02, 0.10, 0.05, 0.05, 0.04, 0.05])
DRIFT_P_END = DRIFT_P_END / DRIFT_P_END.sum()
assert len(DRIFT_P_START) == K and len(DRIFT_P_END) == K


def _gen_drift_stream(K: int, p_start: np.ndarray, p_end: np.ndarray, rng: np.random.Generator, n: int):
    """Single-campaign-per-observation stream whose campaign probability drifts linearly from p_start to p_end.

    Weight per observation ~ Uniform{1,...,MAX_TOUCHES}. Returns (Tj, T, p_t) with the
    exact per-step probability p_t, so the true cumulative share is known exactly.
    """
    frac = np.linspace(0.0, 1.0, n)[:, None]
    p_t = p_start[None, :] * (1 - frac) + p_end[None, :] * frac
    p_t = p_t / p_t.sum(axis=1, keepdims=True)
    campaigns = np.array([rng.choice(K, p=p_t[i]) for i in range(n)])
    weights = rng.integers(1, MAX_TOUCHES + 1, size=n).astype(float)
    Tj = np.zeros((n, K))
    Tj[np.arange(n), campaigns] = weights
    T = weights
    return Tj, T, p_t


def run_drift(
    n_reps: int = N_REPS,
    seed: int = SEED,
    reported_looks: tuple[int, ...] = REPORTED_LOOKS,
    n0: int = N0,
    n_full: int = N_FULL,
    finest_l: int = FINEST_L,
) -> dict:
    """Panel (ii): naive_cs/robust_cs coverage of the exact time-varying cumulative share under drift."""
    rng = np.random.default_rng(seed + 5000)
    finest = _finest_schedule(n0, n_full, finest_l)
    sub_idx = {L: _subschedule_indices(finest, L) for L in reported_looks}
    finest_set = set(finest.tolist())

    marginal_sum = {m: {L: 0.0 for L in reported_looks} for m in ("naive_cs", "robust_cs")}
    simultaneous_sum = {m: {L: 0 for L in reported_looks} for m in ("naive_cs", "robust_cs")}
    t0 = time.time()

    for rep in range(n_reps):
        Tj, T, p_t = _gen_drift_stream(K, DRIFT_P_START, DRIFT_P_END, rng, n_full)
        true_a_t = np.cumsum(p_t, axis=0) / np.arange(1, n_full + 1)[:, None]

        mg_robust = MultiCampaignCumulativeShareCS(K, ALPHA, B_BOUND)
        mg_naive = [MultiCampaignCumulativeShareCS(1, ALPHA, B_BOUND) for _ in range(K)]
        covered_robust = np.zeros((K, len(finest)), dtype=bool)
        covered_naive = np.zeros((K, len(finest)), dtype=bool)
        fi = 0
        for i in range(n_full):
            mg_robust.update(Tj[i], T[i])
            for j in range(K):
                mg_naive[j].update(Tj[i, j:j + 1], T[i])
            if (i + 1) in finest_set:
                true_a = true_a_t[i]
                Lr, Ur = mg_robust.intervals()
                covered_robust[:, fi] = (Lr <= true_a) & (true_a <= Ur)
                for j in range(K):
                    Ln, Un = mg_naive[j].intervals()
                    covered_naive[j, fi] = (Ln[0] <= true_a[j]) & (true_a[j] <= Un[0])
                fi += 1

        for method, covered_arr in [("naive_cs", covered_naive), ("robust_cs", covered_robust)]:
            for L in reported_looks:
                sticky = covered_arr[:, sub_idx[L]].all(axis=1)
                marginal_sum[method][L] += sticky.mean()
                simultaneous_sum[method][L] += int(sticky.all())


    marginal_table = {m: {L: marginal_sum[m][L] / n_reps for L in reported_looks} for m in ("naive_cs", "robust_cs")}
    simultaneous_table = {m: {L: simultaneous_sum[m][L] / n_reps for L in reported_looks} for m in ("naive_cs", "robust_cs")}
    return {
        "marginal_table": marginal_table,
        "simultaneous_table": simultaneous_table,
        "p_start": DRIFT_P_START.tolist(),
        "p_end": DRIFT_P_END.tolist(),
        "relative_move": (np.abs(DRIFT_P_END - DRIFT_P_START) / DRIFT_P_START).tolist(),
        "n_reps": n_reps,
        "n0": n0,
        "n_full": n_full,
        "finest_l": finest_l,
        "alpha": ALPHA,
        "K": K,
        "seed": seed,
        "wall_time_s": time.time() - t0,
    }


def write_outputs(result: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e1_table1.json", "w") as f:
        json.dump(result, f, indent=2)

    methods_label = {
        "bootstrap": "Bootstrap (fixed-sample)",
        "fixed_eb": "Fixed EB (fixed-sample)",
        "csranks_style": "csranks-style (fixed-sample)",
        "naive_cs": "Naive CS (anytime)",
        "robust_cs": "Robust CS (anytime)",
    }
    looks = REPORTED_LOOKS
    bonf = result["bonferroni_status"]
    note = (
        f"# L = number of looks (peek times), NOT sample size; a single stream of "
        f"N_FULL={result['n_full']} conversions per replication, looks log-spaced between "
        f"n0={result['n0']} and N_FULL (L=1 = one look, at N_FULL). naive_cs/robust_cs (the "
        f"production Howard et al. 2021 ratio CS) are checked at EVERY finest-schedule checkpoint "
        f"(sticky-AND reduction per L, same convention as the fixed-sample baselines) -- their coverage "
        f"can in principle differ across L for this construction (see module docstring), unlike the old "
        f"WSR-betting mechanism's explicit sticky-rejection guarantee. Bonferroni column states whether "
        f"each method internally corrects for the K={result['K']} simultaneous test."
    )

    for table_key, out_name in [("marginal_table", "table1_marginal"), ("simultaneous_table", "table1_simultaneous")]:
        table = result[table_key]
        csv_lines = [note, "method,bonferroni," + ",".join(f"looks_{L}" for L in looks)]
        for m in METHODS:
            row = [methods_label[m], bonf[m]] + [f"{table[m][L]:.4f}" for L in looks]
            csv_lines.append(",".join(row))
        with open(f"{out_dir}/{out_name}.csv", "w") as f:
            f.write("\n".join(csv_lines) + "\n")

        tex_lines = [
            r"% " + note,
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            "Method & Bonferroni & " + " & ".join(f"$L={L}$" for L in looks) + r" \\",
            r"\midrule",
        ]
        for m in METHODS:
            row = [methods_label[m], bonf[m]] + [f"{table[m][L]:.3f}" for L in looks]
            tex_lines.append(" & ".join(row) + r" \\")
        tex_lines += [r"\bottomrule", r"\end{tabular}"]
        with open(f"{out_dir}/{out_name}.tex", "w") as f:
            f.write("\n".join(tex_lines) + "\n")


def write_drift_outputs(result: dict, out_dir: str) -> None:
    """Panel (ii) CSVs, separate from the constant-share tables of panel (i)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e1_table1_drift.json", "w") as f:
        json.dump(result, f, indent=2)

    looks = REPORTED_LOOKS
    note = (
        f"# Panel (ii) drift: naive_cs/robust_cs only (fixed-sample baselines omitted -- never "
        f"anytime-valid regardless of drift). p_start={result['p_start']}, p_end={result['p_end']}, "
        f"relative_move={[round(v, 3) for v in result['relative_move']]}. Coverage is of the TRUE, "
        f"exactly-known time-varying cumulative share a_{{j,t}} (Assumption A3), not a "
        f"fixed constant."
    )
    methods_label = {"naive_cs": "Naive CS (anytime, no Bonferroni)", "robust_cs": "Robust CS (anytime, Prop 9)"}
    for table_key, out_name in [("marginal_table", "table1_drift_marginal"), ("simultaneous_table", "table1_drift_simultaneous")]:
        table = result[table_key]
        csv_lines = [note, "method," + ",".join(f"looks_{L}" for L in looks)]
        for m in ("naive_cs", "robust_cs"):
            row = [methods_label[m]] + [f"{table[m][L]:.4f}" for L in looks]
            csv_lines.append(",".join(row))
        with open(f"{out_dir}/{out_name}.csv", "w") as f:
            f.write("\n".join(csv_lines) + "\n")


if __name__ == "__main__":
    result = run()
    print(json.dumps({"marginal": result["marginal_table"], "simultaneous": result["simultaneous_table"]}, indent=2))
    write_outputs(result, "results/e1")

    drift_result = run_drift()
    print(json.dumps({"marginal": drift_result["marginal_table"], "simultaneous": drift_result["simultaneous_table"]}, indent=2))
    write_drift_outputs(drift_result, "results/e1")
