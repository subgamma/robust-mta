"""E8: instability decomposition.

One-at-a-time empirical decomposition: vary one source of instability, hold the rest at
their reference (day-30) values, and measure the rank-reversal rate against the
reference certified graph R_0. This is a tractable proxy, not a variance decomposition.

Primary (K=20; reference R_0 = linear model, Gamma=1.25, delta=0.005, as in e7):
  - sampling: bootstrap (B=200) the 109,221 journeys' linear masses; reversal rate is
    the fraction of (replicate, R_0 pair) combinations where the resampled point-estimate
    ratio disagrees in sign with R_0's certified direction.
  - model choice: for each of time_decay, last_touch and logistic_shap, the fraction of
    R_0 pairs whose point estimate disagrees in sign with R_0. Within-journey credit
    splitting is zero by construction on primary (every journey touches one campaign), so
    any nonzero value comes from each model's weight-scale convention.
  - observability: reused from `e7_interventions` G2 (naive_wrong_rate averaged over
    r in {1.25, 1.5, 2.0}).

Secondary (K=10, literature-convention linkage; 804/20,089 (4.0%) multi-campaign journeys):
  - model choice: the five materialized models' credit vectors used directly as mass
    (within-journey splitting is non-degenerate here), reference = linear.
  - DARNN seed: reversal rate of seeds 1-4 against seed 0.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from ..certify import Certification, active_cert_method, snapshot
from ..criteo import load_config
from ._primary_journeys import build_and_cache_primary_journeys, build_and_cache_secondary_journeys
from .e5_certified_fraction import MODELS as PRIMARY_MODELS
from .e5_certified_fraction import _fit_logistic_shap_primary, _model_masses
from .e7_interventions import (
    REFERENCE_GAMMA,
    _campaign_idx_and_n_touches,
    _certified_graph,
)
from ..cs import MultiCampaignCumulativeShareCS

CONFIG_PATH = "config/criteo.yaml"
CS_METHOD = "howard_eb_ratio"  
BOOTSTRAP_B = 200
SEED = 0
SECONDARY_MODELS = ("last_touch", "linear", "time_decay", "logistic_shap", "darnn")


def _direction(a_j: float, a_k: float) -> int:
    return 1 if a_j > a_k else (-1 if a_k > a_j else 0)


def _sign_matches_cert(a_j: float, a_k: float, cert: Certification) -> bool:
    d = _direction(a_j, a_k)
    if cert == Certification.J_OVER_K:
        return d == 1
    if cert == Certification.K_OVER_J:
        return d == -1
    return True  # unresolved reference pairs are not scored


def run(config_path: str = CONFIG_PATH, bootstrap_b: int = BOOTSTRAP_B,
        seed: int = SEED, e7_result: dict | None = None) -> dict:
    cfg = load_config(config_path)
    alpha = cfg["certification"]["alpha"]
    delta = cfg["certification"]["delta"]
    top_20 = cfg["campaigns"]["top_20"]
    K = len(top_20)
    rng = np.random.default_rng(seed)

    journeys, _ = build_and_cache_primary_journeys(config_path)
    campaign_idx, n_touches = _campaign_idx_and_n_touches(journeys)
    n_total = len(campaign_idx)

    fitted_logistic_shap = _fit_logistic_shap_primary(config_path, seed)
    masses = {m: _model_masses(journeys, m, K, fitted_logistic_shap)[0] for m in PRIMARY_MODELS}
    a_hat = {m: masses[m].sum(axis=0) / masses[m].sum() for m in PRIMARY_MODELS}

    Tj_lin = masses["linear"]
    T_lin = Tj_lin.sum(axis=1)
    mg0 = MultiCampaignCumulativeShareCS(K, alpha, 10.0)
    for i in range(n_total):
        mg0.update(Tj_lin[i], T_lin[i])
    box0 = snapshot(mg0)
    R0 = _certified_graph(box0, REFERENCE_GAMMA, delta, K)

    result = {"K_primary": K, "n_reference_certified": len(R0), "reference_model": "linear",
              "reference_gamma": REFERENCE_GAMMA, "cs_method": CS_METHOD,
              "beta": alpha / (K + 1), "alpha_used": alpha / (K + 1) / 2.0, "cs_running_intersection": False,
              "cert_method": active_cert_method(), "boundary": mg0.boundary}

    t0 = time.time()
    n_scored = 0
    n_reversed = 0
    for _ in range(bootstrap_b):
        idx = rng.integers(0, n_total, size=n_total)
        Tj_boot = Tj_lin[idx].sum(axis=0)
        a_boot = Tj_boot / Tj_boot.sum()
        for (j, k), cert in R0.items():
            n_scored += 1
            if not _sign_matches_cert(a_boot[j], a_boot[k], cert):
                n_reversed += 1
    result["sampling"] = {
        "bootstrap_b": bootstrap_b,
        "reversal_rate": n_reversed / n_scored if n_scored else None,
        "wall_time_s": time.time() - t0,
    }

    model_choice = {}
    for m in PRIMARY_MODELS:
        if m == "linear":
            continue
        n_scored = n_reversed = 0
        for (j, k), cert in R0.items():
            n_scored += 1
            if not _sign_matches_cert(a_hat[m][j], a_hat[m][k], cert):
                n_reversed += 1
        model_choice[m] = n_reversed / n_scored if n_scored else None
    result["model_choice_primary"] = {
        "reversal_rate_by_model": model_choice,
        "reversal_rate_mean": float(np.mean(list(model_choice.values()))),
        "note": "Within-journey credit SPLITTING is 0 by construction on primary (every journey touches "
                "exactly one campaign, so every model assigns it entirely to that campaign); any nonzero "
                "reversal rate above comes entirely from each model's own weight-scale convention "
                "(touch count / decay-weighted sum / one unit of conversion credit), not from attribution "
                "disagreement -- consistent with e6's finding that Gamma* does not exactly coincide across models.",
    }

    if e7_result is not None:
        naive_rates = [e7_result["G2"][r]["naive_wrong_rate_mean"] for r in e7_result["G2"]]
        result["observability_primary"] = {
            "source": "e7_interventions G2 (differential loss), naive_wrong_rate_mean, averaged over r",
            "reversal_rate_mean": float(np.mean(naive_rates)),
            "reversal_rate_by_r": e7_result["G2"],
        }
    else:
        result["observability_primary"] = {"note": "e7_result not provided; run e7_interventions first and pass its output"}

    t0 = time.time()
    sec_journeys, top_10 = build_and_cache_secondary_journeys(config_path)
    n_sec = len(sec_journeys)
    n_multi = sum(1 for j in sec_journeys if len(set(t.campaign for t in j.touches)) > 1)

    sec_credits = {}
    for m in ("last_touch", "linear", "time_decay", "logistic_shap"):
        df = pl.read_parquet(f"results/credits_secondary/{m}/0.parquet").sort("journey_idx")
        cols = [c for c in df.columns if c.startswith("credit_")]
        sec_credits[m] = df.select(cols).to_numpy()
    darnn_seeds = {}
    for s in range(5):
        df = pl.read_parquet(f"results/credits_secondary/darnn/{s}.parquet").sort("journey_idx")
        cols = [c for c in df.columns if c.startswith("credit_")]
        darnn_seeds[s] = df.select(cols).to_numpy()
    sec_credits["darnn"] = darnn_seeds[0]

    K_sec = sec_credits["linear"].shape[1]
    a_hat_sec = {m: c.sum(axis=0) / c.sum() for m, c in sec_credits.items()}
    ref_a = a_hat_sec["linear"]
    ref_pairs = [(j, k, 1 if ref_a[j] > ref_a[k] else (-1 if ref_a[k] > ref_a[j] else 0))
                 for j, k in combinations(range(K_sec), 2) if ref_a[j] != ref_a[k]]

    model_choice_sec = {}
    for m in ("last_touch", "time_decay", "logistic_shap", "darnn"):
        n_scored = n_reversed = 0
        for j, k, d in ref_pairs:
            n_scored += 1
            d2 = 1 if a_hat_sec[m][j] > a_hat_sec[m][k] else (-1 if a_hat_sec[m][k] > a_hat_sec[m][j] else 0)
            if d2 != d:
                n_reversed += 1
        model_choice_sec[m] = n_reversed / n_scored if n_scored else None

    a_hat_seeds = {s: c.sum(axis=0) / c.sum() for s, c in darnn_seeds.items()}
    ref_a_darnn = a_hat_seeds[0]
    ref_pairs_darnn = [(j, k, 1 if ref_a_darnn[j] > ref_a_darnn[k] else (-1 if ref_a_darnn[k] > ref_a_darnn[j] else 0))
                       for j, k in combinations(range(K_sec), 2) if ref_a_darnn[j] != ref_a_darnn[k]]
    darnn_seed_reversal = {}
    for s in range(1, 5):
        n_scored = n_reversed = 0
        for j, k, d in ref_pairs_darnn:
            n_scored += 1
            d2 = 1 if a_hat_seeds[s][j] > a_hat_seeds[s][k] else (-1 if a_hat_seeds[s][k] > a_hat_seeds[s][j] else 0)
            if d2 != d:
                n_reversed += 1
        darnn_seed_reversal[s] = n_reversed / n_scored if n_scored else None

    result["secondary"] = {
        "K": K_sec,
        "n_journeys": n_sec,
        "n_multi_campaign": n_multi,
        "multi_campaign_fraction": n_multi / n_sec,
        "effective_n_note": f"{n_multi}/{n_sec} ({n_multi/n_sec:.1%}) journeys are multi-campaign; "
                            f"within-journey model-choice differences are only possible on that subset.",
        "model_choice_reversal_rate_by_model": model_choice_sec,
        "darnn_seed_reversal_rate": darnn_seed_reversal,
        "darnn_seed_reversal_rate_mean": float(np.mean(list(darnn_seed_reversal.values()))),
        "wall_time_s": time.time() - t0,
    }

    return result


def write_outputs(result: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e8_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    csv_lines = ["source,detail,reversal_rate"]
    csv_lines.append(f"sampling,bootstrap B={result['sampling']['bootstrap_b']},{result['sampling']['reversal_rate']:.4f}")
    for m, v in result["model_choice_primary"]["reversal_rate_by_model"].items():
        csv_lines.append(f"model_choice (primary),{m},{v:.4f}")
    if "reversal_rate_mean" in result["observability_primary"]:
        csv_lines.append(f"observability (primary, G2),mean over r,{result['observability_primary']['reversal_rate_mean']:.4f}")
    for m, v in result["secondary"]["model_choice_reversal_rate_by_model"].items():
        csv_lines.append(f"model_choice (secondary),{m},{v:.4f}")
    for s, v in result["secondary"]["darnn_seed_reversal_rate"].items():
        csv_lines.append(f"darnn_seed (secondary),seed {s} vs 0,{v:.4f}")
    with open(f"{out_dir}/table5.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")


if __name__ == "__main__":
    e7_path = Path("results/e7/e7_results.json")
    if e7_path.exists():
        with open(e7_path) as f:
            e7_res = json.load(f)
        # JSON round-trip stringifies the G2 dict's float keys
        e7_res["G2"] = {float(k): v for k, v in e7_res["G2"].items()}
    else:
        from .e7_interventions import run as run_e7
        e7_res = run_e7()
    result = run(e7_result=e7_res)
    write_outputs(result, "results/e8")
    print("wrote results/e8/e8_results.json")
