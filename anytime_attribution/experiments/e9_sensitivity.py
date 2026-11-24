"""E9: delta/Gamma sensitivity and |Q_j - A_j|, SD(S)/E[S] diagnostics.

Primary construction (K=20), full monitored window. This is a full-sample diagnostic,
so it uses a single checkpoint at n_total rather than the E5-E8 schedules.

1. `sensitivity`: delta x Gamma certified-pair counts at day 30 for linear and
   time_decay only (the guaranteed models). One CS pass per model: the box
   [L,U] (`cs.MultiCampaignCumulativeShareCS`) does not depend on (delta, Gamma), only
   `certify.robust_rule` does. The naive rule is the Gamma=1 row of this grid.

2. `distortion`: per-campaign observed |q_j - a_j| (the empirical proxy for |Q_j - A_j|)
   and each model's SD(S)/E[S], with S the model's own mass scale (e5 convention). A
   descriptive statistic of the observed stream: no Gamma, CS or identified-set claim.
   Reported for all four frozen models and labelled "descriptive". Since
   Q_j - A_j = -Cov(S, Z_j)/E[S], a model with constant S has Q_j = A_j exactly; this is
   checked for last_touch and logistic_shap.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

from ..certify import Certification, PRODUCTION_CERT_METHOD, box_rule
from ..cs import MultiCampaignCumulativeShareCS, boundary_override
from ..criteo import load_config
from ._primary_journeys import build_and_cache_primary_journeys
from .e5_certified_fraction import CS_METHOD, _cs_history, _fit_logistic_shap_primary, _model_masses

CONFIG_PATH = "config/criteo.yaml"
DELTA_GRID = (0.0, 0.0025, 0.005)
GAMMA_GRID = (1.0, 1.25, 1.5, 2.0)
SENSITIVITY_MODELS = ("linear", "time_decay")  # guaranteed aggregate-share models only
DISTORTION_MODELS = ("linear", "time_decay", "last_touch", "logistic_shap")
GUARANTEED_MODELS = {"linear", "time_decay"}


def _sensitivity_grid(box, K: int, delta_grid, gamma_grid, cert_method=None) -> dict:
    """Certified-pair counts over the delta x Gamma grid from one day-30 CS box [L, U]; the naive rule is the Gamma=1.0
    row.
    """
    pairs = list(combinations(range(K), 2))
    grid = {}
    for delta in delta_grid:
        grid[delta] = {}
        for Gamma in gamma_grid:
            n_j_over_k = n_k_over_j = n_unresolved = 0
            for j, k in pairs:
                cert = box_rule(box, Gamma, delta, j, k, method=cert_method)
                if cert == Certification.J_OVER_K:
                    n_j_over_k += 1
                elif cert == Certification.K_OVER_J:
                    n_k_over_j += 1
                else:
                    n_unresolved += 1
            grid[delta][Gamma] = {
                "n_certified": n_j_over_k + n_k_over_j,
                "n_j_over_k": n_j_over_k,
                "n_k_over_j": n_k_over_j,
                "n_unresolved": n_unresolved,
                "n_pairs": len(pairs),
                "certified_fraction": (n_j_over_k + n_k_over_j) / len(pairs),
                "is_naive_definition": Gamma == 1.0,
            }
    return grid


def _distortion_row(Tj_all: np.ndarray, top_20: list) -> dict:
    """Observed a_j, q_j, |q_j - a_j| per campaign and SD(S)/E[S] from one model's (n_journeys, K) raw-mass matrix (e5
    `_model_masses` convention).
    """
    S = Tj_all.sum(axis=1)
    if np.any(S <= 0):
        raise ValueError("zero-mass journey encountered; per-journey share undefined (see the paper)")
    a = Tj_all.sum(axis=0) / Tj_all.sum()
    Z = Tj_all / S[:, None]
    q = Z.mean(axis=0)
    sd_over_mean_S = float(np.std(S, ddof=0) / np.mean(S))
    per_campaign = [
        {
            "campaign_idx": j,
            "campaign_id": top_20[j],
            "a_j": float(a[j]),
            "q_j": float(q[j]),
            "abs_diff_q_minus_a": float(abs(q[j] - a[j])),
        }
        for j in range(len(top_20))
    ]
    return {"per_campaign": per_campaign, "sd_S_over_mean_S": sd_over_mean_S}


def run(config_path: str = CONFIG_PATH, delta_grid=DELTA_GRID, gamma_grid=GAMMA_GRID, seed: int = 0,
        cert_method: str | None = None, boundary: str | None = None) -> dict:
    cert_method = PRODUCTION_CERT_METHOD if cert_method is None else cert_method
    if boundary is not None:
        with boundary_override(boundary):
            return run(config_path, delta_grid, gamma_grid, seed, cert_method, None)
    cfg = load_config(config_path)
    alpha = cfg["certification"]["alpha"]
    top_20 = cfg["campaigns"]["top_20"]
    K = len(top_20)
    beta = alpha / (K + 1)

    journeys, top_20_check = build_and_cache_primary_journeys(config_path)
    assert top_20_check == top_20
    n_total = len(journeys)
    checkpoints = np.array([n_total])

    result = {
        "K": K, "alpha": alpha, "n_total": n_total, "delta_grid": list(delta_grid), "gamma_grid": list(gamma_grid),
        "cs_method": CS_METHOD, "beta": beta, "alpha_used": beta / 2.0, "cs_running_intersection": False,
        "sensitivity_models": list(SENSITIVITY_MODELS), "distortion_models": list(DISTORTION_MODELS),
        "naive_definition": f"{cert_method} rule at Gamma=1.0 (kickoff decision 4)",
        "boundary": MultiCampaignCumulativeShareCS(K, alpha, 1.0).boundary, "cert_method": cert_method,
    }

    fitted_logistic_shap = _fit_logistic_shap_primary(config_path, seed)

    masses = {}
    for model in DISTORTION_MODELS:
        Tj_all, B = _model_masses(journeys, model, K, fitted_logistic_shap)
        masses[model] = Tj_all

    sensitivity = {}
    for model in SENSITIVITY_MODELS:
        Tj_all = masses[model]
        T_all = Tj_all.sum(axis=1)
        Tj_bound = 10.0
        box = _cs_history(Tj_all, T_all, alpha, Tj_bound, checkpoints, return_boxes=True)[0]
        L, U = box.L, box.U
        sensitivity[model] = {
            "guaranteed": True,
            "L_day30": L.tolist(),
            "U_day30": U.tolist(),
            "grid": _sensitivity_grid(box, K, delta_grid, gamma_grid, cert_method),
        }
    result["sensitivity"] = sensitivity

    distortion = {}
    for model in DISTORTION_MODELS:
        row = _distortion_row(masses[model], top_20)
        row["guarantee_label"] = "descriptive"  # observed a_j/q_j directly; no CS, no Gamma, no identified set here
        row["guaranteed_aggregate_share_elsewhere"] = model in GUARANTEED_MODELS
        distortion[model] = row
    result["distortion"] = distortion

    return result


def write_outputs(result: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e9_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    sens_lines = ["model,guaranteed,delta,Gamma,n_certified,n_j_over_k,n_k_over_j,n_unresolved,n_pairs,"
                  "certified_fraction,is_naive_definition"]
    for model, mres in result["sensitivity"].items():
        for delta, drow in mres["grid"].items():
            for Gamma, cell in drow.items():
                sens_lines.append(
                    f"{model},{mres['guaranteed']},{delta},{Gamma},{cell['n_certified']},{cell['n_j_over_k']},"
                    f"{cell['n_k_over_j']},{cell['n_unresolved']},{cell['n_pairs']},"
                    f"{cell['certified_fraction']:.4f},{cell['is_naive_definition']}"
                )
    with open(f"{out_dir}/table6_sensitivity.csv", "w") as f:
        f.write("\n".join(sens_lines) + "\n")

    dist_lines = ["model,guarantee_label,campaign_idx,campaign_id,a_j,q_j,abs_diff_q_minus_a,sd_S_over_mean_S"]
    for model, row in result["distortion"].items():
        for pc in row["per_campaign"]:
            dist_lines.append(
                f"{model},{row['guarantee_label']},{pc['campaign_idx']},{pc['campaign_id']},"
                f"{pc['a_j']:.6f},{pc['q_j']:.6f},{pc['abs_diff_q_minus_a']:.6f},{row['sd_S_over_mean_S']:.6f}"
            )
    with open(f"{out_dir}/table6_distortion.csv", "w") as f:
        f.write("\n".join(dist_lines) + "\n")


if __name__ == "__main__":
    result = run()
    write_outputs(result, "results/e9")
    for model, mres in result["sensitivity"].items():
        print(f"{model} (guaranteed): delta=0.005 Gamma grid certified fractions:",
              {g: c["certified_fraction"] for g, c in mres["grid"][0.005].items()})
    for model, row in result["distortion"].items():
        diffs = [pc["abs_diff_q_minus_a"] for pc in row["per_campaign"]]
        print(f"{model}: max|q-a|={max(diffs):.4f}, SD(S)/E[S]={row['sd_S_over_mean_S']:.4f}")
