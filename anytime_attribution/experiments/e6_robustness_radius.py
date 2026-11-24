"""E6: robustness radius.

Primary (K=20): Gamma* for the guaranteed models and a fragility ranking. Secondary
(K=10, literature-convention linkage): Gamma_hat_sens for the empirical model.

Gamma*_jk (`certify.gamma_star`) is the largest Gamma at which the robust rule still
certifies j>k from the day-30 CS box. It is computed separately for linear and
time_decay; their Gamma* distributions are compared, not assumed equal, even though
their per-journey credit split coincides.

Gamma_hat_sens (`certify.gamma_hat_sens`) is computed for last-touch only: logistic+SHAP
and DARNN would need refitting on re-thinned journeys. On the secondary construction the
envelope is informative only for the 804/20,089 (4.0%) multi-campaign journeys, since the
rest are single-campaign. Pairs receive a Gamma_hat_sens evaluation by a model-agnostic
raw-gap threshold (`_raw_gap_pairs`: |a_hat_j - a_hat_k| > delta, using last-touch mass),
not by `gamma_star`.

Fragility ranking: a campaign's most fragile certified relationship is min_k Gamma*_jk
over the k it is certified against (either direction); campaigns are ranked ascending.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np

from ..certify import Certification, PRODUCTION_CERT_METHOD, box_gamma_star, box_rule, gamma_hat_sens
from ..cs import boundary_override
from ..criteo import load_config
from ..estimands import per_journey_shares
from ._primary_journeys import (
    build_and_cache_primary_journeys,
    build_and_cache_secondary_journeys,
    last_touch_journey,
)
from ..cs import MultiCampaignCumulativeShareCS
from .e5_certified_fraction import CS_METHOD, _cs_history, _model_masses

CONFIG_PATH = "config/criteo.yaml"
GAMMA_HAT_GRID = (1.25, 1.5)
LAST_TOUCH_SUBSAMPLE_N = 5000
SEED = 0


def _last_touch_credit_fn(journey, K):
    return per_journey_shares(last_touch_journey(journey), K)


GAMMA_STAR_TOL = 1e-4  # tolerance on Gamma


def _gamma_star_distribution(journeys, model: str, K: int, delta: float, alpha: float,
                             cert_method: str | None = None) -> dict:
    """Gamma*_jk for every pair under the same certification method as `certify.box_rule`.

    Also checks Gamma*-rule consistency: the rule certifies at Gamma* and not at Gamma* + 2*tol.
    """
    Tj_all, B = _model_masses(journeys, model, K)
    T_all = Tj_all.sum(axis=1)
    n_total = len(journeys)
    box = _cs_history(Tj_all, T_all, alpha, B, np.array([n_total]), return_boxes=True)[0]

    pairs = list(combinations(range(K), 2))
    gstars, bad = {}, []
    for j, k in pairs:
        g_jk = box_gamma_star(box, delta, j, k, method=cert_method, tol=GAMMA_STAR_TOL)
        g_kj = box_gamma_star(box, delta, k, j, method=cert_method, tol=GAMMA_STAR_TOL)
        g = g_jk if g_jk is not None else g_kj
        gstars[f"{j}-{k}"] = g
        for (a, b, gg) in ((j, k, g_jk), (k, j, g_kj)):
            if gg is None:
                continue
            ok_in = box_rule(box, gg, delta, a, b, method=cert_method) == Certification.J_OVER_K
            ok_out = box_rule(box, gg + 2 * GAMMA_STAR_TOL, delta, a, b, method=cert_method) != Certification.J_OVER_K or gg >= 1000.0
            if not (ok_in and ok_out):
                bad.append({"pair": f"{a}-{b}", "gamma_star": gg, "certifies_at": ok_in, "fails_above": ok_out})
    return gstars, box, bad


def _raw_gap_pairs(journeys, model: str, K: int, delta: float) -> list[tuple[int, int]]:
    """Pairs whose raw point-estimate gap exceeds delta, using the model's own mass (no robust_rule or Gamma* involved)."""
    Tj_all, _ = _model_masses(journeys, model, K)
    a_hat = Tj_all.sum(axis=0) / Tj_all.sum()
    return [(j, k) for j, k in combinations(range(K), 2) if abs(a_hat[j] - a_hat[k]) > delta]


def _fragility_ranking(gstars: dict, K: int) -> list[dict]:
    per_campaign = {c: [] for c in range(K)}
    for pair_key, g in gstars.items():
        if g is None:
            continue
        j, k = (int(x) for x in pair_key.split("-"))
        per_campaign[j].append(g)
        per_campaign[k].append(g)
    ranking = [
        {"campaign_idx": c, "min_gamma_star": min(vals) if vals else None, "n_certified_pairs": len(vals)}
        for c, vals in per_campaign.items()
    ]
    ranking.sort(key=lambda r: (r["min_gamma_star"] is None, r["min_gamma_star"] if r["min_gamma_star"] is not None else np.inf))
    return ranking


def run(
    config_path: str = CONFIG_PATH,
    last_touch_subsample_n: int = LAST_TOUCH_SUBSAMPLE_N,
    seed: int = SEED,
    cert_method: str | None = None,
    boundary: str | None = None,
) -> dict:
    """cert_method None is the production hybrid; boundary None is the production exact boundary (see e5.run)."""
    cert_method = PRODUCTION_CERT_METHOD if cert_method is None else cert_method
    if boundary is not None:
        with boundary_override(boundary):
            return run(config_path, last_touch_subsample_n, seed, cert_method, None)
    cfg = load_config(config_path)
    alpha = cfg["certification"]["alpha"]
    delta = cfg["certification"]["delta"]
    p_lo = cfg["certification"]["p_lo"]
    top_20 = cfg["campaigns"]["top_20"]
    K = len(top_20)
    rng = np.random.default_rng(seed)
    beta = alpha / (K + 1)

    journeys, _ = build_and_cache_primary_journeys(config_path)
    n_journeys_primary = len(journeys)

    result = {"K_primary": K, "alpha": alpha, "delta": delta, "p_lo": p_lo, "n_journeys_primary": n_journeys_primary,
              "cs_method": CS_METHOD, "beta": beta, "alpha_used": beta / 2.0, "gamma_star_tol": GAMMA_STAR_TOL,
              "boundary": MultiCampaignCumulativeShareCS(K, alpha, 1.0).boundary, "cert_method": cert_method,
              "floor_cap_1_over_p_lo": 1.0 / p_lo, "gamma_star_convention": "ratio-only (no p_lo floor); see n_above_floor_cap",
              "cs_running_intersection": False}

    guaranteed = {}
    for model in ("linear", "time_decay"):
        t0 = time.time()
        gstars, _box, bad = _gamma_star_distribution(journeys, model, K, delta, alpha, cert_method)
        vals = [g for g in gstars.values() if g is not None]
        guaranteed[model] = {
            "gamma_star_by_pair": gstars,
            "n_certified": len(vals),
            "n_pairs": len(gstars),
            "median": float(np.median(vals)) if vals else None,
            "q25": float(np.percentile(vals, 25)) if vals else None,
            "q75": float(np.percentile(vals, 75)) if vals else None,
            "max": float(np.max(vals)) if vals else None,
            "n_above_floor_cap": int(sum(1 for g in vals if g > 1.0 / p_lo)),
            "gamma_star_rule_consistency_violations": bad,
            "fragility_ranking": _fragility_ranking(gstars, K),
            "wall_time_s": time.time() - t0,
        }
    result["guaranteed"] = guaranteed

    common_pairs = [k for k in guaranteed["linear"]["gamma_star_by_pair"]
                     if guaranteed["linear"]["gamma_star_by_pair"][k] is not None
                     and guaranteed["time_decay"]["gamma_star_by_pair"][k] is not None]
    diffs = [abs(guaranteed["linear"]["gamma_star_by_pair"][k] - guaranteed["time_decay"]["gamma_star_by_pair"][k])
             for k in common_pairs]
    result["linear_vs_time_decay_coincide"] = {
        "n_common_certified_pairs": len(common_pairs),
        "max_abs_diff": float(max(diffs)) if diffs else None,
        "mean_abs_diff": float(np.mean(diffs)) if diffs else None,
    }

    t0 = time.time()
    idx = rng.choice(n_journeys_primary, size=min(last_touch_subsample_n, n_journeys_primary), replace=False)
    subsample = [journeys[i] for i in idx]
    raw_gap_lt_pairs = _raw_gap_pairs(journeys, "last_touch", K, delta)
    sens = {}
    for j, k in raw_gap_lt_pairs:
        shifts = gamma_hat_sens(subsample, lambda jy: _last_touch_credit_fn(jy, K), K, j, k, GAMMA_HAT_GRID, p_lo, rng)
        sens[f"{j}-{k}"] = shifts
    result["empirical_primary"] = {
        "model": "last_touch",
        "subsample_n": len(subsample),
        "gamma_hat_sens_by_pair": sens,
        "n_pairs_evaluated": len(sens),
        "wall_time_s": time.time() - t0,
    }

    t0 = time.time()
    sec_journeys, top_10 = build_and_cache_secondary_journeys(config_path)
    K_sec = len(top_10)
    n_multi = sum(1 for j in sec_journeys if len(set(t.campaign for t in j.touches)) > 1)
    raw_gap_sec_pairs = _raw_gap_pairs(sec_journeys, "last_touch", K_sec, delta)
    sec_sub_n = min(last_touch_subsample_n, len(sec_journeys))
    sec_idx = rng.choice(len(sec_journeys), size=sec_sub_n, replace=False)
    sec_subsample = [sec_journeys[i] for i in sec_idx]
    sens_sec = {}
    for j, k in raw_gap_sec_pairs:
        shifts = gamma_hat_sens(sec_subsample, lambda jy: _last_touch_credit_fn(jy, K_sec), K_sec, j, k, GAMMA_HAT_GRID, p_lo, rng)
        sens_sec[f"{j}-{k}"] = shifts
    result["empirical_secondary"] = {
        "model": "last_touch",
        "construction": "secondary (literature-convention linkage)",
        "n_journeys_secondary": len(sec_journeys),
        "n_multi_campaign": n_multi,
        "multi_campaign_fraction": n_multi / len(sec_journeys),
        "note": f"{n_multi}/{len(sec_journeys)} ({n_multi/len(sec_journeys):.1%}) journeys span more than one "
                f"campaign; the envelope below is informative on that minority only -- the other "
                f"{len(sec_journeys)-n_multi} journeys are single-campaign like all of primary.",
        "subsample_n": len(sec_subsample),
        "gamma_hat_sens_by_pair": sens_sec,
        "n_pairs_evaluated": len(sens_sec),
        "wall_time_s": time.time() - t0,
    }

    return result


def write_outputs(result: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e6_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    csv_lines = ["campaign_idx,model,min_gamma_star,n_certified_pairs"]
    for model in ("linear", "time_decay"):
        for row in result["guaranteed"][model]["fragility_ranking"]:
            csv_lines.append(f"{row['campaign_idx']},{model},{row['min_gamma_star']},{row['n_certified_pairs']}")
    with open(f"{out_dir}/table4_fragility.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")


if __name__ == "__main__":
    result = run()
    write_outputs(result, "results/e6")
    for model in ("linear", "time_decay"):
        g = result["guaranteed"][model]
        print(f"{model}: median Gamma*={g['median']} (n={g['n_certified']}/{g['n_pairs']})")
    print("linear vs time_decay coincide:", result["linear_vs_time_decay_coincide"])
