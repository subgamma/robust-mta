"""E5: conversions required to certify on Criteo (primary construction, K=20).

Guaranteed vs empirical: linear and time-decay are theorem-covered and certified by
`robust_rule`. Last-touch and logistic+SHAP carry no identification guarantee and are
certified with the empirical `certify.gamma_hat_sens` margin (credit-difference shift
under re-thinning a held-out subsample) in place of the box widening; their curves are
labelled empirical in every output (see `_empirical_certified_fraction`).

Per-model mass T_ij, on each model's own weight scale (the normalised per-journey
credit is one-hot for all four models and would hide the mass-scale differences):
  - linear: n_touches(journey) if the journey's campaign is j, else 0
  - time_decay: sum(2**(-tau_l/7)) over the journey's touches if j, else 0
  - last_touch: 1 if j, else 0
  - logistic_shap: 1 if j, else 0 (SHAP credit has no raw mass of its own)
last_touch and logistic_shap therefore have identical aggregate-share mass (journey
count share); linear and time_decay differ because campaigns' converting journeys
have different mean touch counts and timings.

CS box: one `cs.MultiCampaignCumulativeShareCS` per model with beta=alpha/(K+1)
(K=20 numerators plus the shared denominator). The CS does not depend on Gamma, so one
pass per model gives the [L,U] history from which all Gamma values are evaluated.
Checkpoints every 500 conversions.

Certification is evaluated fresh at every checkpoint against that checkpoint's own
CS or envelope, with no carry-forward, and withdrawals are allowed (cumulative-to-date
target). Reported per (model, Gamma):
  - `current_certified_fraction[t]`: fraction of pairs certified at checkpoint t. The
    primary curve; it may be non-monotone.
  - `ever_certified_fraction[t]`: fraction certified at least once by t. Monotone;
    descriptive only.
  - `t_first_certified` (per pair): first checkpoint at which the pair certifies;
    `median_t_first_certified` is over pairs that ever certify (`n_ever_certified`
    is the denominator).
At the final checkpoint it also reports `n_current_certified`, `n_ever_certified` and
`n_withdrawn = n_ever_certified - n_current_certified`. Sampling- vs
observability-limited splits use the final current state: sampling-limited means the
ordering is identified at this Gamma (observed a_j/a_k or a_k/a_j > Gamma for
guaranteed models; a nonzero raw gap for empirical models) but the CS has not
tightened enough; observability-limited means it is unidentified at this Gamma.

Direction flips (CERT(j>k) directly to CERT(k>j), or the reverse, between consecutive
checkpoints) are tracked separately and reported with pair, checkpoint and gap detail
whenever nonzero. They do not contradict the guarantee, since each certification is
valid for its own time-t target, but they are stronger instability than a withdrawal.
"""
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl

from ..certify import (BoxSet, Certification, PRODUCTION_CERT_METHOD, box_lower_gap, direct_lower_gap,
                       gamma_hat_sens, snapshot)
from ..criteo import build_journeys, build_negative_examples, converting_touches, load_config, load_raw
from ..cs import BOUNDARY_DEFAULT, MultiCampaignCumulativeShareCS, boundary_override
from ..estimands import campaign_masses, per_journey_shares
from ..models.logistic_shap import LogisticSHAP
from ._primary_journeys import build_and_cache_primary_journeys, last_touch_journey, time_decay_journey

CONFIG_PATH = "config/criteo.yaml"
GAMMA_GRID = (1.0, 1.25, 1.5)
GAMMA_HAT_GRID = (1.25, 1.5)
CHECKPOINT_STEP = 500
CS_METHOD = "howard_eb_ratio"
EMPIRICAL_HELD_OUT_N = 5000
NEGATIVE_N_SAMPLE = 50000

MODELS = ("linear", "time_decay", "last_touch", "logistic_shap")
MODEL_LABEL = {
    "linear": "linear (guaranteed)",
    "time_decay": "time-decay (guaranteed)",
    "last_touch": "last-touch (empirical)",
    "logistic_shap": "logistic+SHAP (empirical)",
}
GUARANTEED_MODELS = {"linear", "time_decay"}
EMPIRICAL_MODELS = {"last_touch", "logistic_shap"}


def _fit_logistic_shap_primary(config_path: str, seed: int = 0) -> LogisticSHAP:
    """Refit LogisticSHAP on primary warm-up (days 1-7) data; no fitted model was cached. SHAP for a linear model is
    closed-form.
    """
    cfg = load_config(config_path)
    top_20 = cfg["campaigns"]["top_20"]
    campaign_index = {c: i for i, c in enumerate(top_20)}
    lookback_days = cfg["cap"]["lookback_days"]
    max_touches = cfg["cap"]["max_touches"]

    lf = load_raw(cfg["dataset"]["path"])
    warmup_touches = converting_touches(lf).filter(pl.col("conv_day").is_between(1, 7))
    warmup_journeys = build_journeys(warmup_touches, campaign_index, lookback_days=lookback_days, max_touches=max_touches)
    negatives = build_negative_examples(lf, campaign_index, day_lo=1, day_hi=7, max_touches=max_touches,
                                          n_sample=NEGATIVE_N_SAMPLE, seed=seed)
    model = LogisticSHAP()
    model.fit(warmup_journeys, len(top_20), seed=seed, negative_journeys=negatives)
    return model


def _model_masses(journeys, model: str, K: int, fitted_logistic_shap: LogisticSHAP | None = None) -> tuple[np.ndarray, float]:
    """(Tj_all: (n,K) array, B_bound) for the given model."""
    if model == "linear":
        Tj = np.array([campaign_masses(j, K) for j in journeys])
        return Tj, 10.0
    if model == "time_decay":
        Tj = np.array([campaign_masses(time_decay_journey(j), K) for j in journeys])
        return Tj, 10.0
    if model == "last_touch":
        Tj = np.array([campaign_masses(last_touch_journey(j), K) for j in journeys])
        return Tj, 1.0
    if model == "logistic_shap":
        # Mass is the fitted model's credit output on the last-touch scale (B=1.0).
        if fitted_logistic_shap is None:
            raise ValueError("logistic_shap requires a fitted model")
        Tj = np.array([fitted_logistic_shap.credit(j) for j in journeys])
        return Tj, 1.0
    raise ValueError(f"unknown model {model}")


def _empirical_credit_fn(model: str, K: int, fitted_logistic_shap: LogisticSHAP | None):
    if model == "last_touch":
        return lambda j: per_journey_shares(last_touch_journey(j), K)
    if model == "logistic_shap":
        return fitted_logistic_shap.credit
    raise ValueError(f"not an empirical model: {model}")


def _cs_history(Tj_all: np.ndarray, T_all: np.ndarray, alpha: float, B: float,
                 checkpoints: np.ndarray, return_boxes: bool = False):
    """Stream Tj_all/T_all through `MultiCampaignCumulativeShareCS`, recording the boxes at each checkpoint.

    Boundary form is `cs.BOUNDARY_DEFAULT` unless inside `cs.boundary_override`. `alpha`
    is the raw alpha; the class derives beta=alpha/(K+1) itself. Returns (L_hist, U_hist)
    of shape (n_checkpoints, K), or with return_boxes=True a list of `certify.BoxSet`
    so the hybrid rule can be evaluated.
    """
    n, K = Tj_all.shape
    mg = MultiCampaignCumulativeShareCS(K, alpha, B)
    checkpoint_set = set(checkpoints.tolist())
    boxes = []
    for i in range(n):
        mg.update(Tj_all[i], T_all[i])
        if (i + 1) in checkpoint_set:
            boxes.append(snapshot(mg))
    if return_boxes:
        return boxes
    return np.array([b.L for b in boxes]), np.array([b.U for b in boxes])


def _empirical_thresholds(
    held_out, credit_fn, K: int, pairs, gamma_hat_grid: tuple[float, ...], p_lo: float, rng: np.random.Generator,
) -> dict:
    """Per-pair, per-Gamma, per-direction empirical margin from `certify.gamma_hat_sens` on a held-out subsample.

    Returns {(j,k): {Gamma: shift_jk}}, where shift_jk is the shift from perturbing j
    (to test j>k); query (k,j) for the reverse direction.
    """
    thresholds = {}
    for j, k in pairs:
        thresholds[(j, k)] = gamma_hat_sens(held_out, credit_fn, K, j, k, gamma_hat_grid, p_lo, rng)
        thresholds[(k, j)] = gamma_hat_sens(held_out, credit_fn, K, k, j, gamma_hat_grid, p_lo, rng)
    return thresholds


def _guaranteed_gaps_for_pair(boxes: list, Gamma: float, j: int, k: int, method: str | None = None,
                              delta: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-checkpoint Gamma-widened robust lower gap in both directions under the chosen certification method."""
    gap_jk = np.array([box_lower_gap(b, Gamma, j, k, method=method, delta=delta) for b in boxes])
    gap_kj = np.array([box_lower_gap(b, Gamma, k, j, method=method, delta=delta) for b in boxes])
    return gap_jk, gap_kj


def _empirical_gaps_for_pair(L_hist: np.ndarray, U_hist: np.ndarray, j: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-checkpoint raw CS-box gap in both directions. Gamma enters only through the empirical margin in
    `_states_from_gaps`.
    """
    return L_hist[:, j] - U_hist[:, k], L_hist[:, k] - U_hist[:, j]


def _states_from_gaps(
    gap_jk: np.ndarray, gap_kj: np.ndarray, delta: float, margin_jk: float = 0.0, margin_kj: float = 0.0,
) -> list[Certification]:
    """Fresh per-checkpoint certification state from a pair's gap arrays, with no carry-forward.

    Shared by the guaranteed path (margin=0, Gamma already in the gap) and the
    empirical path (raw gap plus the `gamma_hat_sens` margin).
    """
    states = []
    for gjk, gkj in zip(gap_jk, gap_kj):
        if gjk > delta + margin_jk:
            states.append(Certification.J_OVER_K)
        elif gkj > delta + margin_kj:
            states.append(Certification.K_OVER_J)
        else:
            states.append(Certification.UNRESOLVED)
    return states


def _pair_stats(states: list[Certification], checkpoints: np.ndarray, gap_jk: np.ndarray, gap_kj: np.ndarray) -> dict:
    """Per-pair summary from the fresh state sequence: first certification time, final status, withdrawal and direction
    flips.
    """
    t_first_certified = None
    for ci, st in enumerate(states):
        if st != Certification.UNRESOLVED:
            t_first_certified = int(checkpoints[ci])
            break
    ever_certified = t_first_certified is not None
    current_certified = states[-1] != Certification.UNRESOLVED
    withdrawn = ever_certified and not current_certified
    current_indicator = np.array([s != Certification.UNRESOLVED for s in states])

    flips = []
    for ci in range(len(states) - 1):
        a, b = states[ci], states[ci + 1]
        if a != Certification.UNRESOLVED and b != Certification.UNRESOLVED and a != b:
            flips.append({
                "checkpoint_before": int(checkpoints[ci]), "checkpoint_after": int(checkpoints[ci + 1]),
                "direction_before": a.value, "direction_after": b.value,
                "gap_before_jk": float(gap_jk[ci]), "gap_before_kj": float(gap_kj[ci]),
                "gap_after_jk": float(gap_jk[ci + 1]), "gap_after_kj": float(gap_kj[ci + 1]),
            })

    return {
        "current_indicator": current_indicator, "t_first_certified": t_first_certified,
        "ever_certified": ever_certified, "current_certified": current_certified,
        "withdrawn": withdrawn, "flips": flips,
    }


def _finalize_gammas_out(
    current_matrix: np.ndarray, checkpoints: np.ndarray, t_first: dict, ever_certified_pairs: list,
    current_certified_pairs: list, withdrawn_pairs: list, sampling_limited: list, observability_limited: list,
    all_flips: list, n_pairs_with_flip: int, n_pairs: int,
) -> dict:
    ever_matrix = np.maximum.accumulate(current_matrix, axis=1)
    current_frac = current_matrix.mean(axis=0)
    ever_frac = ever_matrix.mean(axis=0)
    resolved_ts = [v for v in t_first.values() if v is not None]
    return {
        "current_certified_fraction": {int(c): float(f) for c, f in zip(checkpoints, current_frac)},
        "ever_certified_fraction": {int(c): float(f) for c, f in zip(checkpoints, ever_frac)},
        "t_first_certified": t_first,
        "median_t_first_certified": float(np.median(resolved_ts)) if resolved_ts else None,
        "n_current_certified": len(current_certified_pairs),
        "n_ever_certified": len(ever_certified_pairs),
        "n_withdrawn": len(withdrawn_pairs),
        "withdrawn_pairs": withdrawn_pairs,
        "n_pairs": n_pairs,
        "current_unresolved_fraction_day30": 1.0 - len(current_certified_pairs) / n_pairs,
        "sampling_limited": sampling_limited,
        "observability_limited": observability_limited,
        "direction_flips": {
            "n_pairs_with_flip": n_pairs_with_flip,
            "n_flip_events": len(all_flips),
            "events": all_flips,
        },
    }


def _guaranteed_certified_fraction(
    boxes: list, checkpoints: np.ndarray, delta: float,
    gamma_grid: tuple[float, ...], pairs, a_hat: np.ndarray, method: str | None = None,
) -> dict:
    """Fresh-per-checkpoint certification for linear/time_decay (guaranteed).

    Sampling- vs observability-limited uses the final unresolved set and the observed ratio a_hat.
    """
    gammas_out = {}
    n_pairs = len(pairs)
    for Gamma in gamma_grid:
        current_matrix = np.zeros((n_pairs, len(checkpoints)), dtype=bool)
        t_first, withdrawn_pairs, current_certified_pairs, ever_certified_pairs = {}, [], [], []
        sampling_limited, observability_limited = [], []
        all_flips, n_pairs_with_flip = [], 0
        for pi, (j, k) in enumerate(pairs):
            gap_jk, gap_kj = _guaranteed_gaps_for_pair(boxes, Gamma, j, k, method, delta)
            states = _states_from_gaps(gap_jk, gap_kj, delta)
            stats = _pair_stats(states, checkpoints, gap_jk, gap_kj)
            current_matrix[pi] = stats["current_indicator"]
            key = f"{j}-{k}"
            t_first[key] = stats["t_first_certified"]
            if stats["ever_certified"]:
                ever_certified_pairs.append(key)
            if stats["current_certified"]:
                current_certified_pairs.append(key)
            else:
                # Plug-in split: would the observed shares a_hat, as point boxes, certify under the Gamma-widened
                # direct rule at this delta? If so the remaining gap is sampling-limited, otherwise observability-
                # limited.
                pt = np.asarray(a_hat, float)
                plug = max(direct_lower_gap(pt, pt, Gamma, j, k, delta=delta),
                           direct_lower_gap(pt, pt, Gamma, k, j, delta=delta))
                (sampling_limited if plug > delta else observability_limited).append(key)
            if stats["withdrawn"]:
                withdrawn_pairs.append(key)
            if stats["flips"]:
                n_pairs_with_flip += 1
                for f in stats["flips"]:
                    all_flips.append({"pair": key, **f})
        gammas_out[Gamma] = _finalize_gammas_out(
            current_matrix, checkpoints, t_first, ever_certified_pairs, current_certified_pairs,
            withdrawn_pairs, sampling_limited, observability_limited, all_flips, n_pairs_with_flip, n_pairs,
        )
    return gammas_out


def _empirical_certified_fraction(
    L_hist: np.ndarray, U_hist: np.ndarray, checkpoints: np.ndarray, thresholds: dict,
    delta: float, gamma_grid: tuple[float, ...], pairs,
) -> dict:
    """Fresh-per-checkpoint empirical certification.

    CERT(j>k) at checkpoint t iff L_t^j - U_t^k > delta + shift_jk(Gamma), evaluated
    independently at every checkpoint. At Gamma=1 the margin is exactly 0 (delta alone),
    as in the guaranteed rule.
    """
    gammas_out = {}
    n_pairs = len(pairs)
    for Gamma in gamma_grid:
        current_matrix = np.zeros((n_pairs, len(checkpoints)), dtype=bool)
        t_first, withdrawn_pairs, current_certified_pairs, ever_certified_pairs = {}, [], [], []
        sampling_limited, observability_limited = [], []
        all_flips, n_pairs_with_flip = [], 0
        for pi, (j, k) in enumerate(pairs):
            margin = 0.0 if Gamma == 1.0 else thresholds[(j, k)][Gamma]
            margin_rev = 0.0 if Gamma == 1.0 else thresholds[(k, j)][Gamma]
            gap_jk, gap_kj = _empirical_gaps_for_pair(L_hist, U_hist, j, k)
            states = _states_from_gaps(gap_jk, gap_kj, delta, margin, margin_rev)
            stats = _pair_stats(states, checkpoints, gap_jk, gap_kj)
            current_matrix[pi] = stats["current_indicator"]
            key = f"{j}-{k}"
            t_first[key] = stats["t_first_certified"]
            if stats["ever_certified"]:
                ever_certified_pairs.append(key)
            if stats["current_certified"]:
                current_certified_pairs.append(key)
            else:
                final_gap = max(gap_jk[-1], gap_kj[-1])
                (sampling_limited if final_gap > 0 else observability_limited).append(key)
            if stats["withdrawn"]:
                withdrawn_pairs.append(key)
            if stats["flips"]:
                n_pairs_with_flip += 1
                for f in stats["flips"]:
                    all_flips.append({"pair": key, **f})
        gammas_out[Gamma] = _finalize_gammas_out(
            current_matrix, checkpoints, t_first, ever_certified_pairs, current_certified_pairs,
            withdrawn_pairs, sampling_limited, observability_limited, all_flips, n_pairs_with_flip, n_pairs,
        )
    return gammas_out


def run(
    config_path: str = CONFIG_PATH,
    gamma_grid: tuple[float, ...] = GAMMA_GRID,
    gamma_hat_grid: tuple[float, ...] = GAMMA_HAT_GRID,
    checkpoint_step: int = CHECKPOINT_STEP,
    models: tuple[str, ...] = MODELS,
    empirical_held_out_n: int = EMPIRICAL_HELD_OUT_N,
    seed: int = 0,
    cert_method: str | None = None,
    boundary: str | None = None,
) -> dict:
    """cert_method: None is the production hybrid; "legacy_ratio_box" applies to guaranteed models only (the empirical
    path is the raw box gap plus the `gamma_hat_sens` margin).

    boundary: None is the exact boundary; "simplified" is the comparator.
    """
    cert_method = PRODUCTION_CERT_METHOD if cert_method is None else cert_method
    if boundary is not None:
        with boundary_override(boundary):
            return run(config_path, gamma_grid, gamma_hat_grid, checkpoint_step, models,
                       empirical_held_out_n, seed, cert_method, None)
    cfg = load_config(config_path)
    alpha = cfg["certification"]["alpha"]
    delta = cfg["certification"]["delta"]
    p_lo = cfg["certification"]["p_lo"]
    top_20 = cfg["campaigns"]["top_20"]
    K = len(top_20)
    pairs = list(combinations(range(K), 2))
    rng = np.random.default_rng(seed)
    beta = alpha / (K + 1)

    journeys, top_20_check = build_and_cache_primary_journeys(config_path)
    assert top_20_check == top_20
    n_total = len(journeys)
    checkpoints = np.arange(checkpoint_step, n_total + 1, checkpoint_step)
    if checkpoints[-1] != n_total:
        checkpoints = np.append(checkpoints, n_total)

    result = {"models": {}, "n_total": n_total, "checkpoints": checkpoints.tolist(),
              "alpha": alpha, "delta": delta, "gamma_grid": list(gamma_grid), "K": K,
              "cs_method": CS_METHOD, "beta": beta, "alpha_used": beta / 2.0,
              "boundary": MultiCampaignCumulativeShareCS(K, alpha, 1.0).boundary, "cert_method_guaranteed": cert_method,
              "cert_method_empirical": "N/A (raw box gap + gamma_hat_sens margin; empirical, no identification theorem)",
              "cs_boundary": {"theorem": "Howard et al. (2021) Thm. 1 eq. (8) / Thm. 4 (exact stitched boundary)",
                              "eta": 2.0, "s": 1.4, "m": 1.0, "forecast": "lagged running mean, Xhat_1=0.5"},
              "cs_running_intersection": False}

    fitted_logistic_shap = _fit_logistic_shap_primary(config_path, seed) if "logistic_shap" in models else None

    for model in models:
        t0 = time.time()
        Tj_all, B = _model_masses(journeys, model, K, fitted_logistic_shap)
        T_all = Tj_all.sum(axis=1)
        boxes = _cs_history(Tj_all, T_all, alpha, B, checkpoints, return_boxes=True)
        L_hist = np.array([b.L for b in boxes]); U_hist = np.array([b.U for b in boxes])

        if model in GUARANTEED_MODELS:
            a_hat = Tj_all.sum(axis=0) / T_all.sum()
            gammas_out = _guaranteed_certified_fraction(boxes, checkpoints, delta, gamma_grid, pairs, a_hat, cert_method)
        else:
            # Fresh model-independent rng so every empirical model's held-out subsample and re-thinning draws are identical.
            held_out_rng = np.random.default_rng(seed + 1000)
            idx = held_out_rng.choice(n_total, size=min(empirical_held_out_n, n_total), replace=False)
            held_out = [journeys[i] for i in idx]
            credit_fn = _empirical_credit_fn(model, K, fitted_logistic_shap)
            thresholds = _empirical_thresholds(held_out, credit_fn, K, pairs, gamma_hat_grid, p_lo, held_out_rng)
            gammas_out = _empirical_certified_fraction(L_hist, U_hist, checkpoints, thresholds, delta, gamma_grid, pairs)

        result["models"][model] = {"gammas": gammas_out, "wall_time_s": time.time() - t0}

    return result


def write_outputs(result: dict, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{out_dir}/e5_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    csv_lines = ["model,guaranteed,Gamma,median_t_first_certified,n_current_certified,n_ever_certified,"
                 "n_withdrawn,n_pairs,current_unresolved_fraction_day30,n_sampling_limited,"
                 "n_observability_limited,n_pairs_with_flip,n_flip_events"]
    flip_lines = ["model,Gamma,pair,checkpoint_before,checkpoint_after,direction_before,direction_after,"
                  "gap_before_jk,gap_before_kj,gap_after_jk,gap_after_kj"]
    for model in MODELS:
        if model not in result["models"]:
            continue
        guaranteed = model in GUARANTEED_MODELS
        for Gamma, g in result["models"][model]["gammas"].items():
            df = g["direction_flips"]
            csv_lines.append(
                f"{model},{guaranteed},{Gamma},{g['median_t_first_certified']},{g['n_current_certified']},"
                f"{g['n_ever_certified']},{g['n_withdrawn']},{g['n_pairs']},"
                f"{g['current_unresolved_fraction_day30']:.4f},{len(g['sampling_limited'])},"
                f"{len(g['observability_limited'])},{df['n_pairs_with_flip']},{df['n_flip_events']}"
            )
            for ev in df["events"]:
                flip_lines.append(
                    f"{model},{Gamma},{ev['pair']},{ev['checkpoint_before']},{ev['checkpoint_after']},"
                    f"{ev['direction_before']},{ev['direction_after']},{ev['gap_before_jk']:.6f},"
                    f"{ev['gap_before_kj']:.6f},{ev['gap_after_jk']:.6f},{ev['gap_after_kj']:.6f}"
                )
    with open(f"{out_dir}/table3.csv", "w") as f:
        f.write("\n".join(csv_lines) + "\n")
    with open(f"{out_dir}/table3_direction_flips.csv", "w") as f:
        if len(flip_lines) == 1:
            f.write(flip_lines[0] + "\n# no direction flips detected in any (model,Gamma) cell\n")
        else:
            f.write("\n".join(flip_lines) + "\n")


if __name__ == "__main__":
    result = run()
    write_outputs(result, "results/e5")
    for model, mres in result["models"].items():
        for Gamma, g in mres["gammas"].items():
            print(f"{model} Gamma={Gamma}: median_t_first_certified={g['median_t_first_certified']} "
                  f"current_unresolved@30={g['current_unresolved_fraction_day30']:.3f} "
                  f"n_current={g['n_current_certified']} n_ever={g['n_ever_certified']} n_withdrawn={g['n_withdrawn']} "
                  f"(sampling-lim={len(g['sampling_limited'])}, obs-lim={len(g['observability_limited'])}, "
                  f"flips={g['direction_flips']['n_flip_events']})")
