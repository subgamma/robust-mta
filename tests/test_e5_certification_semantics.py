"""Regression tests for the E5 certification bookkeeping. Two groups:

1. Pure, always-run unit tests of the shared per-checkpoint state helpers
   (`_states_from_gaps`, `_pair_stats`) against small synthetic gap/state
   sequences -- no real data needed. These confirm the direction-flip
   detector operates ONLY on the fresh per-checkpoint
   state sequence passed in, never on any ever-certified/cumulative
   bookkeeping (the states are constructed by hand here, bypassing
   `_states_from_gaps` entirely, so there is no ever-certified matrix in
   scope for the detector to have used even by accident).
2. The exact, independently re-verified against the
   production CS + `certify.robust_rule` directly (not against e5's own
   bookkeeping), plus broader invariants over a fast subsample run.
"""
from __future__ import annotations

import os
from itertools import combinations

import numpy as np
import pytest

from anytime_attribution.certify import Certification, robust_rule
from anytime_attribution.criteo import load_config
from anytime_attribution.experiments._primary_journeys import CACHE_PATH
from anytime_attribution.experiments import e5_certified_fraction as e5

CONFIG_PATH = "config/criteo.yaml"


def test_states_from_gaps_basic():
    gap_jk = np.array([0.001, 0.01, -0.02])
    gap_kj = np.array([-0.001, -0.01, 0.03])
    delta = 0.005
    states = e5._states_from_gaps(gap_jk, gap_kj, delta)
    assert states == [Certification.UNRESOLVED, Certification.J_OVER_K, Certification.K_OVER_J]


def test_states_from_gaps_applies_empirical_margin():
    gap_jk = np.array([0.01])
    gap_kj = np.array([0.0])
    delta = 0.005
    # margin=0 -> certifies (0.01 > 0.005); margin=0.01 -> no longer certifies
    assert e5._states_from_gaps(gap_jk, gap_kj, delta, margin_jk=0.0) == [Certification.J_OVER_K]
    assert e5._states_from_gaps(gap_jk, gap_kj, delta, margin_jk=0.01) == [Certification.UNRESOLVED]


def test_pair_stats_first_certification_ever_current_withdrawn():
    states = [Certification.UNRESOLVED, Certification.J_OVER_K, Certification.J_OVER_K, Certification.UNRESOLVED]
    checkpoints = np.array([100, 200, 300, 400])
    gap_jk = np.array([0.0, 0.1, 0.1, 0.0])
    gap_kj = np.array([0.0, 0.0, 0.0, 0.0])
    stats = e5._pair_stats(states, checkpoints, gap_jk, gap_kj)
    assert stats["t_first_certified"] == 200
    assert stats["ever_certified"] is True
    assert stats["current_certified"] is False
    assert stats["withdrawn"] is True
    assert stats["flips"] == []


def test_pair_stats_detects_direction_flip_from_fresh_states_only():
    """The flip detector must fire from the STATES SEQUENCE ALONE (fresh
    per-checkpoint evaluations), not from any ever-certified/cumulative
    structure. These states are constructed by hand -- no ever-certified
    matrix exists anywhere in this test's scope -- so a correct flip result
    here proves the detector doesn't depend on that bookkeeping."""
    states = [
        Certification.UNRESOLVED,   # checkpoint 100
        Certification.J_OVER_K,     # checkpoint 200 -- first certification
        Certification.K_OVER_J,     # checkpoint 300 -- DIRECTION FLIP (neither endpoint unresolved)
        Certification.UNRESOLVED,   # checkpoint 400 -- withdrawn
    ]
    checkpoints = np.array([100, 200, 300, 400])
    gap_jk = np.array([0.0, 0.10, -0.10, 0.0])
    gap_kj = np.array([0.0, -0.10, 0.10, 0.0])
    stats = e5._pair_stats(states, checkpoints, gap_jk, gap_kj)

    assert stats["t_first_certified"] == 200
    assert stats["ever_certified"] is True
    assert stats["current_certified"] is False
    assert stats["withdrawn"] is True
    assert len(stats["flips"]) == 1
    flip = stats["flips"][0]
    assert flip["checkpoint_before"] == 200
    assert flip["checkpoint_after"] == 300
    assert flip["direction_before"] == Certification.J_OVER_K.value
    assert flip["direction_after"] == Certification.K_OVER_J.value
    assert flip["gap_before_jk"] == pytest.approx(0.10)
    assert flip["gap_after_kj"] == pytest.approx(0.10)


def test_pair_stats_no_flip_when_endpoint_is_unresolved():
    """A transition through UNRESOLVED (withdraw, then re-certify in the
    opposite direction later) is NOT a checkpoint-to-checkpoint flip -- the
    spec requires neither endpoint to be UNRESOLVED."""
    states = [Certification.J_OVER_K, Certification.UNRESOLVED, Certification.K_OVER_J]
    checkpoints = np.array([100, 200, 300])
    gap_jk = np.zeros(3)
    gap_kj = np.zeros(3)
    stats = e5._pair_stats(states, checkpoints, gap_jk, gap_kj)
    assert stats["flips"] == []


pytestmark_real_data = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH) and not os.path.exists("data/criteo_attribution_dataset.tsv.gz"),
    reason="primary journey cache (built from the raw Criteo file)/raw data not available",
)

PAIR = (3, 4)
GAMMA = 1.0
# (label, cert_method, boundary, expected first-certification time). The legacy ratio box with the simplified boundary
# (config A) certifies at 42000; the production hybrid with the exact boundary (config D) certifies earlier (28000)
# but is still withdrawn at day 30.
ENGINES = [("A_legacy", "legacy_ratio_box", "simplified", 42000), ("D_final", "hybrid", "exact", 28000)]


@pytestmark_real_data
@pytest.mark.parametrize("label,cert_method,boundary,expected_t", ENGINES)
def test_pair_3_4_counterexample_independently_verified_and_matches_run_output(label, cert_method, boundary, expected_t):
    """The exact: linear, Gamma=1.0, delta=0.005, seed 0, full primary
    stream (n_total=109221), pair (3,4). Ground truth is computed HERE via a fresh, independent call to the
    LOW-LEVEL rule (`robust_rule` for legacy, `hybrid_robust_rule` on the stored boxes for hybrid) on the CS boxes
    built from `_model_masses`/`_cs_history` directly, not by trusting `e5.run`'s bookkeeping; `run` is then
    checked against it."""
    from anytime_attribution.certify import hybrid_robust_rule
    from anytime_attribution.cs import boundary_override
    cfg = load_config(CONFIG_PATH)
    top_20 = cfg["campaigns"]["top_20"]
    K = len(top_20)
    alpha = cfg["certification"]["alpha"]
    delta = cfg["certification"]["delta"]
    assert delta == 0.005

    journeys, top_20_check = e5.build_and_cache_primary_journeys(CONFIG_PATH)
    assert top_20_check == top_20
    n_total = len(journeys)
    checkpoints = np.arange(500, n_total + 1, 500)
    if checkpoints[-1] != n_total:
        checkpoints = np.append(checkpoints, n_total)

    Tj_all, B = e5._model_masses(journeys, "linear", K)
    T_all = Tj_all.sum(axis=1)
    with boundary_override(boundary):
        boxes = e5._cs_history(Tj_all, T_all, alpha, B, checkpoints, return_boxes=True)

    j, k = PAIR

    def low_level(b):
        if cert_method == "legacy_ratio_box":
            return robust_rule(b.L, b.U, GAMMA, delta, j, k)
        return hybrid_robust_rule(b.LN, b.UN, b.L, b.U, GAMMA, delta, j, k)

    assert int(checkpoints[-1]) == n_total
    assert low_level(boxes[-1]) == Certification.UNRESOLVED, (
        "counterexample premise failed: pair (3,4) should be UNRESOLVED at day 30 under linear, Gamma=1.0"
    )
    t_first_independent = None
    for ci in range(len(checkpoints)):
        if low_level(boxes[ci]) != Certification.UNRESOLVED:
            t_first_independent = int(checkpoints[ci])
            break
    assert t_first_independent == expected_t

    result = e5.run(models=("linear",), gamma_grid=(GAMMA,), cert_method=cert_method, boundary=boundary)
    g = result["models"]["linear"]["gammas"][GAMMA]
    key = f"{j}-{k}"

    assert g["t_first_certified"][key] == expected_t
    assert key in [p for p in _all_pair_keys(K) if p == key]  # sanity: key format matches
    assert key not in _current_certified_keys(g, K)
    assert key in g["withdrawn_pairs"]


def _all_pair_keys(K: int) -> list[str]:
    return [f"{j}-{k}" for j, k in combinations(range(K), 2)]


def _current_certified_keys(g: dict, K: int) -> set[str]:
    all_keys = set(_all_pair_keys(K))
    uncertified = set(g["sampling_limited"]) | set(g["observability_limited"])
    return all_keys - uncertified


def _small_run(n_journeys=6000, checkpoint_step=1000):
    journeys, top_20 = e5.build_and_cache_primary_journeys()
    import anytime_attribution.experiments._primary_journeys as pj
    orig = pj.build_and_cache_primary_journeys
    pj.build_and_cache_primary_journeys = lambda *a, **k: (journeys[:n_journeys], top_20)
    e5.build_and_cache_primary_journeys = pj.build_and_cache_primary_journeys
    try:
        return e5.run(checkpoint_step=checkpoint_step, models=("linear", "time_decay"))
    finally:
        pj.build_and_cache_primary_journeys = orig
        e5.build_and_cache_primary_journeys = orig


@pytestmark_real_data
def test_day30_current_certified_matches_independent_direct_evaluation():
    """For every guaranteed (model, Gamma) cell, the day-30
    n_current_certified count must exactly match a fresh, independent
    evaluation of `robust_rule` on the final CS box -- computed here without
    calling any of e5's own per-pair bookkeeping helpers."""
    result = _small_run()
    cfg = load_config(CONFIG_PATH)
    delta = cfg["certification"]["delta"]
    top_20 = cfg["campaigns"]["top_20"]
    K = len(top_20)
    pairs = list(combinations(range(K), 2))

    journeys, _ = e5.build_and_cache_primary_journeys(CONFIG_PATH)
    journeys = journeys[:6000]
    checkpoints = np.arange(1000, len(journeys) + 1, 1000)
    if checkpoints[-1] != len(journeys):
        checkpoints = np.append(checkpoints, len(journeys))

    for model in ("linear", "time_decay"):
        Tj_all, B = e5._model_masses(journeys, model, K)
        T_all = Tj_all.sum(axis=1)
        alpha = cfg["certification"]["alpha"]
        from anytime_attribution.certify import hybrid_robust_rule
        fb = e5._cs_history(Tj_all, T_all, alpha, B, checkpoints, return_boxes=True)[-1]

        for Gamma, g in result["models"][model]["gammas"].items():
            n_direct = sum(
                1 for j, k in pairs if hybrid_robust_rule(fb.LN, fb.UN, fb.L, fb.U, Gamma, delta, j, k) != Certification.UNRESOLVED
            )
            assert g["n_current_certified"] == n_direct, f"{model} Gamma={Gamma}: bookkeeping vs direct mismatch"


@pytestmark_real_data
def test_all_pairs_accounted_for_current_certified_or_unresolved_split():
    result = _small_run()
    for model in result["models"]:
        for Gamma, g in result["models"][model]["gammas"].items():
            total = g["n_current_certified"] + len(g["sampling_limited"]) + len(g["observability_limited"])
            assert total == g["n_pairs"] == 190
