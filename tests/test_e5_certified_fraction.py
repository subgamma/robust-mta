import os

import pytest

from anytime_attribution.experiments._primary_journeys import CACHE_PATH
from anytime_attribution.experiments import e5_certified_fraction as e5

pytestmark = pytest.mark.skipif(not os.path.exists(CACHE_PATH) and not os.path.exists("data/criteo_attribution_dataset.tsv.gz"),
                                 reason="primary journey cache (built from the raw Criteo file)/raw data not available")


def _small_run(n_journeys=4000, checkpoint_step=1000, models=("linear", "last_touch")):
    journeys, top_20 = e5.build_and_cache_primary_journeys()
    import anytime_attribution.experiments._primary_journeys as pj
    orig = pj.build_and_cache_primary_journeys
    pj.build_and_cache_primary_journeys = lambda *a, **k: (journeys[:n_journeys], top_20)
    e5.build_and_cache_primary_journeys = pj.build_and_cache_primary_journeys
    try:
        return e5.run(checkpoint_step=checkpoint_step, models=models)
    finally:
        pj.build_and_cache_primary_journeys = orig
        e5.build_and_cache_primary_journeys = orig


def test_e5_current_fraction_bounded_ever_fraction_monotone():
    """current_certified_fraction is a fresh per-checkpoint
    evaluation and is allowed to be non-monotone (withdrawals under drift);
    only ever_certified_fraction (a cumulative-max by construction) must be
    non-decreasing."""
    result = _small_run()
    for model in result["models"]:
        for Gamma, g in result["models"][model]["gammas"].items():
            current = list(g["current_certified_fraction"].values())
            ever = list(g["ever_certified_fraction"].values())
            assert all(0.0 <= f <= 1.0 for f in current)
            assert all(0.0 <= f <= 1.0 for f in ever)
            assert all(b >= a - 1e-9 for a, b in zip(ever, ever[1:]))
            # ever >= current pointwise, always
            assert all(e >= c - 1e-9 for c, e in zip(current, ever))


def test_e5_unresolved_split_covers_all_uncertified_pairs():
    result = _small_run()
    for model in result["models"]:
        for Gamma, g in result["models"][model]["gammas"].items():
            n_uncertified = g["n_pairs"] - g["n_current_certified"]
            assert len(g["sampling_limited"]) + len(g["observability_limited"]) == n_uncertified


def test_e5_withdrawn_equals_ever_minus_current():
    result = _small_run()
    for model in result["models"]:
        for Gamma, g in result["models"][model]["gammas"].items():
            assert g["n_withdrawn"] == g["n_ever_certified"] - g["n_current_certified"]
            assert g["n_ever_certified"] >= g["n_current_certified"]


def test_e5_gamma1_stricter_than_gamma1p5():
    """Larger Gamma can only make certification harder (weakly) at every
    fixed checkpoint, including day 30: a larger box-widening factor only
    shrinks the robust lower gap, so both current- and ever-certified counts
    at Gamma=1.5 are <= Gamma=1's."""
    result = _small_run(n_journeys=8000, checkpoint_step=2000)
    for model in result["models"]:
        g1 = result["models"][model]["gammas"][1.0]
        g15 = result["models"][model]["gammas"][1.5]
        assert g15["n_current_certified"] <= g1["n_current_certified"]
        assert g15["n_ever_certified"] <= g1["n_ever_certified"]
