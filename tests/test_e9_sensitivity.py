import os

import pytest

from anytime_attribution.experiments._primary_journeys import CACHE_PATH
from anytime_attribution.experiments import e9_sensitivity as e9

pytestmark = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH) and not os.path.exists("data/criteo_attribution_dataset.tsv.gz"),
    reason="primary journey cache (built from the raw Criteo file)/raw data not available",
)


def _small_run(n_journeys=4000):
    journeys, top_20 = e9.build_and_cache_primary_journeys()
    import anytime_attribution.experiments._primary_journeys as pj

    orig = pj.build_and_cache_primary_journeys
    pj.build_and_cache_primary_journeys = lambda *a, **k: (journeys[:n_journeys], top_20)
    e9.build_and_cache_primary_journeys = pj.build_and_cache_primary_journeys
    try:
        return e9.run()
    finally:
        pj.build_and_cache_primary_journeys = orig
        e9.build_and_cache_primary_journeys = orig


def test_e9_sensitivity_grid_monotone_in_gamma_and_delta():
    """Larger Gamma or larger delta can only make certification weakly
    harder (fewer or equal certified pairs), since both widen/raise the
    threshold the robust lower gap must clear."""
    result = _small_run()
    for model in e9.SENSITIVITY_MODELS:
        grid = result["sensitivity"][model]["grid"]
        for delta in e9.DELTA_GRID:
            certified = [grid[delta][g]["n_certified"] for g in e9.GAMMA_GRID]
            assert all(b <= a for a, b in zip(certified, certified[1:]))
        for Gamma in e9.GAMMA_GRID:
            certified = [grid[d][Gamma]["n_certified"] for d in e9.DELTA_GRID]
            assert all(b <= a for a, b in zip(certified, certified[1:]))


def test_e9_sensitivity_gamma1_is_naive_definition():
    result = _small_run()
    for model in e9.SENSITIVITY_MODELS:
        grid = result["sensitivity"][model]["grid"]
        for delta in e9.DELTA_GRID:
            assert grid[delta][1.0]["is_naive_definition"] is True
            assert grid[delta][1.25]["is_naive_definition"] is False


def test_e9_distortion_degenerate_mass_models_have_zero_gap_and_zero_sd():
    """last_touch and logistic_shap use a one-unit-per-journey mass scale (S=1 for every journey on this
        construction), so Q_j = A_j exactly (Q-A = -Cov(S,Z_j)/E[S], with Cov=0 for constant S) and SD(S)/E[S]=0.
    """
    result = _small_run()
    for model in ("last_touch", "logistic_shap"):
        row = result["distortion"][model]
        assert row["sd_S_over_mean_S"] == pytest.approx(0.0, abs=1e-9)
        for pc in row["per_campaign"]:
            assert pc["abs_diff_q_minus_a"] == pytest.approx(0.0, abs=1e-9)


def test_e9_distortion_linear_time_decay_have_nonzero_dispersion():
    result = _small_run()
    for model in ("linear", "time_decay"):
        row = result["distortion"][model]
        assert row["sd_S_over_mean_S"] > 0.0


def test_e9_distortion_labels_are_descriptive_not_guaranteed_or_empirical():
    result = _small_run()
    for model in e9.DISTORTION_MODELS:
        assert result["distortion"][model]["guarantee_label"] == "descriptive"
