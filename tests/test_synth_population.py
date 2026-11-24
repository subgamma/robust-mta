import numpy as np
import pytest

from anytime_attribution.synth import (
    sample_journey_lengths,
    true_aggregate_share,
    generate_synthetic_journeys,
)
from anytime_attribution.estimands import aggregate_share


def test_sample_journey_lengths_quantiles_roughly_match_criteo():
    rng = np.random.default_rng(0)
    lengths = sample_journey_lengths(100_000, rng)
    assert lengths.min() >= 1
    assert lengths.max() <= 40
    assert np.quantile(lengths, 0.5) == 1
    assert np.quantile(lengths, 0.75) == 1
    assert np.quantile(lengths, 0.90) <= 3


def test_true_aggregate_share_is_normalized_weights():
    w = [4, 3, 2, 1]
    A = true_aggregate_share(w)
    assert A == pytest.approx([0.4, 0.3, 0.2, 0.1])
    assert A.sum() == pytest.approx(1.0)


def test_generate_synthetic_journeys_matches_closed_form_ground_truth():
    rng = np.random.default_rng(1)
    weights = [5, 3, 2]
    A_true = true_aggregate_share(weights)
    journeys = generate_synthetic_journeys(200_000, weights, rng)
    A_hat = aggregate_share(journeys, 3)
    assert A_hat == pytest.approx(A_true, abs=0.01)


def test_generate_synthetic_journeys_touch_timestamps_precede_conversion():
    rng = np.random.default_rng(2)
    journeys = generate_synthetic_journeys(500, [1, 1], rng, window_days=7.0)
    for j in journeys:
        for t in j.touches:
            assert 0.0 <= t.timestamp <= j.conversion_time
