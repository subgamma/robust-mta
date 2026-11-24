import numpy as np
import pytest

from anytime_attribution.journeys import (
    build_journey,
    linear_weight,
    time_decay_weight,
    apply_cap,
    thin,
)


def test_linear_weight_is_one():
    assert linear_weight(0.0, 10.0) == 1.0
    assert linear_weight(5.0, 10.0) == 1.0


def test_time_decay_weight_formula():
    assert time_decay_weight(10.0, 10.0, h=7.0) == pytest.approx(1.0)
    assert time_decay_weight(3.0, 10.0, h=7.0) == pytest.approx(0.5)


def test_time_decay_weight_rejects_future_touch():
    with pytest.raises(ValueError):
        time_decay_weight(11.0, 10.0, h=7.0)


def test_build_journey_linear():
    raw = [(0, 1.0), (1, 5.0), (0, 8.0)]
    j = build_journey(raw, conversion_time=10.0, weight_fn=linear_weight)
    assert j.n_touches == 3
    assert j.mass == 3.0
    assert all(t.weight == 1.0 for t in j.touches)


def test_apply_cap_lookback():
    raw = [(0, 0.0), (0, 5.0), (0, 9.0)]  # tau = 10, 5, 1
    j = build_journey(raw, conversion_time=10.0, weight_fn=linear_weight)
    capped = apply_cap(j, lookback_days=6.0, max_touches=10)
    assert capped.n_touches == 2
    assert [t.timestamp for t in capped.touches] == [5.0, 9.0]


def test_apply_cap_max_touches_keeps_nearest_to_conversion():
    raw = [(0, 0.0), (0, 3.0), (0, 6.0), (0, 9.0)]
    j = build_journey(raw, conversion_time=10.0, weight_fn=linear_weight)
    capped = apply_cap(j, lookback_days=100.0, max_touches=2)
    assert [t.timestamp for t in capped.touches] == [6.0, 9.0]


def test_apply_cap_is_pure():
    raw = [(0, 0.0), (0, 9.0)]
    j = build_journey(raw, conversion_time=10.0, weight_fn=linear_weight)
    c1 = apply_cap(j, lookback_days=5.0, max_touches=10)
    c2 = apply_cap(j, lookback_days=5.0, max_touches=10)
    assert c1 == c2
    assert j.n_touches == 2  # original untouched


def test_thin_drops_touches_and_can_return_none():
    raw = [(0, 0.0), (1, 5.0)]
    j = build_journey(raw, conversion_time=10.0, weight_fn=linear_weight)
    rng = np.random.default_rng(0)
    obs = thin(j, {0: 1.0, 1: 0.0}, rng)
    assert obs is not None
    assert obs.n_touches == 1
    assert obs.touches[0].campaign == 0

    obs_none = thin(j, {0: 0.0, 1: 0.0}, rng)
    assert obs_none is None


def test_thin_retained_touch_keeps_weight():
    raw = [(0, 3.0)]
    j = build_journey(raw, conversion_time=10.0, weight_fn=time_decay_weight)
    rng = np.random.default_rng(0)
    obs = thin(j, {0: 1.0}, rng)
    assert obs.touches[0].weight == j.touches[0].weight
