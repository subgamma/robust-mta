import numpy as np
import pytest

from anytime_attribution.journeys import build_journey, linear_weight
from anytime_attribution.models.last_touch import LastTouch
from anytime_attribution.models.linear import Linear
from anytime_attribution.models.time_decay import TimeDecay


def _journey(raw, conversion_time=10.0):
    return build_journey(raw, conversion_time, linear_weight)


def test_last_touch_credits_most_recent_campaign():
    j = _journey([(0, 1.0), (1, 5.0), (0, 8.0)])
    m = LastTouch()
    m.fit([], n_campaigns=2, seed=0)
    z = m.credit(j)
    assert z.tolist() == [1.0, 0.0]


def test_linear_credits_equally():
    j = _journey([(0, 1.0), (1, 5.0), (0, 8.0)])
    m = Linear()
    m.fit([], n_campaigns=2, seed=0)
    z = m.credit(j)
    assert z == pytest.approx([2 / 3, 1 / 3])
    assert z.sum() == pytest.approx(1.0)


def test_time_decay_favours_recent_touches():
    j = _journey([(0, 3.0), (1, 9.0)], conversion_time=10.0)  # tau=7, tau=1
    m = TimeDecay(h=7.0)
    m.fit([], n_campaigns=2, seed=0)
    z = m.credit(j)
    # campaign 1 (tau=1, weight=2**(-1/7)) should dominate campaign 0 (tau=7, weight=0.5)
    assert z[1] > z[0]
    assert z.sum() == pytest.approx(1.0)
    w0, w1 = 2.0 ** (-7 / 7), 2.0 ** (-1 / 7)
    assert z[0] == pytest.approx(w0 / (w0 + w1))
    assert z[1] == pytest.approx(w1 / (w0 + w1))


def test_time_decay_reduces_to_linear_at_equal_recency():
    j = _journey([(0, 9.0), (1, 9.0)], conversion_time=10.0)
    m = TimeDecay(h=7.0)
    m.fit([], n_campaigns=2, seed=0)
    z = m.credit(j)
    assert z == pytest.approx([0.5, 0.5])
