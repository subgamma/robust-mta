import pytest

from anytime_attribution.journeys import build_journey, linear_weight
from anytime_attribution.estimands import (
    campaign_masses,
    per_journey_shares,
    aggregate_share,
    dashboard_share,
)


def _journey(raw, conversion_time=10.0):
    return build_journey(raw, conversion_time, linear_weight)


def test_campaign_masses():
    j = _journey([(0, 1.0), (0, 2.0), (1, 3.0)])
    m = campaign_masses(j, n_campaigns=2)
    assert list(m) == [2.0, 1.0]


def test_per_journey_shares_sum_to_one():
    j = _journey([(0, 1.0), (1, 2.0), (1, 3.0)])  # linear: masses (1, 2)
    z = per_journey_shares(j, n_campaigns=2)
    assert z.sum() == pytest.approx(1.0)
    assert list(z) == pytest.approx([1 / 3, 2 / 3])


def test_per_journey_shares_raises_on_zero_mass():
    j = _journey([])
    with pytest.raises(ValueError):
        per_journey_shares(j, n_campaigns=2)


def test_aggregate_share_matches_hand_computation():
    stream = [
        _journey([(0, 1.0), (0, 1.0)]),  # S=(2,0)
        _journey([(1, 1.0)]),  # S=(0,1)
        _journey([(0, 1.0), (1, 1.0)]),  # S=(1,1)
    ]
    a = aggregate_share(stream, n_campaigns=2)
    assert list(a) == pytest.approx([3 / 5, 2 / 5])


def test_dashboard_share_matches_hand_computation():
    stream = [
        _journey([(0, 1.0), (0, 1.0)]),  # share (1, 0)
        _journey([(1, 1.0)]),  # share (0, 1)
        _journey([(0, 1.0), (1, 1.0)]),  # share (0.5, 0.5)
    ]
    q = dashboard_share(stream, n_campaigns=2)
    assert list(q) == pytest.approx([(1 + 0 + 0.5) / 3, (0 + 1 + 0.5) / 3])


def test_aggregate_and_dashboard_differ_under_length_imbalance():
    """Q and A need not coincide when journey mass correlates with a campaign's share (campaign-0 journeys are short,
    campaign-1 journeys long).
    """
    stream = [
        _journey([(0, 1.0)]),
        _journey([(1, 1.0)] * 9),
    ]
    a = aggregate_share(stream, n_campaigns=2)
    q = dashboard_share(stream, n_campaigns=2)
    assert a[0] != pytest.approx(q[0])


def test_aggregate_share_raises_on_empty_stream():
    with pytest.raises(ValueError):
        aggregate_share([], n_campaigns=2)


def test_dashboard_share_raises_on_empty_stream():
    with pytest.raises(ValueError):
        dashboard_share([], n_campaigns=2)
