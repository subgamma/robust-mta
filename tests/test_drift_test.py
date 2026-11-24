"""No silent clipping of OBSERVATIONS in any confidence-sequence
input path (older betting/diagnostic paths included). Internal parameter constraints
(bet-size clipping, variance floors, anchor clamps) are legitimate and untouched."""
import numpy as np
import pytest

from anytime_attribution.cs import (
    ConfidenceSequence,
    HowardEBProcess,
    MassBoundError,
    MultiCampaignCumulativeShareCS,
    MultiCampaignShareCSGrid,
    MultiHowardEBProcess,
    ShareCSGrid,
    drift_test_cs,
    invert_share_cs,
    theta_rejected,
)

B = 10.0
BAD = [-0.5, B + 0.5, float("nan"), float("inf"), float("-inf")]
IDS = ["negative", "above_B", "nan", "+inf", "-inf"]


@pytest.mark.parametrize("bad", BAD, ids=IDS)
def test_confidence_sequence_update_raises(bad):
    cs = ConfidenceSequence(0.05, lo=0.0, hi=B)
    cs.update(1.0)
    with pytest.raises(MassBoundError):
        cs.update(bad)
    assert len(cs._history) == 1  # rejected observation not stored
    assert issubclass(MassBoundError, ValueError)  # backward compatible with `except ValueError`


@pytest.mark.parametrize("bad", BAD, ids=IDS)
def test_share_grid_updates_raise_and_leave_state(bad):
    g = ShareCSGrid(0.05, B, grid_size=51)
    g.update(1.0, 3.0)
    before = (g._lower.rejected.copy(), g._lower.t if hasattr(g._lower, "t") else None)
    with pytest.raises(MassBoundError):
        g.update(bad, 3.0)
    with pytest.raises(MassBoundError):
        g.update(1.0, bad)
    assert np.array_equal(g._lower.rejected, before[0])
    mg = MultiCampaignShareCSGrid(3, 0.05, B, grid_size=51)
    mg.update(np.array([1.0, 0.0, 0.0]), 1.0)
    with pytest.raises(MassBoundError):
        mg.update(np.array([1.0, bad, 0.0]), 3.0)
    with pytest.raises(MassBoundError):
        mg.update(np.array([1.0, 0.0, 0.0]), bad)


@pytest.mark.parametrize("bad", BAD, ids=IDS)
def test_theta_rejected_invert_and_drift_raise(bad):
    good = [1.0, 2.0, 0.0]
    tot = [1.0, 2.0, 1.0]
    with pytest.raises(MassBoundError):
        theta_rejected([1.0, bad], [2.0, 2.0], 0.3, 0.05, B)
    with pytest.raises(MassBoundError):
        theta_rejected(good, [1.0, 2.0, bad], 0.3, 0.05, B)
    with pytest.raises(MassBoundError):
        invert_share_cs([1.0, bad], [2.0, 2.0], 0.05, B)
    with pytest.raises(MassBoundError):
        drift_test_cs([1.0, bad, 0.0], tot, 0.05, B)
    with pytest.raises(MassBoundError):
        drift_test_cs(good, [1.0, bad, 1.0], 0.05, B)


@pytest.mark.parametrize("bad", BAD, ids=IDS)
def test_howard_paths_still_raise(bad):
    with pytest.raises(MassBoundError):
        MultiCampaignCumulativeShareCS(2, 0.05, B).update(np.array([bad, 0.0]), 1.0)
    with pytest.raises(MassBoundError):
        MultiCampaignCumulativeShareCS(2, 0.05, B).update(np.array([1.0, 0.0]), bad)
    with pytest.raises(MassBoundError):
        HowardEBProcess(0.01).update(bad)
    with pytest.raises(MassBoundError):
        MultiHowardEBProcess(2, 0.01).update(np.array([0.1, bad]))


def test_valid_boundary_and_ordinary_values_accepted():
    cs = ConfidenceSequence(0.05, lo=0.0, hi=B)
    for x in (0.0, B, 3.3, B * (1 + 1e-12), -1e-12):  # last two: floating-point rounding slack, snapped
        cs.update(x)
    assert min(cs._history) >= 0.0 and max(cs._history) <= B
    g = ShareCSGrid(0.05, B, grid_size=51)
    g.update(0.0, 0.0); g.update(B, B); g.update(2.5, 6.0)
    mg = MultiCampaignShareCSGrid(2, 0.05, B, grid_size=51)
    mg.update(np.array([0.0, B]), B); mg.update(np.array([0.0, 0.0]), 0.0)
    assert isinstance(theta_rejected([0.0, B, 3.0], [B, B, 5.0], 0.5, 0.05, B), (bool, np.bool_))
    r = drift_test_cs([1.0, 2.0, 0.0, 3.0] * 10, [3.0, 4.0, 2.0, 5.0] * 10, 0.05, B, n_coarse=51, n_refine=201)
    assert r.status in {"interval", "empty", "indeterminate"}
    lo, hi = invert_share_cs([1.0, 2.0, 0.0, 3.0] * 5, [3.0, 4.0, 2.0, 5.0] * 5, 0.05, B)
    assert 0.0 <= lo <= hi <= 1.0


def test_validators_do_not_change_valid_results():
    """Valid inputs pass through the validators bit-for-bit: identical to a direct
    computation from the raw values (no rescaling)."""
    rng = np.random.default_rng(0)
    Tj = rng.uniform(0, 3, 60); T = Tj + rng.uniform(0, 3, 60)
    g = ShareCSGrid(0.05, B, grid_size=41)
    g2 = ShareCSGrid(0.05, B, grid_size=41)
    for a, b in zip(Tj, T):
        g.update(a, b)
        g2.update(float(a), float(b))
    assert g.interval() == g2.interval()
