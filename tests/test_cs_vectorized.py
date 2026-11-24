"""The vectorised Howard evaluation reproduces the production incremental classes exactly (m=1)."""
import numpy as np
import pytest

from anytime_attribution.cs import HowardEBProcess, MultiCampaignCumulativeShareCS, howard_eb_radius
from anytime_attribution.cs_vectorized import (
    cumulative_share_boxes,
    howard_radius_vec,
    lagged_mean_variance_process,
)


def test_radius_matches_production_function():
    rng = np.random.default_rng(0)
    for _ in range(200):
        V = float(rng.uniform(0, 5000)); t = int(rng.integers(1, 100000)); a = float(rng.choice([0.0125, 0.001, 0.2]))
        assert howard_radius_vec(np.array([V]), np.array([float(t)]), a)[0] == pytest.approx(howard_eb_radius(V, t, a), rel=1e-12)


def test_scalar_process_matches_incremental_class():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 300) * (rng.random(300) < 0.3)
    mean, V = lagged_mean_variance_process(x)
    p = HowardEBProcess(0.01)
    for i, xi in enumerate(x):
        p.update(float(xi))
        assert p.V_hat == pytest.approx(V[i], rel=1e-12, abs=1e-12)
        assert p.sum_x / p.t == pytest.approx(mean[i], rel=1e-12)


def test_ratio_boxes_match_production_cs_at_every_step():
    rng = np.random.default_rng(2)
    K, n, B, alpha = 4, 400, 10.0, 0.05
    camp = rng.choice(K, size=n, p=[0.4, 0.3, 0.2, 0.1]); w = rng.integers(1, 11, size=n).astype(float)
    Tj = np.zeros((n, K)); Tj[np.arange(n), camp] = w; T = w
    cs = MultiCampaignCumulativeShareCS(K, alpha, B)
    box = cumulative_share_boxes(Tj, T, B, cs.alpha_used)
    for i in range(n):
        cs.update(Tj[i], T[i])
        L, U = cs.intervals(); LN, UN = cs.numerator_boxes()
        assert np.allclose(L, box["L"][i], atol=1e-12) and np.allclose(U, box["U"][i], atol=1e-12)
        assert np.allclose(LN, box["LN"][i], atol=1e-12) and np.allclose(UN, box["UN"][i], atol=1e-12)


def test_out_of_range_raises_not_clips():
    with pytest.raises(ValueError):
        cumulative_share_boxes(np.array([[11.0, 0.0]]), np.array([11.0]), 10.0, 0.01)
    with pytest.raises(ValueError):
        cumulative_share_boxes(np.array([[np.nan, 0.0]]), np.array([1.0]), 10.0, 0.01)
