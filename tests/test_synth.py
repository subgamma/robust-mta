import numpy as np
import pytest

from anytime_attribution.synth import (
    JourneyType,
    Mixture,
    exact_A,
    exact_Q,
    exact_a,
    exact_q,
    type_moments,
    two_type_mixture,
    two_campaign_mixture,
    retention_grid,
    P_LO_GRID,
    GAMMA_GRID,
)


def test_journey_type_rejects_empty():
    with pytest.raises(ValueError):
        JourneyType((0, 0, 0))


def test_mixture_rejects_mismatched_lengths_and_bad_weights():
    with pytest.raises(ValueError):
        Mixture(types=(JourneyType((1, 0)),), weights=(0.5, 0.5))
    with pytest.raises(ValueError):
        Mixture(types=(JourneyType((1, 0)), JourneyType((0, 1))), weights=(0.4, 0.4))


def test_exact_A_and_Q_two_type_mixture():
    mix = two_type_mixture(2, j=0, k=1, n_short=1, n_long=9, pi=0.5)
    A = exact_A(mix, 2)
    Q = exact_Q(mix, 2)
    assert A == pytest.approx([0.1, 0.9])
    assert Q == pytest.approx([0.5, 0.5])  # Q_j = pi exactly, independent of n_long


def test_exact_a_equals_A_under_equal_retention():
    """Gamma=1 (equal retention): a equals A exactly (Theta(1)={a})."""
    mix = two_type_mixture(2, j=0, k=1, n_short=1, n_long=9, pi=0.3)
    A = exact_A(mix, 2)
    a = exact_a(mix, p=(0.5, 0.5), n_campaigns=2)
    assert a == pytest.approx(A)


def test_exact_q_differs_from_Q_via_whole_journey_dropout():
    """Even at equal retention, q != Q under length imbalance."""
    mix = two_type_mixture(2, j=0, k=1, n_short=1, n_long=9, pi=0.5)
    Q = exact_Q(mix, 2)
    q = exact_q(mix, p=(0.5, 0.5), n_campaigns=2, method="exact")
    assert q[0] < Q[0]  # short-journey campaign under-represented in the observed stream


def test_exact_q_matches_exact_a_at_gamma_one_in_the_sparse_limit():
    """As retention -> 0 (fixed ratio r=1), q_j -> the count-aggregate share."""
    mix = two_type_mixture(2, j=0, k=1, n_short=1, n_long=4, pi=0.5)
    q_small_p = exact_q(mix, p=(0.01, 0.01), n_campaigns=2, method="exact")
    count_aggregate = exact_A(Mixture(types=mix.types, weights=mix.weights), 2)  # same as exact_A since weight=1 (linear)
    assert q_small_p[0] == pytest.approx(count_aggregate[0], abs=0.02)


def test_type_moments_exact_matches_monte_carlo():
    counts = (3, 4)
    p = (0.6, 0.3)
    rng = np.random.default_rng(0)
    exact_ratio, exact_surv = type_moments(counts, p, 2, method="exact")
    mc_ratio, mc_surv = type_moments(counts, p, 2, method="mc", mc_samples=2_000_000, rng=rng)
    assert mc_ratio == pytest.approx(exact_ratio, abs=5e-3)
    assert mc_surv == pytest.approx(exact_surv, abs=5e-3)


def test_two_campaign_mixture_general_family():
    mix = two_campaign_mixture(2, j=0, k=1, n_j1=1, n_k1=10, n_j2=0, n_k2=1, pi=0.01)
    Q = exact_Q(mix, 2)
    q = exact_q(mix, p=(0.3, 0.2), n_campaigns=2, method="exact")
    assert Q[0] == pytest.approx(0.01 * (1 / 11))
    assert q[0] > 0


def test_retention_grid_matches_constants():
    grid = retention_grid()
    assert len(grid) == len(P_LO_GRID) * len(GAMMA_GRID)
    assert (0.2, 1.0) in grid
    assert (0.8, 2.0) in grid
