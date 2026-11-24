"""Identification tests. The exact-enumeration helpers compute per-journey expected observed shares directly."""
import itertools
import math
import random

import numpy as np
import pytest

from anytime_attribution.identification import (
    gamma_eff,
    prop2_interval,
    ratio_bounds,
    ordering_identified,
    gap_vertex_extrema,
    prop3_gap_interval,
    lemma1_bounds,
    prop7_envelope,
    robust_A_interval,
)
from anytime_attribution.synth import two_type_mixture, two_campaign_mixture, exact_Q, exact_q


def _binom_pmf(n, p):
    return [math.comb(n, m) * p ** m * (1 - p) ** (n - m) for m in range(n + 1)]


def _per_journey_obs_share(N, p, j):
    pmfs = [_binom_pmf(N[k], p[k]) for k in range(len(N))]
    num = 0.0
    den = 0.0
    for ms in itertools.product(*[range(N[k] + 1) for k in range(len(N))]):
        M = sum(ms)
        if M == 0:
            continue
        pr = 1.0
        for k in range(len(N)):
            pr *= pmfs[k][ms[k]]
        num += pr * ms[j] / M
        den += pr
    return num / den


def _sample_p(K, Gamma, p_lo, rng):
    while True:
        base = rng.uniform(p_lo, 1.0)
        p = [min(1.0, max(p_lo, base * Gamma ** rng.uniform(-1, 1))) for _ in range(K)]
        if max(p) / min(p) <= Gamma + 1e-12:
            return p


def test_lemma1_envelope_holds_on_2000_random_journeys():
    rng = random.Random(0)
    K = 3
    for Gamma, p_lo in [(1.5, 0.05), (2.0, 0.05), (3.0, 0.02)]:
        checked = 0
        trials = 0
        while checked < 2000 and trials < 20000:
            trials += 1
            N = tuple(rng.randint(0, 5) for _ in range(K))
            if sum(N) == 0:
                continue
            j = rng.randrange(K)
            if N[j] == 0:
                continue
            p = _sample_p(K, Gamma, p_lo, rng)
            e = _per_journey_obs_share(N, p, j)
            s_j = N[j] / sum(N)
            lo, hi = lemma1_bounds(s_j, Gamma)
            assert lo - 1e-9 <= e <= hi + 1e-9, (N, p, j, e, lo, hi)
            checked += 1
        assert checked == 2000


def _obs_share_weighted(N, w, p, j):
    touches = [(c, wt) for c in range(len(N)) for wt in w[c][: N[c]]]
    num = den = 0.0
    for R in itertools.product([0, 1], repeat=len(touches)):
        pr = 1.0
        for (c, _), r in zip(touches, R):
            pr *= p[c] if r else 1 - p[c]
        T = sum(wt for (c, wt), r in zip(touches, R) if r)
        if T == 0:
            continue
        Tj = sum(wt for (c, wt), r in zip(touches, R) if r and c == j)
        num += pr * Tj / T
        den += pr
    return num / den


@pytest.mark.parametrize("p,expected_ratio", [(0.9, 10.0), (0.5, 34.0), (0.2, 45.0)])
def test_prop8_weighted_distortion_reproduces_reference_numbers(p, expected_ratio):
    """Dashboard distortion with weights: reference values 10.0, 34.0, 45.0, ratio = (1 + omega*(1-p)) / (2-p) at
    omega=100.
    """
    N = (1, 1)
    w = ([1.0], [100.0])
    e = _obs_share_weighted(N, w, (p, p), 0)
    true_share = 1.0 / 101.0
    assert e / true_share == pytest.approx(expected_ratio, rel=1e-3)


def test_prop2_interval_attained_at_stated_p():
    rng = random.Random(1)
    for _ in range(200):
        K = rng.randint(2, 5)
        a = np.array([rng.uniform(0.01, 1.0) for _ in range(K)])
        a = a / a.sum()
        j = rng.randrange(K)
        Gamma = rng.choice([1.0, 1.25, 1.5, 2.0, 3.0])
        lo, hi = prop2_interval(a[j], Gamma)

        u_lo = np.full(K, Gamma)
        u_lo[j] = 1.0
        A_lo = (a[j] * u_lo[j]) / np.sum(a * u_lo)
        assert A_lo == pytest.approx(lo, rel=1e-9)

        u_hi = np.ones(K)
        u_hi[j] = Gamma
        A_hi = (a[j] * u_hi[j]) / np.sum(a * u_hi)
        assert A_hi == pytest.approx(hi, rel=1e-9)


def test_prop2_interval_trivial_at_gamma_one():
    lo, hi = prop2_interval(0.3, 1.0)
    assert lo == pytest.approx(0.3)
    assert hi == pytest.approx(0.3)


def test_ratio_bounds_matches_corollary():
    a_j, a_k, Gamma = 0.4, 0.2, 1.5
    lo, hi = ratio_bounds(a_j, a_k, Gamma)
    assert lo == pytest.approx(a_j / (Gamma * a_k))
    assert hi == pytest.approx(Gamma * a_j / a_k)


def _bruteforce_gap_extrema(l, h, j, k):
    K = len(l)
    best_min, best_max = math.inf, -math.inf
    for corner in itertools.product([0, 1], repeat=K):
        z = np.array([l[m] if corner[m] == 0 else h[m] for m in range(K)])
        d = z.sum()
        if d <= 0:
            continue
        g = (z[j] - z[k]) / d
        best_min = min(best_min, g)
        best_max = max(best_max, g)
    return best_min, best_max


def test_gap_vertex_extrema_matches_bruteforce_corners():
    rng = random.Random(2)
    for _ in range(300):
        K = rng.randint(2, 5)
        l = np.array([rng.choice([0.0, round(rng.uniform(0.01, 1.0), 4)]) for _ in range(K)])
        h = l + np.array([rng.uniform(0.1, 3.0) for _ in range(K)])
        j, k = rng.sample(range(K), 2)
        g_min, g_max = gap_vertex_extrema(l, h, j, k)
        best_min, best_max = _bruteforce_gap_extrema(l, h, j, k)
        assert g_min == pytest.approx(best_min, abs=1e-9)
        assert g_max == pytest.approx(best_max, abs=1e-9)


def test_gap_vertex_extrema_handles_zero_lower_bounds():
    l = np.array([0.0, 0.0, 0.2])
    h = np.array([1.0, 1.0, 1.0])
    g_min, g_max = gap_vertex_extrema(l, h, 0, 1)
    best_min, best_max = _bruteforce_gap_extrema(l, h, 0, 1)
    assert g_min == pytest.approx(best_min, abs=1e-9)
    assert g_max == pytest.approx(best_max, abs=1e-9)


def test_ordering_corollary_matches_gap_sign():
    rng = random.Random(3)
    for _ in range(300):
        a_j = rng.uniform(0.01, 1.0)
        a_k = rng.uniform(0.01, 1.0)
        Gamma = rng.uniform(1.0, 3.0)
        ident = ordering_identified(a_j, a_k, Gamma)
        lo, _ = prop3_gap_interval(np.array([a_j, a_k]), Gamma, 0, 1)
        assert ident == (lo > 0)


def test_gamma_eff():
    assert gamma_eff(2.0, 0.5) == 2.0
    assert gamma_eff(3.0, 0.5) == 2.0  # capped at 1/p_lo


def test_prop7_envelope_contains_q_and_is_valid_interval():
    for Gamma, p_lo, n_min, q_j in [(1.5, 0.2, 1, 0.3), (2.0, 0.5, 3, 0.1)]:
        lo, hi = prop7_envelope(q_j, Gamma, p_lo, n_min)
        assert lo <= q_j <= hi
        assert lo >= 0


def test_prop7_envelope_matches_tightened_formula():
    """Q_j in [(s_low/Gamma_eff) q_j, q_j/p_lo]; the upper endpoint is equivalently q_j/Q_j >= p_lo, not p_lo/Gamma_eff."""
    Gamma, p_lo, n_min, q_j = 1.5, 0.2, 1, 0.3
    g_eff = gamma_eff(Gamma, p_lo)
    s_low = 1.0 - (1.0 - p_lo) ** n_min
    lo, hi = prop7_envelope(q_j, Gamma, p_lo, n_min)
    assert lo == pytest.approx((s_low / g_eff) * q_j)
    assert hi == pytest.approx(q_j / p_lo)


def test_prop7_ratio_bounds_p_lo_to_gamma_eff_over_s_low():
    """Equivalent form: p_lo <= q_j/Q_j <= Gamma_eff/s_low."""
    Gamma, p_lo, n_min, q_j = 2.0, 0.25, 2, 0.15
    g_eff = gamma_eff(Gamma, p_lo)
    s_low = 1.0 - (1.0 - p_lo) ** n_min
    lo, hi = prop7_envelope(q_j, Gamma, p_lo, n_min)
    # Q_j in [lo, hi] => q_j/Q_j in [q_j/hi, q_j/lo]
    ratio_lo, ratio_hi = q_j / hi, q_j / lo
    assert ratio_lo == pytest.approx(p_lo)
    assert ratio_hi == pytest.approx(g_eff / s_low)


# Attaining sequences: never violated at finite N, approaching the endpoints as pi -> 0, N -> infinity.
def test_prop7_lower_attaining_sequence_never_violates_and_converges():
    p_lo, Gamma = 0.2, 1.5
    ratios = []
    for N in [10, 100, 1000, 10000]:
        pi = 1.0 / N  # pi -> 0 alongside N -> infinity
        mix = two_type_mixture(2, j=0, k=1, n_short=1, n_long=N, pi=pi)
        Q = exact_Q(mix, 2)
        q = exact_q(mix, p=(p_lo, p_lo), n_campaigns=2, method="exact")
        ratio = q[0] / Q[0]
        ratios.append(ratio)
        assert ratio >= p_lo - 1e-9  # never violates the tightened lower bound
    # monotone approach to p_lo from above
    assert ratios[-1] < ratios[0]
    assert ratios[-1] == pytest.approx(p_lo, abs=1e-3)


def test_prop7_upper_attaining_sequence_never_violates_and_converges():
    p_lo, Gamma = 0.2, 1.5
    g_eff = gamma_eff(Gamma, p_lo)
    p_hi = g_eff * p_lo
    ratios = []
    for N in [10, 100, 1000, 10000]:
        pi = 1.0 / N
        mix = two_campaign_mixture(2, j=0, k=1, n_j1=1, n_k1=N, n_j2=0, n_k2=1, pi=pi)
        Q = exact_Q(mix, 2)
        q = exact_q(mix, p=(p_hi, p_lo), n_campaigns=2, method="exact")
        ratio = q[0] / Q[0]
        ratios.append(ratio)
        assert ratio <= g_eff / p_lo + 1e-6  # never violates the upper bound (n_min=1: s_low=p_lo)
    # monotone approach to Gamma_eff/p_lo from below
    assert ratios[-1] > ratios[0]
    assert ratios[-1] == pytest.approx(g_eff / p_lo, rel=1e-3)


def test_prop2_closed_form_identity_eq_aggmap():
    """Closed-form identity a_j = p_j A_j / sum(p_k A_k), checked against a direct computation for random p, A. An
    algebraic unit test, not a coverage claim.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        K = rng.integers(2, 6)
        A = rng.uniform(0.1, 1.0, size=K)
        A = A / A.sum()
        p = rng.uniform(0.1, 1.0, size=K)
        a = p * A / np.sum(p * A)
        assert a.sum() == pytest.approx(1.0)
        for j in range(K):
            assert a[j] == pytest.approx(p[j] * A[j] / np.sum(p * A))


def test_robust_A_interval_monotone_composition_contains_truth():
    """`robust_A_interval` composes the monotone identified-set map with a CS box [L_t,U_t] on the observed share.

        Whenever a_j is in [L_t,U_t], A_j is in [lo(L_t), hi(U_t)] deterministically, since both endpoints of
        Theta_j(Gamma) are strictly increasing in a_j.
    """
    Gamma = 1.5
    rng = np.random.default_rng(1)
    for _ in range(200):
        a_j = rng.uniform(0.01, 0.99)
        width = rng.uniform(0.001, 0.05)
        L_t = max(0.0, a_j - width)
        U_t = min(1.0, a_j + width)
        A_true_lo, A_true_hi = prop2_interval(a_j, Gamma)
        lo, hi = robust_A_interval(L_t, U_t, Gamma)
        assert lo <= A_true_lo
        assert hi >= A_true_hi
        assert lo <= hi


def test_robust_A_interval_matches_prop2_interval_at_zero_width_box():
    lo, hi = robust_A_interval(0.3, 0.3, 1.5)
    exact_lo, exact_hi = prop2_interval(0.3, 1.5)
    assert lo == pytest.approx(exact_lo)
    assert hi == pytest.approx(exact_hi)
