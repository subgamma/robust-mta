"""Targeted tests: floored identification, time-varying retention counterexample and extension, marginal-mean
invariance, the G3 mechanism, and raise-instead-of-clip.

Counterexamples reproduce the worked example a=(0.4,0.3,0.3), Gamma=2, p_lo=0.8.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from anytime_attribution.cs import (
    HowardEBProcess,
    MassBoundError,
    MultiCampaignCumulativeShareCS,
    MultiHowardEBProcess,
)
from anytime_attribution.identification import (
    gamma_eff,
    gamma_star_floor_saturated,
    ordering_identified,
    prop2_interval,
    prop3_gap_interval,
    ratio_bounds,
    robust_A_interval,
)


def _feasible_vertices(K: int, Gamma: float, p_lo: float, grid: int = 41):
    """Retention vectors spanning P(Gamma, p_lo): for each minimum value pmin
    on a grid, every coordinate is pmin or min(1, Gamma*pmin). A_j and the gap
    are monotone coordinate-wise, so extremes sit at such vertices."""
    for pmin in np.linspace(p_lo, 1.0, grid):
        hi = min(1.0, Gamma * pmin)
        for combo in itertools.product([pmin, hi], repeat=K):
            if min(combo) == pmin:
                yield np.array(combo)


def _A_from(a, p):
    u = np.asarray(a) / np.asarray(p)
    return u / u.sum()


def _feasible_A_range(a, j, Gamma, p_lo):
    vals = [_A_from(a, p)[j] for p in _feasible_vertices(len(a), Gamma, p_lo)]
    return min(vals), max(vals)


def _feasible_gap_range(a, j, k, Gamma, p_lo):
    vals = []
    for p in _feasible_vertices(len(a), Gamma, p_lo):
        A = _A_from(a, p)
        vals.append(A[j] - A[k])
    return min(vals), max(vals)


A_OBS = [0.4, 0.3, 0.3]


@pytest.mark.parametrize("Gamma,p_lo", [(2.0, 1e-6), (2.0, 0.5), (5.0, 0.2), (1.25, 0.8)])
def test_prop2_sharp_when_gamma_leq_inverse_floor(Gamma, p_lo):
    assert Gamma <= 1.0 / p_lo + 1e-9
    lo, hi = _feasible_A_range(A_OBS, 0, Gamma, p_lo)
    tlo, thi = prop2_interval(A_OBS[0], Gamma)
    assert lo == pytest.approx(tlo, abs=1e-9) and hi == pytest.approx(thi, abs=1e-9)


@pytest.mark.parametrize("Gamma,p_lo", [(2.0, 0.8), (2.0, 0.9)])
def test_gamma_interval_not_sharp_when_gamma_exceeds_inverse_floor(Gamma, p_lo):
    """Counterexample: Gamma=2, p_lo=0.8 gives a maximum retention ratio of 1.25."""
    assert Gamma > 1.0 / p_lo
    lo, hi = _feasible_A_range(A_OBS, 0, Gamma, p_lo)
    glo, ghi = prop2_interval(A_OBS[0], Gamma)  # ratio-only set: valid superset
    assert glo < lo - 1e-3 and ghi > hi + 1e-3  # strictly larger: NOT sharp here
    flo, fhi = prop2_interval(A_OBS[0], Gamma, p_lo)  # floored: Gamma_eff
    assert flo == pytest.approx(lo, abs=1e-9) and fhi == pytest.approx(hi, abs=1e-9)
    # documented numbers for Gamma=2, p_lo=0.8
    if p_lo == 0.8:
        assert (round(lo, 4), round(hi, 4)) == (0.3478, 0.4545)
        assert (round(glo, 4), round(ghi, 4)) == (0.25, 0.5714)


def test_ordering_threshold_uses_gamma_eff_under_floor():
    aj, ak = 0.5, 0.3  # ratio 1.667
    assert not ordering_identified(aj, ak, 2.0)  # ratio-only: Gamma = 2
    assert ordering_identified(aj, ak, 2.0, p_lo=0.8)  # floor: Gamma_eff = 1.25
    assert ordering_identified(aj, ak, 2.0, p_lo=0.8) == (aj / ak > gamma_eff(2.0, 0.8))
    assert not ordering_identified(aj, ak, 2.0, p_lo=1e-6)  # floor inactive


@pytest.mark.parametrize("Gamma,p_lo", [(2.0, 0.8), (3.0, 0.5), (1.5, 0.9), (4.0, 0.2)])
def test_gap_and_ratio_sharp_with_gamma_eff(Gamma, p_lo):
    rng = np.random.default_rng(7)
    for _ in range(5):
        a = rng.dirichlet(np.ones(4))
        for j, k in [(0, 1), (2, 3)]:
            lo, hi = _feasible_gap_range(a, j, k, Gamma, p_lo)
            tlo, thi = prop3_gap_interval(a, Gamma, j, k, p_lo)
            assert lo == pytest.approx(tlo, abs=1e-9) and hi == pytest.approx(thi, abs=1e-9)
            rlo, rhi = ratio_bounds(a[j], a[k], Gamma, p_lo)
            ratios = [(_A_from(a, p)[j] / _A_from(a, p)[k]) for p in _feasible_vertices(4, Gamma, p_lo)]
            assert min(ratios) == pytest.approx(rlo, rel=1e-9) and max(ratios) == pytest.approx(rhi, rel=1e-9)


def test_defaults_are_ratio_only_and_unchanged():
    """Without p_lo the formulas equal the ratio-only ones."""
    a, G = 0.37, 1.7
    assert prop2_interval(a, G) == (a / (G - (G - 1) * a), G * a / (1 + (G - 1) * a))
    assert robust_A_interval(0.3, 0.4, 1.5) == (prop2_interval(0.3, 1.5)[0], prop2_interval(0.4, 1.5)[1])
    assert robust_A_interval(0.3, 0.4, 3.0, p_lo=0.5) == robust_A_interval(0.3, 0.4, 2.0)


def test_gamma_star_floor_saturation_semantics():
    assert gamma_star_floor_saturated(5.5, 0.2)  # Gamma* > 1/p_lo = 5: floor alone suffices
    assert not gamma_star_floor_saturated(4.28, 0.2)  # largest E6 value is below 5
    assert not gamma_star_floor_saturated(5.0, 0.2)  # strict
    with pytest.raises(ValueError):
        gamma_star_floor_saturated(2.0, 0.0)


def _a_cumulative(S, P):
    """Observed cumulative shares from per-period expected latent masses S
    (periods x K) and retention P (periods x K): mu_ij = p_ij * s_ij."""
    S, P = np.asarray(S, float), np.asarray(P, float)
    num = (P * S).sum(axis=0)
    return num / num.sum()


def _A_cumulative(S):
    S = np.asarray(S, float)
    return S.sum(axis=0) / S.sum()


def test_time_varying_level_shift_with_mix_drift_breaks_original_identity():
    """Each period has per-period retention ratio exactly 1 (Gamma = 1 holds
    period by period), but a common level shift eps between periods with
    campaign-mix drift pushes a_j/a_k to 1/eps while A_j/A_k = 1."""
    eps = 0.05
    S = [[1.0, 0.0], [0.0, 1.0]]  # period 1: only j has mass; period 2: only k
    P = [[1.0, 1.0], [eps, eps]]  # ratio 1 within each period
    assert all(max(r) / min(r) == 1.0 for r in P)
    a, A = _a_cumulative(S, P), _A_cumulative(S)
    assert A[0] / A[1] == pytest.approx(1.0)
    assert a[0] / a[1] == pytest.approx(1.0 / eps)
    # the Gamma=1 identified set for A_j given a_j is the single point a_j: it misses the truth
    lo, hi = prop2_interval(a[0], 1.0)
    assert not (lo - 1e-12 <= A[0] <= hi + 1e-12)
    # even Gamma=10 (ratio-only) cannot contain the truth when eps is small enough
    lo, hi = prop2_interval(a[0], 10.0)
    assert not (lo <= A[0] <= hi)


def test_constant_p_identity_under_drifting_latent_mix():
    """Time-invariant p_j: a_{j,t}/a_{k,t} = (p_j/p_k) A_{j,t}/A_{k,t} at every t
    even when the latent mix drifts arbitrarily (exact expectations)."""
    rng = np.random.default_rng(3)
    K, T = 4, 40
    p = np.array([0.9, 0.5, 0.7, 0.45])
    S = rng.gamma(2.0, size=(T, K)) * np.linspace(0.2, 2.0, T)[:, None] ** rng.integers(-1, 2, size=K)
    for t in range(1, T + 1):
        a = _a_cumulative(S[:t], np.tile(p, (t, 1)))
        A = _A_cumulative(S[:t])
        for j, k in [(0, 1), (2, 3), (1, 3)]:
            assert a[j] / a[k] == pytest.approx(p[j] / p[k] * A[j] / A[k], rel=1e-12)


@pytest.mark.parametrize("Gamma", [1.3, 2.0, 4.0])
def test_spatiotemporal_extension_with_global_gamma(Gamma):
    """Time-varying predictable p_ij whose GLOBAL ratio max p / min p <= Gamma:
    the cumulative A_{j,t} lies in Theta_j(Gamma) computed from a_{j,t}, and the
    cumulative gap lies in the identified gap interval, at every t (pathwise)."""
    rng = np.random.default_rng(11)
    K, T = 4, 60
    pmin = 0.2
    for _ in range(40):
        P = rng.uniform(pmin, pmin * Gamma, size=(T, K))
        P[rng.integers(T), rng.integers(K)] = pmin  # force the extremes to be attained
        P[rng.integers(T), rng.integers(K)] = pmin * Gamma
        assert P.max() / P.min() <= Gamma + 1e-12
        S = rng.gamma(1.5, size=(T, K)) * rng.uniform(0.1, 3.0, size=K)  # drifting, arbitrary mix
        S *= np.linspace(2.0, 0.3, T)[:, None] ** rng.integers(-1, 2, size=K)
        for t in (5, 20, 60):
            a = _a_cumulative(S[:t], P[:t])
            A = _A_cumulative(S[:t])
            for j in range(K):
                lo, hi = prop2_interval(a[j], Gamma)
                assert lo - 1e-12 <= A[j] <= hi + 1e-12
            glo, ghi = prop3_gap_interval(a, Gamma, 0, 1)
            assert glo - 1e-12 <= A[0] - A[1] <= ghi + 1e-12


def test_effective_retention_weighted_average_lies_between_period_extremes():
    """p_bar_{j,t} = sum_i p_ij s_ij / sum_i s_ij is a convex combination of p_ij."""
    rng = np.random.default_rng(5)
    S, P = rng.gamma(2.0, size=(30, 3)), rng.uniform(0.2, 0.9, size=(30, 3))
    pbar = (P * S).sum(axis=0) / S.sum(axis=0)
    assert np.all(pbar >= P.min(axis=0) - 1e-12) and np.all(pbar <= P.max(axis=0) + 1e-12)
    a = _a_cumulative(S, P)
    A = _A_cumulative(S)
    expect = (pbar * A) / (pbar * A).sum()
    assert np.allclose(a, expect, rtol=1e-12)


# ------------------------------------------------------------------------- 
def _journeys():
    """Latent journeys: list of (probability, [(campaign, weight),...])."""
    return [
        (0.5, [(0, 0.25), (1, 1.0)]),
        (0.3, [(1, 0.5), (0, 0.5), (0, 1.0)]),
        (0.2, [(0, 1.0)]),
    ]


def _expected_masses_comonotone(p, K=2):
    """E[T_j] and E[S_j] when retention is comonotone across touches: ONE
    uniform U per journey and R_l = 1{U <= p_{c_l}} (maximal positive
    correlation; marginals are exactly p_{c_l})."""
    ET, ES = np.zeros(K), np.zeros(K)
    for pr, touches in _journeys():
        # integrate over U exactly: breakpoints are the p values
        cuts = sorted({0.0, 1.0, *[p[c] for c, _ in touches]})
        for u0, u1 in zip(cuts[:-1], cuts[1:]):
            u = 0.5 * (u0 + u1)
            for c, w in touches:
                if u <= p[c]:
                    ET[c] += pr * (u1 - u0) * w
        for c, w in touches:
            ES[c] += pr * w
    return ET, ES


def test_marginal_condition_gives_mean_invariance_under_arbitrary_correlation():
    p = np.array([0.8, 0.4])
    ET, ES = _expected_masses_comonotone(p)
    assert np.allclose(ET, p * ES, rtol=1e-12)  # E[T_j] = p_j E[S_j] despite correlation


def _expected_masses_position_dependent(rho, K=2):
    """Position-dependent fragmentation (NOT the e7 G3 mechanism): with
    probability rho the identity splits at a cut uniform on {1..n-1} and only
    the fragment containing the conversion (touches after the cut) is
    attributed. Touches are listed oldest -> newest. Retention of position r in
    a journey with n touches is 1 - rho (n-r)/(n-1) for r < n, and 1 for r = n
    (marginal retention depends on recency, hence on the time-decay weight)."""
    ET, ES = np.zeros(K), np.zeros(K)
    per_journey_dev = []
    for pr, touches in _journeys():
        n = len(touches)
        for r, (c, w) in enumerate(touches, start=1):
            keep = 1.0 if n == 1 or r == n else 1.0 - rho * (n - r) / (n - 1)
            ET[c] += pr * w * keep
            ES[c] += pr * w
    return ET, ES


def test_position_dependent_fragmentation_violates_marginal_condition_for_weighted_rules():
    ET, ES = _expected_masses_position_dependent(rho=0.9)
    eff = ET / ES  # effective campaign retention E[T_j]/E[S_j]
    # Retention of a touch depends on its position, so the per-touch marginal
    # condition fails; the effective retention of a campaign then depends on
    # WHERE its touches sit, i.e. on the latent mix. Shift the mix and it moves:
    base = eff.copy()
    mixes = []
    for weights in [(0.5, 0.3, 0.2), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)]:
        ET = np.zeros(2)
        ES = np.zeros(2)
        for pr, (_, touches) in zip(weights, _journeys()):
            n = len(touches)
            for r, (c, w) in enumerate(touches, start=1):
                keep = 1.0 if n == 1 or r == n else 1.0 - 0.9 * (n - r) / (n - 1)
                ET[c] += pr * w * keep
                ES[c] += pr * w
        mixes.append(ET / ES)
    mixes = np.array(mixes)
    # effective retention of campaign 0 is NOT constant over latent mixes (a
    # constant p_j would be required by the time-invariance identity)
    assert mixes[:, 0].max() - mixes[:, 0].min() > 0.05
    assert not np.allclose(base, mixes[1])


def test_g3_as_implemented_preserves_every_touch_and_is_a_clock_violation():
    """e7's `_stream_g3` keeps all touches (marginal retention is exactly 1 for
    every touch, weights untouched), so column sums of T_ij (hence the final
    point shares) are invariant. Interval widths are NOT invariant (more rows act
    like extra conversions), so a DRR of 1.00 is an empirical outcome, not
    algebra; what G3 breaks is the one-row-per-conversion arrival clock."""
    from anytime_attribution.experiments.e7_interventions import _stream_g3_legacy_duplicating as _stream_g3  # `_stream_g3` is now the corrected one-row mechanism

    rng = np.random.default_rng(0)
    n, K = 5000, 6
    campaign_idx = rng.integers(0, K, size=n)
    n_touches = rng.integers(1, 11, size=n)
    base = np.zeros((n, K))
    base[np.arange(n), campaign_idx] = n_touches
    for rho in (0.2, 0.6):
        Tj = _stream_g3(campaign_idx, n_touches, rho, rng, K)
        assert np.allclose(Tj.sum(axis=0), base.sum(axis=0))  # exact mass preservation
        assert Tj.shape[0] > n  # extra rows: clock violated (fragments counted as arrivals)


def _comonotone_q_Q(mix, p_by_campaign, j):
    """Exact q_j and Q_j (w = 1) under COMONOTONE retention: one uniform U per
    journey, R_l = 1{U <= p_{c_l}}. `mix` = [(prob, [campaigns...])]. Integrates
    over U exactly between the sorted retention probabilities."""
    num = den = Qj = 0.0
    for pr, camps in mix:
        n = len(camps)
        Qj += pr * sum(c == j for c in camps) / n
        pts = sorted({0.0, 1.0, *[p_by_campaign[c] for c in camps]})
        for u0, u1 in zip(pts[:-1], pts[1:]):
            u = 0.5 * (u0 + u1)
            kept = [c for c in camps if u <= p_by_campaign[c]]
            if kept:
                num += pr * (u1 - u0) * sum(c == j for c in kept) / len(kept)
                den += pr * (u1 - u0)
    return num / den, Qj


def test_prop8_lower_endpoint_survives_comonotone_but_upper_endpoint_does_not():
    """q_j/Q_j >= p_lo needs only marginal retention (pointwise argument), while q_j/Q_j <= Gamma_eff/s_low needs
    independence: under comonotone blocking P(M>0) = max p, not 1-prod(1-p), so s_low fails.
    """
    rng = np.random.default_rng(2)
    p_lo = 0.3
    for _ in range(300):  # lower endpoint: holds for arbitrary journeys under comonotone retention
        K = 3
        p = {c: rng.uniform(p_lo, 1.0) for c in range(K)}
        mix = []
        w = rng.dirichlet(np.ones(4))
        for pr in w:
            n = int(rng.integers(1, 6))
            mix.append((pr, list(rng.integers(0, K, size=n))))
        for j in range(K):
            q, Q = _comonotone_q_Q(mix, p, j)
            if Q > 0:
                assert q >= p_lo * Q - 1e-12
    # upper endpoint counterexample: n_min = 2, p_lo = 0.5, Gamma = 2 -> Gamma_eff = 2,
    # s_low = 1-(1-p_lo)^2 = 0.75, bound Gamma_eff/s_low = 2.667; comonotone gives 2*Gamma-1 = 3
    p = {0: 1.0, 1: 0.5}
    pi = 1e-6
    mix = [(pi, [0, 1]), (1 - pi, [1, 1])]
    q, Q = _comonotone_q_Q(mix, p, 0)
    s_low, bound = 1 - (1 - 0.5) ** 2, 2.0 / (1 - (1 - 0.5) ** 2)
    assert q / Q == pytest.approx(3.0, rel=1e-3) and q / Q > bound
    # independent retention at the same p: the bound holds
    pj, pk = 1.0, 0.5
    e_s_A, e_se_A = 1 - (1 - pj) * (1 - pk), pj * pk * 0.5 + pj * (1 - pk) * 1.0 + 0  # j kept: M_j/M = 1/2 if k kept, 1 else
    q_ind = pi * e_se_A / (pi * e_s_A + (1 - pi) * (1 - (1 - pk) ** 2))
    assert q_ind / (pi / 2) <= bound + 1e-9


# ------------------------------------------------------------------------- 
def test_out_of_range_mass_raises_not_clips():
    cs = MultiCampaignCumulativeShareCS(2, alpha=0.05, B=10.0)
    cs.update(np.array([3.0, 0.0]), 3.0)
    snap = (cs._num.t, cs._num.sum_x.copy(), cs._den.t, cs._den.sum_x)
    for bad_vec, bad_T in [([10.5, 0.0], 10.5), ([-1.0, 0.0], 3.0), ([np.nan, 0.0], 3.0), ([3.0, 0.0], 11.0)]:
        with pytest.raises(MassBoundError):
            cs.update(np.array(bad_vec), bad_T)
    # a rejected observation leaves the sequence state untouched
    assert cs._num.t == snap[0] and np.array_equal(cs._num.sum_x, snap[1])
    assert cs._den.t == snap[2] and cs._den.sum_x == snap[3]


def test_boundary_and_rounding_slack_accepted():
    cs = MultiCampaignCumulativeShareCS(2, alpha=0.05, B=10.0)
    cs.update(np.array([10.0, 0.0]), 10.0)  # exactly B is allowed
    cs.update(np.array([10.0 * (1 + 1e-12), 0.0]), 10.0)  # fp rounding slack is snapped, not raised
    assert cs._num.sum_x[0] == pytest.approx(2.0)


def test_scalar_and_vector_processes_also_raise():
    with pytest.raises(MassBoundError):
        HowardEBProcess(alpha_used=0.01).update(1.5)
    with pytest.raises(MassBoundError):
        HowardEBProcess(alpha_used=0.01).update(-0.2)
    with pytest.raises(MassBoundError):
        MultiHowardEBProcess(3, alpha_used=0.01).update(np.array([0.1, 1.2, 0.0]))
    assert issubclass(MassBoundError, ValueError)
