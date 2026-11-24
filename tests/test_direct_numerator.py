"""Direct-numerator certification.

g_jk(z) = (z_j - z_k)/sum z is degree-0 homogeneous, so a common denominator d cancels for every delta (delta is
an absolute share gap over the pool in the denominator sum). The pool matters: untracked mass must be a tracked
"other" process, or the estimand must be the tracked-pool share.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from anytime_attribution.certify import (
    Certification,
    UntrackedMassError,
    certify_pair,
    direct_gamma_star,
    direct_lower_gap,
    direct_robust_rule,
    gamma_star,
    robust_lower_gap,
    robust_rule,
)
from anytime_attribution.cs import (
    DirectNumeratorCS,
    MassBoundError,
    MultiCampaignCumulativeShareCS,
)
from anytime_attribution.identification import gamma_star_floor_saturated


def _g(z, j, k):
    z = np.asarray(z, float)
    return (z[j] - z[k]) / z.sum()


def _vertex_min(l, h, j, k):
    """Independent reference: g is linear-fractional, so over a box with l > 0
    its minimum is at a vertex; enumerate all 2^K vertices (no Lemma 1)."""
    best = np.inf
    for corner in itertools.product(*[(a, b) for a, b in zip(l, h)]):
        best = min(best, _g(corner, j, k))
    return best


def _legacy_unclipped_lower(LN, UN, LD, UD, Gamma, j, k):
    """Legacy ratio box WITHOUT the [0,1] clip: prod [LN/UD, Gamma UN/LD]."""
    return robust_lower_gap(LN / UD, UN / LD, Gamma, j, k)


def test_a_denominator_cancellation_objective_and_rule():
    rng = np.random.default_rng(0)
    for _ in range(300):
        K = int(rng.integers(2, 7))
        m = rng.uniform(0.01, 1.0, K)
        u = rng.uniform(1.0, 3.0, K)
        d = rng.uniform(0.2, 5.0)
        j, k = rng.choice(K, 2, replace=False)
        # identity on the objective: shares a=m/d give the same g as masses m
        assert _g((m / d) * u, j, k) == pytest.approx(_g(m * u, j, k), rel=1e-12, abs=1e-14)
        # scale-freedom of the box minimiser, including nonzero delta
        LN, UN = m * 0.8, m * 1.2
        for G in (1.0, 1.7):
            base = direct_lower_gap(LN, UN, G, j, k)
            assert direct_lower_gap(LN * d, UN * d, G, j, k) == pytest.approx(base, rel=1e-12, abs=1e-14)
            for delta in (0.0, 0.01, 0.2):
                a = direct_robust_rule(LN, UN, G, delta, j, k)
                b = direct_robust_rule(LN / 37.0, UN / 37.0, G, delta, j, k)
                assert a == b


def test_a_direct_equals_vertex_reference():
    rng = np.random.default_rng(1)
    for _ in range(200):
        K = int(rng.integers(2, 6))
        l = rng.uniform(0.01, 0.5, K)
        h = l * rng.uniform(1.0, 2.0, K)
        j, k = rng.choice(K, 2, replace=False)
        G = float(rng.choice([1.0, 1.3, 2.0]))
        assert direct_lower_gap(l, h, G, j, k) == pytest.approx(_vertex_min(l, G * h, j, k), rel=1e-9, abs=1e-12)


def test_a_delta_zero_reduces_to_ratio_threshold():
    rng = np.random.default_rng(2)
    for _ in range(500):
        LN = rng.uniform(0.0, 0.4, 4)
        UN = LN + rng.uniform(0.0, 0.3, 4) + 1e-3
        G = float(rng.uniform(1.0, 3.0))
        j, k = 0, 1
        cert = direct_robust_rule(LN, UN, G, 0.0, j, k)
        want = Certification.J_OVER_K if LN[j] > G * UN[k] else (Certification.K_OVER_J if LN[k] > G * UN[j] else Certification.UNRESOLVED)
        assert cert == want


def test_b_untracked_mass_delta_zero_vs_positive():
    LN = UN = np.array([0.3, 0.1])  # exact tracked masses (point boxes)
    # tracked pool: estimand is the share among the two tracked campaigns -> gap 0.5
    assert direct_lower_gap(LN, UN, 1.0, 0, 1, pool="tracked", untracked_mass=True, delta=0.3) == pytest.approx(0.5)
    # full pool with untracked mass and delta > 0 and no 'other': refuse
    with pytest.raises(UntrackedMassError):
        direct_robust_rule(LN, UN, 1.0, 0.3, 0, 1, pool="full", untracked_mass=True)
    # delta = 0: only the numerator sign matters, other mass irrelevant, no error
    assert direct_robust_rule(LN, UN, 1.0, 0.0, 0, 1, pool="full", untracked_mass=True) == Certification.J_OVER_K
    # explicit 'other' bucket (mass 0.6): full-pool gap is (0.3-0.1)/1.0 = 0.2
    lg = direct_lower_gap(LN, UN, 1.0, 0, 1, pool="full", untracked_mass=True, other=(0.6, 0.6), delta=0.3)
    assert lg == pytest.approx(0.2)
    # DROPPING the other mass would certify j > k at delta=0.3 although the true full-pool gap 0.2 <= 0.3
    assert direct_robust_rule(LN, UN, 1.0, 0.3, 0, 1, pool="tracked") == Certification.J_OVER_K  # (valid only for the tracked-pool estimand)
    assert direct_robust_rule(LN, UN, 1.0, 0.3, 0, 1, pool="full", untracked_mass=True, other=(0.6, 0.6)) == Certification.UNRESOLVED
    # no untracked mass flagged: 'full' pool needs no other bucket
    assert direct_lower_gap(LN, UN, 1.0, 0, 1, pool="full", untracked_mass=False, delta=0.3) == pytest.approx(0.5)


def test_b_other_bucket_only_weakens_positive_certificates_and_keeps_sign():
    rng = np.random.default_rng(3)
    for _ in range(300):
        K = 4
        LN = rng.uniform(0.02, 0.3, K)
        UN = LN * rng.uniform(1.0, 1.5, K)
        oth = (float(rng.uniform(0.0, 0.2)), 0.0)
        oth = (oth[0], oth[0] + float(rng.uniform(0.0, 0.2)))
        G = float(rng.choice([1.0, 1.5]))
        a = direct_lower_gap(LN, UN, G, 0, 1)
        b = direct_lower_gap(LN, UN, G, 0, 1, other=oth, pool="full", untracked_mass=True)
        if a > 0:
            assert b <= a + 1e-12  # extra denominator mass only shrinks a positive gap
        # sign (delta = 0 decision) is unaffected: positive iff LN_j > G*UN_k
        assert (a > 0) == (b > 0) == (LN[0] > G * UN[1])


def test_b_direct_cs_other_bucket_semantics():
    cs = DirectNumeratorCS(3, 0.05, 10.0, include_other=True)
    cs.update(np.array([2.0, 1.0, 0.0]), 5.0)  # other = 2
    assert cs.untracked_mass_seen and cs.other_box() is not None
    with pytest.raises(MassBoundError):
        cs.update(np.array([2.0, 1.0, 0.0]), 2.0)  # total below tracked sum
    with pytest.raises(ValueError):
        cs.update(np.array([2.0, 1.0, 0.0]))  # include_other needs T_i
    cs2 = DirectNumeratorCS(3, 0.05, 10.0, include_other=False)
    cs2.update(np.array([2.0, 1.0, 0.0]), 5.0)  # untracked mass recorded, tracked-pool estimand unaffected
    assert cs2.untracked_mass_seen and cs2.other_box() is None
    cs3 = DirectNumeratorCS(3, 0.05, 10.0)
    cs3.update(np.array([2.0, 1.0, 0.0]))  # T_i optional when no other bucket
    assert not cs3.untracked_mass_seen
    # certify_pair raises under full pool + delta>0 + untracked mass + no other bucket
    with pytest.raises(UntrackedMassError):
        certify_pair(cs2, 1.0, 0.01, 0, 1, pool="full")
    certify_pair(cs2, 1.0, 0.01, 0, 1, pool="tracked")  # fine: tracked-pool estimand
    certify_pair(cs, 1.0, 0.01, 0, 1, pool="full")  # fine: other process tracked


def test_b_zero_and_degenerate_boxes_are_explicit():
    # t = 0 style box [0, 1]^K: nothing certified, no division error
    z, o = np.zeros(3), np.ones(3)
    assert direct_robust_rule(z, o, 1.0, 0.0, 0, 1) == Certification.UNRESOLVED
    # zero upper endpoint for j or k: no certificate (-inf), never NaN
    assert direct_lower_gap(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 1.0, 0, 1) == float("-inf")
    assert direct_robust_rule(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 1.0, 0.0, 0, 1) == Certification.UNRESOLVED
    # zero lower endpoint on the leader: no certificate
    assert direct_robust_rule(np.array([0.0, 0.0]), np.array([0.5, 0.1]), 1.0, 0.0, 0, 1) == Certification.UNRESOLVED
    with pytest.raises(ValueError):
        direct_lower_gap(np.array([0.2, np.nan]), np.array([0.3, 0.3]), 1.0, 0, 1)
    with pytest.raises(ValueError):
        direct_lower_gap(np.array([0.3, 0.2]), np.array([0.2, 0.3]), 1.0, 0, 1)  # L > U
    with pytest.raises(ValueError):
        direct_lower_gap(np.array([-0.1, 0.2]), np.array([0.3, 0.3]), 1.0, 0, 1)


def test_c_budget_allocation_and_process_counts():
    K, alpha = 5, 0.05
    d = DirectNumeratorCS(K, alpha, 10.0)
    assert d.n_processes == K and d.beta == pytest.approx(alpha / K) and d.alpha_used == pytest.approx(alpha / K / 2)
    do = DirectNumeratorCS(K, alpha, 10.0, include_other=True)
    assert do.n_processes == K + 1 and do.beta == pytest.approx(alpha / (K + 1))
    legacy = MultiCampaignCumulativeShareCS(K, alpha, 10.0)
    # existing dashboard-share budget unchanged: K numerators + 1 denominator at alpha/(K+1)
    assert legacy.n_processes == K + 1 and legacy.beta == pytest.approx(alpha / (K + 1))
    assert legacy.alpha_used == pytest.approx(alpha / (K + 1) / 2)


def test_c_same_processes_serve_certificates_and_dashboard_intervals():
    rng = np.random.default_rng(4)
    K = 3
    cs = MultiCampaignCumulativeShareCS(K, 0.05, 10.0)
    for _ in range(400):
        x = np.zeros(K)
        x[rng.integers(K)] = rng.integers(1, 4)
        cs.update(x, x.sum())
    LN, UN = cs.numerator_boxes()
    # numerator boxes come from the SAME processes as the ratio box: the legacy
    # box is exactly clip(LN/UD), clip(UN/LD), so no extra budget is spent
    L, U = cs.intervals()
    LD, UD = cs._den.interval()
    assert np.allclose(L, np.clip(LN / UD, 0, 1)) and np.allclose(U, np.clip(UN / LD, 0, 1))
    # both a direct certificate and the dashboard interval are available at once
    assert certify_pair(cs, 1.0, 0.0, 0, 1) in set(Certification)
    assert certify_pair(cs, 1.0, 0.0, 0, 1, method="legacy_ratio_box") in set(Certification)
    with pytest.raises(ValueError):
        certify_pair(cs, 1.0, 0.0, 0, 1, method="nope")


def test_c_simultaneous_numerator_coverage_monte_carlo():
    """Smoke test of the union-bound claim at a LARGE alpha (so a failure would be
    visible): all K numerator running means covered at all checked t, drifting mix.
    Expected miscoverage <= alpha; fail only beyond 3 binomial SE."""
    rng = np.random.default_rng(5)
    K, N, reps, alpha, B = 3, 600, 150, 0.5, 4.0
    bad = 0
    for _ in range(reps):
        cs = DirectNumeratorCS(K, alpha, B)
        cum_mu = np.zeros(K)
        viol = False
        for t in range(1, N + 1):
            w = 0.5 * (1 + np.sin(t / 80))
            pr = np.array([0.1 + 0.3 * w, 0.1 + 0.3 * (1 - w), 0.1])
            c = rng.choice(K + 1, p=np.append(pr, 1 - pr.sum()))
            x = np.zeros(K)
            if c < K:
                x[c] = B
            cum_mu += pr * B
            cs.update(x)
            if t % 20 == 0:
                LN, UN = cs.numerator_boxes()
                m = cum_mu / t / B
                viol |= bool(np.any(m < LN - 1e-12) or np.any(m > UN + 1e-12))
        bad += viol
    se = np.sqrt(alpha * (1 - alpha) / reps)
    assert bad / reps <= alpha + 3 * se


def _random_case(rng):
    K = int(rng.integers(2, 7))
    m = rng.dirichlet(np.ones(K)) * rng.uniform(0.05, 0.9)
    d = m.sum()
    w = rng.uniform(0.0, rng.choice([0.02, 0.2, 0.6]), K) * m
    LN, UN = np.maximum(m - w, 0.0), np.minimum(m + w, 1.0)
    wd = rng.uniform(0.0, rng.choice([0.02, 0.3])) * d
    LD, UD = max(d - wd, 1e-9), min(d + wd, 1.0)
    j, k = rng.choice(K, 2, replace=False)
    return LN, UN, LD, UD, j, k


def test_d_weak_dominance_over_unclipped_legacy_and_strict_improvement():
    rng = np.random.default_rng(6)
    strict = equal = 0
    for _ in range(4000):
        LN, UN, LD, UD, j, k = _random_case(rng)
        G = float(rng.choice([1.0, 1.25, 1.5, 2.0, 3.0]))
        new = direct_lower_gap(LN, UN, G, j, k)
        old = _legacy_unclipped_lower(LN, UN, LD, UD, G, j, k) if UN[j] > 0 and UN[k] > 0 else -np.inf
        assert new >= old - 1e-12  # proven: legacy box is a superset after rescaling
        strict += new > old + 1e-12
        equal += abs(new - old) <= 1e-12
        for delta in (0.0, 0.005, 0.05):  # decision weak dominance
            if old > delta:
                assert new > delta
    assert strict > 0 and equal >= 0


def test_d_equality_when_denominator_slack_is_zero():
    rng = np.random.default_rng(7)
    for _ in range(300):
        LN, UN, _, _, j, k = _random_case(rng)
        d = 0.5
        assert _legacy_unclipped_lower(LN, UN, d, d, 1.7, j, k) == pytest.approx(direct_lower_gap(LN, UN, 1.7, j, k), rel=1e-12, abs=1e-14)


def test_d_edge_case_clipped_legacy_can_beat_direct():
    """Documented exception: the legacy rule's [0,1] clip encodes a_m <= 1 (which uses
    the denominator); the direct rule is scale-free and cannot. Weak dominance is
    therefore proved against the UNCLIPPED legacy box only."""
    rng = np.random.default_rng(8)
    found = False
    for _ in range(40000):
        LN, UN, LD, UD, j, k = _random_case(rng)
        G = float(rng.choice([1.0, 1.5]))
        if UN[j] <= 0 or UN[k] <= 0 or LD <= 0:
            continue
        L, U = np.clip(LN / UD, 0, 1), np.clip(UN / LD, 0, 1)
        clipped = robust_lower_gap(L, U, G, j, k)
        if clipped > direct_lower_gap(LN, UN, G, j, k) + 1e-9:
            assert np.any(UN / LD > 1.0)  # only when the clip binds
            found = True
            break
    assert found


def test_d_zero_numerator_endpoints_and_wide_narrow_boxes():
    LN = np.array([0.0, 0.0, 0.0]); UN = np.array([0.2, 0.1, 0.05])
    assert direct_robust_rule(LN, UN, 1.0, 0.0, 0, 1) == Certification.UNRESOLVED
    narrow_L = np.array([0.30, 0.10]); narrow_U = narrow_L * 1.001
    assert direct_robust_rule(narrow_L, narrow_U, 2.9, 0.0, 0, 1) == Certification.J_OVER_K  # ratio 3 > 2.9*1.001
    assert direct_robust_rule(narrow_L, narrow_U, 3.1, 0.0, 0, 1) == Certification.UNRESOLVED
    assert direct_robust_rule(np.array([0.1, 0.3]), np.array([0.1001, 0.3003]), 1.0, 0.0, 0, 1) == Certification.K_OVER_J


def test_e_gamma_star_matches_independent_reference():
    rng = np.random.default_rng(9)
    for _ in range(80):
        K = int(rng.integers(2, 5))
        l = rng.uniform(0.05, 0.4, K); l[0] += 0.6  # leader clearly ahead
        h = l * rng.uniform(1.0, 1.15, K)
        delta = float(rng.choice([0.0, 0.005, 0.03]))
        got = direct_gamma_star(l, h, delta, 0, 1, tol=1e-7)

        def ok(G):
            return _vertex_min(l, G * h, 0, 1) > delta

        if not ok(1.0):
            assert got is None
            continue
        a, b = 1.0, 1000.0
        if ok(b):
            ref = b
        else:
            for _ in range(200):
                mid = 0.5 * (a + b)
                a, b = (mid, b) if ok(mid) else (a, mid)
            ref = a
        assert got == pytest.approx(ref, rel=1e-5, abs=1e-5)
        if delta == 0.0:
            assert got == pytest.approx(l[0] / h[1], rel=1e-5)  # closed form L_j / U_k


def test_e_gamma_star_direction_and_floor_regime():
    rng = np.random.default_rng(10)
    up = tie = 0
    for _ in range(400):
        LN, UN, LD, UD, j, k = _random_case(rng)
        LN = np.maximum(LN, 1e-6)
        if UN[j] <= 0 or UN[k] <= 0:
            continue
        for delta in (0.0, 0.005):
            new = direct_gamma_star(LN, UN, delta, j, k, tol=1e-6)
            # unclipped legacy radius via the same bisection on the rescaled box
            leg_L, leg_U = LN / UD, UN / LD
            old = gamma_star(leg_L, leg_U, delta, j, k, tol=1e-6)
            if old is not None:
                assert new is not None and new >= old - 1e-5  # tighter certification => radius no smaller
                up += new > old + 1e-5
                tie += abs(new - old) <= 1e-5
    assert up > 0
    # Gamma > 1/p_lo regime: with p_lo = 0.8 the effective ratio never exceeds 1.25
    LN = np.array([0.40, 0.20, 0.20]); UN = np.array([0.42, 0.21, 0.21])  # ratio ~1.9
    gs = direct_gamma_star(LN, UN, 0.0, 0, 1)
    assert gs == pytest.approx(0.40 / 0.21, rel=1e-3) and gs > 1.25
    assert gamma_star_floor_saturated(gs, 0.8)
    assert direct_gamma_star(LN, UN, 0.0, 0, 1, p_lo=0.8) == 1000.0  # certified for every Gamma under the floor
    LN2 = np.array([0.25, 0.20, 0.20]); UN2 = np.array([0.26, 0.205, 0.205])  # Gamma* ~ 1.22 < 1.25
    gs2 = direct_gamma_star(LN2, UN2, 0.0, 0, 1)
    assert gs2 < 1.25 and not gamma_star_floor_saturated(gs2, 0.8)
    assert direct_gamma_star(LN2, UN2, 0.0, 0, 1, p_lo=0.8) == pytest.approx(gs2, rel=1e-3)


def test_f_legacy_rule_unchanged_and_available():
    """The legacy ratio-box functions give the same results as before."""
    L = np.array([0.30, 0.05]); U = np.array([0.35, 0.08])
    assert robust_rule(L, U, 1.0, 0.0, 0, 1) == Certification.J_OVER_K
    assert robust_rule(L, U, 5.0, 0.0, 0, 1) == Certification.UNRESOLVED
    assert gamma_star(L, U, 0.0, 0, 1) == pytest.approx(0.30 / 0.08, rel=1e-3)
