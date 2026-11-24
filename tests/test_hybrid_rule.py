"""Hybrid lower bound max(L_direct, L_legacy_clipped).
Both are lower bounds on the SAME tracked-pool gap Delta_jk = A_j - A_k and are valid
on the single event {all K+1 processes of one MultiCampaignCumulativeShareCS cover}
(shared beta = alpha/(K+1)), so the maximum is valid with no extra budget."""
import numpy as np
import pytest

from anytime_attribution.certify import (
    Certification,
    UntrackedMassError,
    certify_pair,
    direct_gamma_star,
    direct_lower_gap,
    gamma_star,
    hybrid_gamma_star,
    hybrid_lower_gap,
    hybrid_robust_rule,
    robust_lower_gap,
)
from anytime_attribution.cs import MultiCampaignCumulativeShareCS


def _valid_event(rng):
    """Boxes that contain the truth (a valid CS event): numerators around m,
    denominator around d; ratio box = the production clipped box. Returns
    (LN, UN, L, U, m, d); the denominator box is stored in `_valid_event.last`."""
    K = int(rng.integers(2, 7))
    m = rng.dirichlet(np.ones(K)) * rng.uniform(0.05, 0.9)
    d = m.sum()
    wn = rng.choice([0.02, 0.2, 0.6, 1.5]) * rng.uniform(0, 1, K) * m
    LN, UN = np.maximum(m - wn, 0.0), np.minimum(m + wn, 1.0)
    wd = rng.choice([0.02, 0.3, 0.8]) * d
    LD, UD = max(d - wd, 1e-9), min(d + wd, 1.0)
    L, U = np.clip(LN / UD, 0, 1), np.clip(UN / LD, 0, 1)
    _valid_event.last = (LD, UD)
    return LN, UN, L, U, m, d


def _clipped(L, U, G, j, k):
    return robust_lower_gap(L, U, G, j, k) if (U[j] > 0 and U[k] > 0) else -np.inf


def test_hybrid_never_weaker_than_either_component():
    rng = np.random.default_rng(0)
    gt = 0
    for _ in range(5000):
        LN, UN, L, U, m, d = _valid_event(rng)
        j, k = rng.choice(len(m), 2, replace=False)
        G = float(rng.choice([1.0, 1.25, 1.5, 2.0, 3.0]))
        dr = direct_lower_gap(LN, UN, G, j, k)
        lg = _clipped(L, U, G, j, k)
        h = hybrid_lower_gap(LN, UN, L, U, G, j, k)
        assert h == max(dr, lg) and h >= dr and h >= lg
        gt += h > dr + 1e-12
        for delta in (0.0, 0.005, 0.05):  # decisions: hybrid certifies whenever either does
            if dr > delta or lg > delta:
                assert h > delta
    assert gt > 0  # the clipped component does sometimes strictly help


def test_hybrid_is_valid_lower_bound_of_true_tracked_pool_gap():
    """On a valid event (truth inside all boxes) and ANY retention in P(Gamma): hybrid <= true gap."""
    rng = np.random.default_rng(1)
    for _ in range(4000):
        LN, UN, L, U, m, d = _valid_event(rng)
        K = len(m)
        G = float(rng.choice([1.0, 1.25, 1.5, 2.0]))
        p = rng.uniform(1.0, G, K) ** -1  # retention vector with max/min <= G
        A = (m / p) / (m / p).sum()  # true tracked-pool shares (m = observed numerators)
        j, k = rng.choice(K, 2, replace=False)
        assert hybrid_lower_gap(LN, UN, L, U, G, j, k) <= A[j] - A[k] + 1e-12


def test_same_tracked_pool_convention_both_components():
    """Both components compute g over the K tracked coordinates only; homogeneity
    makes the legacy objective on a = m/d identical to the direct objective on m."""
    rng = np.random.default_rng(2)
    for _ in range(300):
        LN, UN, L, U, m, d = _valid_event(rng)
        j, k = rng.choice(len(m), 2, replace=False)
        u = rng.uniform(1, 2, len(m))
        g_a = ((m / d)[j] * u[j] - (m / d)[k] * u[k]) / ((m / d) * u).sum()
        g_m = (m[j] * u[j] - m[k] * u[k]) / (m * u).sum()
        assert g_a == pytest.approx(g_m, rel=1e-12, abs=1e-14)


def test_unsupported_conventions_raise_not_guess():
    LN, UN = np.array([0.3, 0.1]), np.array([0.3, 0.1])
    L, U = np.array([0.7, 0.2]), np.array([0.7, 0.2])
    with pytest.raises(NotImplementedError):  # full-pool 'other' bucket: legacy component has no other box
        hybrid_lower_gap(LN, UN, L, U, 1.0, 0, 1, other=(0.6, 0.6), pool="full", untracked_mass=True)
    with pytest.raises(UntrackedMassError):  # same guard as the direct rule (legacy also drops untracked mass)
        hybrid_lower_gap(LN, UN, L, U, 1.0, 0, 1, pool="full", untracked_mass=True, delta=0.1)
    hybrid_lower_gap(LN, UN, L, U, 1.0, 0, 1, pool="full", untracked_mass=True, delta=0.0)  # delta=0: sign only
    with pytest.raises(ValueError):  # ratio box must be the clipped production box
        hybrid_lower_gap(LN, UN, L, np.array([1.5, 0.2]), 1.0, 0, 1)


def test_p_lo_floor_applies_to_both_components():
    rng = np.random.default_rng(3)
    for _ in range(300):
        LN, UN, L, U, m, d = _valid_event(rng)
        j, k = rng.choice(len(m), 2, replace=False)
        h_big = hybrid_lower_gap(LN, UN, L, U, 5.0, j, k, p_lo=0.8)       # Gamma_eff = 1.25
        h_eff = hybrid_lower_gap(LN, UN, L, U, 1.25, j, k)
        assert h_big == pytest.approx(h_eff, rel=1e-12, abs=1e-14)


def test_zero_denominator_lower_bound_and_zero_endpoints():
    # LD <= 0: production returns ratio box [0,1]^K -> clipped component trivial; hybrid == direct
    cs = MultiCampaignCumulativeShareCS(3, 0.05, 10.0)
    LN, UN = cs.numerator_boxes(); L, U = cs.intervals()
    assert np.all(L == 0) and np.all(U == 1)
    assert hybrid_robust_rule(LN, UN, L, U, 1.0, 0.0, 0, 1) == Certification.UNRESOLVED
    z = np.zeros(2)
    assert hybrid_lower_gap(z, z, z, z, 1.0, 0, 1) == float("-inf")  # zero upper endpoints: no certificate, no NaN


def test_clipped_component_can_win_only_via_other_campaigns_in_positive_region():
    """Mathematical fact: in the positive region (clipped lower gap > 0) on a valid event, the
    clip cannot bind on the lagging campaign k: that would force h_k = Gamma >= 1 >= l_j (l_j <= a_j <= 1).
    The clipped component's advantage must come from clipping a THIRD campaign's upper endpoint."""
    rng = np.random.default_rng(4)
    pos = 0
    for _ in range(60000):
        LN, UN, L, U, m, d = _valid_event(rng)
        j, k = rng.choice(len(m), 2, replace=False)
        G = float(rng.choice([1.0, 1.25, 1.5]))
        dr = direct_lower_gap(LN, UN, G, j, k); lg = _clipped(L, U, G, j, k)
        if lg > max(dr, 0.0) + 1e-12:
            pos += 1
            LD, UD = _valid_event.last
            assert UN[k] / LD <= 1.0 + 1e-12  # the clip does NOT bind on the lagging campaign k...
            others = [x for x in range(len(m)) if x not in (j, k)]
            assert any(UN[x] / LD > 1.0 for x in others)  #... a third campaign's clip must bind
    assert pos >= 1  # the situation exists (rarely), so the hybrid is not vacuous


def test_hybrid_gamma_star_equals_max_of_component_radii():
    rng = np.random.default_rng(5)
    checked = strict = 0
    for _ in range(600):
        LN, UN, L, U, m, d = _valid_event(rng)
        j, k = rng.choice(len(m), 2, replace=False)
        delta = float(rng.choice([0.0, 0.005]))
        hs = hybrid_gamma_star(LN, UN, L, U, delta, j, k, tol=1e-7)
        gd = direct_gamma_star(LN, UN, delta, j, k, tol=1e-7)
        gl = gamma_star(L, U, delta, j, k, tol=1e-7) if (U[j] > 0 and U[k] > 0) else None
        comps = [x for x in (gd, gl) if x is not None]
        if not comps:
            assert hs is None
            continue
        assert hs == pytest.approx(max(comps), rel=1e-5, abs=1e-5)  # never below either component
        assert hs >= (gd or 0) - 1e-6 and hs >= (gl or 0) - 1e-6
        checked += 1
        strict += (gl is not None and (gd is None or gl > gd + 1e-5))
    assert checked > 100 and strict >= 0


def test_certify_pair_hybrid_runs_on_production_cs_and_matches_components():
    rng = np.random.default_rng(6)
    K = 3
    cs = MultiCampaignCumulativeShareCS(K, 0.05, 10.0)
    for _ in range(3000):
        x = np.zeros(K); x[rng.choice(K, p=[0.6, 0.3, 0.1])] = rng.integers(1, 3)
        cs.update(x, x.sum())
    for G in (1.0, 1.25):
        for (j, k) in ((0, 1), (0, 2), (1, 2)):
            h = certify_pair(cs, G, 0.005, j, k, method="hybrid")
            dd = certify_pair(cs, G, 0.005, j, k, method="direct")
            ll = certify_pair(cs, G, 0.005, j, k, method="legacy_ratio_box")
            # never weaker than either component
            for comp in (dd, ll):
                if comp != Certification.UNRESOLVED:
                    assert h == comp
    assert certify_pair(cs, 1.0, 0.005, 0, 1) == certify_pair(cs, 1.0, 0.005, 0, 1, method="direct")  # default unchanged
