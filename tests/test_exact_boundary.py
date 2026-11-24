"""The exact Howard et al. (2021) stitched boundary (Thm 1, eq. (8), stitching bound of eq. (43)) is the production default.

Equation numbers follow the published version (Ann. Statist. 49(2), 10.1214/20-AOS1991); the arXiv:1810.08240 numbering differs.
"""
import math

import numpy as np
import pytest
from scipy.special import zeta

from anytime_attribution import cs as C
from anytime_attribution.cs import (HowardEBProcess, MassBoundError, MultiCampaignCumulativeShareCS,
                                    boundary_override, howard_eb_radius)
from anytime_attribution.cs_vectorized import cumulative_share_boxes, howard_radius_vec

ETA, S, M = 2.0, 1.4, 1.0


def indep_exact_S(v, alpha):
    """Written directly from eq. (8)/(10): l(v) = log(l0 h(log_eta(v/m))/alpha), h(k)=zeta(s)(k+1)^s,
    l0 = 1 (the polynomial stitching of eq. (10)); k1,k2 as in (8); c = b-a = 1."""
    v = max(v, M)
    k = math.log(v / M) / math.log(ETA)
    ell = math.log(zeta(S) * (k + 1.0) ** S / alpha)
    k1 = (ETA ** 0.25 + ETA ** -0.25) / math.sqrt(2.0)
    k2 = (math.sqrt(ETA) + 1.0) / 2.0
    return math.sqrt(k1 ** 2 * v * ell + k2 ** 2 * ell ** 2) + k2 * ell, k1 * math.sqrt(v * ell) + 2 * k2 * ell


@pytest.mark.parametrize("alpha", [0.5 / 11 / 2, 0.05 / 11 / 2, 0.05 / 4, 0.025, 0.00225])
def test_matches_independent_formula(alpha):
    for v in [0.0, 1e-12, 0.3, 1.0, 1.0001, 2.0, 7.5, 100.0, 1e4, 1e7]:
        ex, sim = indep_exact_S(v, alpha)
        assert howard_eb_radius(v, 10, alpha) * 10 == pytest.approx(ex, rel=1e-12)
        assert howard_eb_radius(v, 10, alpha, form="simplified") * 10 == pytest.approx(sim, rel=1e-12)


def test_default_is_exact_and_dominance_on_grid():
    assert C.BOUNDARY_DEFAULT == "exact"
    for alpha in (0.5 / 11 / 2, 0.05 / 11 / 2, 0.01, 0.1):
        for v in np.concatenate([[0.0], np.logspace(-6, 8, 400)]):
            e = howard_eb_radius(float(v), 3, alpha)
            s = howard_eb_radius(float(v), 3, alpha, form="simplified")
            assert 0 < e <= s + 1e-15
    # strict inequality whenever l > 0 and v > 0 (sqrt(a^2+b^2) < a+b)
    assert howard_eb_radius(50.0, 3, 0.01) < howard_eb_radius(50.0, 3, 0.01, form="simplified")


def test_v_vee_m_and_zero_variance():
    a = 0.01
    for v in (0.0, 0.5, 0.999999, 1.0):
        assert howard_eb_radius(v, 5, a) == howard_eb_radius(1.0, 5, a)      # v v m with m = 1
    assert math.isfinite(howard_eb_radius(0.0, 1, a)) and howard_eb_radius(0.0, 1, a) > 0
    assert howard_eb_radius(2.0, 5, a) > howard_eb_radius(1.0, 5, a)         # strictly increasing for v >= m
    assert howard_eb_radius(10.0, 0, a) == 1.0                                # t = 0 convention unchanged


def test_ell_positive_on_supported_alpha_range():
    # sqrt(...) requires l(v) >= 0; every alpha_used the repo uses (alpha<=0.5, K>=1) keeps l(v) > 0 at v = m
    for a in (0.5 / 2 / 2, 0.5 / 11 / 2, 0.05 / 21 / 2):
        assert C._howard_eb_L(1.0, a) > 0


def test_error_budget_allocation_unchanged():
    K, alpha = 10, 0.05
    cs = MultiCampaignCumulativeShareCS(K, alpha, 10.0)
    assert cs.beta == alpha / (K + 1) and cs.alpha_used == cs.beta / 2 and cs.boundary == "exact"
    assert cs.n_processes == K + 1
    with boundary_override("simplified"):
        cs2 = MultiCampaignCumulativeShareCS(K, alpha, 10.0)
    assert cs2.boundary == "simplified" and cs2.alpha_used == cs.alpha_used
    assert MultiCampaignCumulativeShareCS(K, alpha, 10.0).boundary == "exact"   # override is scoped


@pytest.mark.parametrize("form", ["exact", "simplified"])
def test_cs_py_equals_vectorized_stepwise(form):
    rng = np.random.default_rng(3)
    K, N, B, alpha = 5, 400, 10.0, 0.05
    Tj = np.zeros((N, K)); T = np.zeros(N)
    for i in range(N):
        n = rng.integers(0, 6)
        for _ in range(n):
            Tj[i, rng.integers(0, K)] += 1.0
        T[i] = Tj[i].sum() + (rng.random() < 0.2) * rng.integers(0, 3)
    cs = MultiCampaignCumulativeShareCS(K, alpha, B, boundary=form)
    vec = cumulative_share_boxes(Tj, T, B, alpha / (K + 1) / 2, form=form)
    for i in range(N):
        cs.update(Tj[i], T[i])
        LN, UN = cs.numerator_boxes(); L, U = cs.intervals()
        assert np.allclose(LN, vec["LN"][i], atol=1e-12) and np.allclose(UN, vec["UN"][i], atol=1e-12)
        assert np.allclose(L, vec["L"][i], atol=1e-12) and np.allclose(U, vec["U"][i], atol=1e-12)


def test_exact_boxes_nested_in_simplified_boxes():
    rng = np.random.default_rng(5)
    N, K, B = 600, 4, 10.0
    Tj = rng.integers(0, 3, size=(N, K)).astype(float); T = Tj.sum(1)
    ex = cumulative_share_boxes(Tj, T, B, 0.05 / 5 / 2, form="exact")
    si = cumulative_share_boxes(Tj, T, B, 0.05 / 5 / 2, form="simplified")
    assert np.all(ex["LN"] >= si["LN"] - 1e-15) and np.all(ex["UN"] <= si["UN"] + 1e-15)
    assert np.all(ex["L"] >= si["L"] - 1e-15) and np.all(ex["U"] <= si["U"] + 1e-15)


def test_time_indexing_and_bounded_mass_unchanged():
    p = HowardEBProcess(0.01)
    assert p.interval() == (0.0, 1.0)                  # t = 0
    p.update(0.3)
    assert p.t == 1 and p._forecast == pytest.approx(0.3)
    with pytest.raises(MassBoundError):
        MultiCampaignCumulativeShareCS(2, 0.05, 10.0).update(np.array([10.5, 0.0]), 10.5)
    with pytest.raises(ValueError):
        cumulative_share_boxes(np.array([[10.5]]), np.array([10.5]), 10.0, 0.001)


@pytest.mark.parametrize("drift", [False, True])
def test_exact_boundary_pathwise_validity_simulation(drift):
    """Predictable-mean validity (Thm 4): every-t two-sided coverage of the cumulative-mean target
    with conditional means that drift. 400 paths, per-process two-sided level 0.2 (alpha_used 0.1):
    far more demanding than production levels; the exact boundary must not exceed the nominal
    miss rate (it is rigorous), and it is checked at EVERY t."""
    rng = np.random.default_rng(11 if drift else 7)
    N, R, a_used = 3000, 400, 0.1
    t = np.arange(1, N + 1)
    miss = 0
    for _ in range(R):
        p = 0.3 + (0.25 * np.sin(np.arange(N) / 400.0) if drift else np.zeros(N))
        x = (rng.random(N) < p).astype(float)
        target = np.cumsum(p) / t
        mean = np.cumsum(x) / t
        fore = np.concatenate([[0.5], mean[:-1]])
        V = np.cumsum((x - fore) ** 2)
        r = howard_radius_vec(V, t.astype(float), a_used, form="exact")
        miss += bool(np.any(np.abs(mean - target) > r))
    assert miss / R <= 2 * a_used + 0.05


def test_override_reaches_vectorized_path_and_forms_differ():
    rng = np.random.default_rng(1)
    Tj = rng.integers(0, 3, size=(500, 3)).astype(float); T = Tj.sum(1)
    base = cumulative_share_boxes(Tj, T, 10.0, 0.002)
    with boundary_override("simplified"):
        a = cumulative_share_boxes(Tj, T, 10.0, 0.002)
        cs = MultiCampaignCumulativeShareCS(3, 0.05, 10.0)
    assert cs.boundary == "simplified"
    assert np.all(a["UN"][50:] > base["UN"][50:])      # simplified strictly wider after the early clip region
    assert np.allclose(base["UN"], cumulative_share_boxes(Tj, T, 10.0, 0.002, form="exact")["UN"])
    assert np.allclose(a["UN"], cumulative_share_boxes(Tj, T, 10.0, 0.002, form="simplified")["UN"])
