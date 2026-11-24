"""E1/E3 runners: budgets, baseline identities, LP verification, dashboard-envelope exact formula."""
import importlib.util
import math

import numpy as np
import pytest

from anytime_attribution.baselines import csranks_style_ci, fixed_eb_ci
from anytime_attribution.identification import gamma_eff, prop2_interval, prop3_gap_interval


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture(scope="module")
def e1(): return _load("scripts/run_e1.py", "e1p")
@pytest.fixture(scope="module")
def e3(): return _load("scripts/run_e3.py", "e3p")


def test_e1_vectorised_baselines_equal_reference_functions(e1):
    from anytime_attribution.experiments.e1_coverage import _stream_Tj_T
    Tj, T = _stream_Tj_T(2000, np.random.default_rng(0))
    r, se, t = e1.delta_method_all_t(Tj, T)
    for h in (30, 400, 2000):
        for j in (0, 4):
            for alpha in (0.05, 0.5):
                lo, hi = fixed_eb_ci(Tj[:h, j], T[:h], alpha, 10.0)
                assert e1.fixed_hw(se, t, alpha, "fixed_eb")[h - 1, j] == pytest.approx((hi - lo) / 2, rel=1e-9)
                lo, hi = fixed_eb_ci(Tj[:h, j], T[:h], alpha / 10, 10.0)
                assert e1.fixed_hw(se, t, alpha, "fixed_eb_bonf")[h - 1, j] == pytest.approx((hi - lo) / 2, rel=1e-9)   # Bonferroni = alpha/K, K = 10
                lo, hi = csranks_style_ci(Tj[:h, j], T[:h], alpha, 10)
                assert e1.fixed_hw(se, t, alpha, "csranks_style")[h - 1, j] == pytest.approx((hi - lo) / 2, rel=1e-9)


def test_e1_budgets_documented_and_cs_matches_production(e1):
    from anytime_attribution.cs import MultiCampaignCumulativeShareCS
    K, alpha = e1.K, 0.05
    assert MultiCampaignCumulativeShareCS(K, alpha, 10.0).alpha_used == pytest.approx(alpha / (K + 1) / 2)       # cs_shared
    assert MultiCampaignCumulativeShareCS(1, alpha, 10.0).alpha_used == pytest.approx(alpha / 2 / 2)             # cs_single (K=1)


def test_e1_one_rep_shapes_and_separate_coverage_notions(e1):
    out = e1.one_rep(("stationary", 0, 0))
    for alpha in e1.ALPHAS:
        for m in ("cs_shared", "cs_single", "fixed_eb", "fixed_eb_bonf", "csranks_style"):
            assert out[alpha][m]["cov_H"].shape == (4, e1.K) and out[alpha][m]["path"].shape == (e1.N - e1.T0_PATH + 1, e1.K)
        for m in ("bootstrap", "bootstrap_bonf"):
            assert out[alpha][m]["cov_H"].shape == (4, e1.K) and "path" not in out[alpha][m]    # bootstrap evaluated at prespecified horizons only


def test_e3_lp_recovers_closed_forms_and_attaining_config_is_admissible(e3):
    rng = np.random.default_rng(0)
    for (G, plo) in ((2.0, 0.8), (1.5, 0.8), (5.0, 0.2), (2.0, 1e-6)):
        for _ in range(10):
            a = rng.dirichlet(np.ones(4))
            lo, p_lo = e3.lp_extreme(a, 1, 0, G, plo, "min", "A"); hi, p_hi = e3.lp_extreme(a, 1, 0, G, plo, "max", "A")
            slo, shi = prop2_interval(a[1], G, plo)
            assert lo == pytest.approx(slo, abs=1e-9) and hi == pytest.approx(shi, abs=1e-9)
            assert e3.admissible(p_lo, G, plo) and e3.admissible(p_hi, G, plo)
            g_lo, _ = e3.lp_extreme(a, 0, 2, G, plo, "min", "gap")
            assert g_lo == pytest.approx(prop3_gap_interval(a, G, 0, 2, plo)[0], abs=1e-9)


def test_e3_floor_binding_cell_values(e3):
    a = np.array([0.4, 0.3, 0.3])
    assert gamma_eff(2.0, 0.8) == pytest.approx(1.25)
    lo, hi = e3.lp_extreme(a, 0, 1, 2.0, 0.8, "min", "A")[0], e3.lp_extreme(a, 0, 1, 2.0, 0.8, "max", "A")[0]
    assert (round(lo, 4), round(hi, 4)) == (0.3478, 0.4545)                     # sharp floored (Gamma_eff = 1.25)
    assert prop2_interval(0.4, 2.0) == pytest.approx((0.25, 0.5714285714285715))   # conservative ratio-only: strictly larger, NOT sharp here


def test_e3_prop8_exact_formula_matches_simulation(e3):
    types = [(0.5, 1, 4), (0.5, 0, 3)]; pj, pk = 0.6, 0.3
    q, Q = e3.exact_q_Q(types, pj, pk)
    rng = np.random.default_rng(1); n = 400000
    kind = rng.random(n) < 0.5
    nj = np.where(kind, 1, 0); nk = np.where(kind, 4, 3)
    a = rng.binomial(nj, pj); b = rng.binomial(nk, pk); T = a + b
    assert abs((a[T > 0] / T[T > 0]).mean() - q) < 0.01
    assert Q == pytest.approx(0.5 * (1 / 5))
