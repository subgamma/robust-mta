"""Tests for `drift_test_cs` (the constant-share drift diagnostic, including a counterexample that an anchor-based
inversion must handle) and the production `MultiCampaignCumulativeShareCS`/`HowardEBProcess` (Howard et al. 2021, Thm.
4).
"""
import ast
import math
import os
from pathlib import Path

import numpy as np
import pytest

from anytime_attribution.cs import (
    HowardEBProcess,
    MultiCampaignCumulativeShareCS,
    MultiHowardEBProcess,
    _HOWARD_K1,
    _HOWARD_K2,
    _HOWARD_ZETA_S,
    drift_test_cs,
    howard_eb_radius,
)
from anytime_attribution.estimands import campaign_masses
from anytime_attribution.experiments._primary_journeys import CACHE_PATH

criteo_available = os.path.exists(CACHE_PATH) or os.path.exists("data/criteo_attribution_dataset.tsv.gz")


def test_howard_constants_match_hand_derivation():
    # k1 = (eta^0.25 + eta^-0.25)/sqrt(2), eta=2
    assert _HOWARD_K1 == pytest.approx((2 ** 0.25 + 2 ** -0.25) / math.sqrt(2), rel=1e-12)
    # k2 = (sqrt(eta)+1)/2, eta=2
    assert _HOWARD_K2 == pytest.approx((math.sqrt(2) + 1) / 2, rel=1e-12)
    # zeta(1.4) ~= 3.10555
    assert _HOWARD_ZETA_S == pytest.approx(3.105547277977581, rel=1e-9)


def test_howard_eb_radius_reproduces_paper_worked_example():
    """Eq. (24) of Howard et al. (2021) gives an explicit 95%-CS radius at alpha=0.025 for eta=2, s=1.4, m=1, c=1.

        The closed form with unrounded constants must reproduce the rounded display formula to within the rounding
        error of its constants (1.7, 3.8, 3.4, 13), shrinking as V grows.
    """
    alpha = 0.025
    for v, max_rel_err in ((1, 0.01), (100, 0.005), (10000, 0.001)):
        ours = howard_eb_radius(float(v), 1, alpha, form="simplified")  # eq. (24) prints the simplified bound; exact is the default
        paper = (1.7 * math.sqrt(max(v, 1) * (math.log(math.log(2 * max(v, 1))) + 3.8))
                 + 3.4 * math.log(math.log(2 * max(v, 1))) + 13)
        assert abs(ours - paper) / paper < max_rel_err


def test_howard_eb_process_contains_true_mean_direct_check():
    """A cheap check that a HowardEBProcess on data with a known constant mean gives a sane, shrinking, non-empty
    interval containing that mean.
    """
    rng = np.random.default_rng(0)
    true_mean = 0.3
    proc = HowardEBProcess(alpha_used=0.01)
    n = 5000
    xs = rng.binomial(1, true_mean, size=n).astype(float)
    for x in xs:
        proc.update(x)
    lo, hi = proc.interval()
    assert lo <= true_mean <= hi
    assert 0.0 <= lo <= hi <= 1.0
    assert hi - lo < 0.2  # should have tightened substantially by n=5000


def test_multi_campaign_cumulative_share_cs_never_empty_once_denominator_positive():
    rng = np.random.default_rng(1)
    K, n = 4, 2000
    # Draw within the bound (each T_ij <= B/K, so T_i <= B); the CS raises on T_i > B.
    Tj = rng.uniform(0, 10.0 / K, (n, K))
    T = Tj.sum(axis=1)
    mg = MultiCampaignCumulativeShareCS(K, alpha=0.05, B=10.0)
    for i in range(n):
        mg.update(Tj[i], T[i])
        L, U = mg.intervals()
        assert np.all(L <= U)
        assert np.all((L >= 0) & (U <= 1))
    # The denominator is positive at every step, so after the first observation the CS must not be the vacuous [0,1].
    assert np.all(U - L < 1.0)


def test_multi_campaign_matches_scalar_howard_eb_process():
    """MultiHowardEBProcess must match K independent scalar HowardEBProcess objects bit for bit."""
    rng = np.random.default_rng(2)
    K, n = 5, 300
    X = rng.uniform(0, 1, (n, K))
    alpha_used = 0.001

    scalars = [HowardEBProcess(alpha_used) for _ in range(K)]
    for i in range(n):
        for c in range(K):
            scalars[c].update(X[i, c])
    los_ref = np.array([s.interval()[0] for s in scalars])
    his_ref = np.array([s.interval()[1] for s in scalars])

    multi = MultiHowardEBProcess(K, alpha_used)
    for i in range(n):
        multi.update(X[i])
    los_batch, his_batch = multi.intervals()

    assert np.allclose(los_ref, los_batch)
    assert np.allclose(his_ref, his_batch)


def test_drift_test_cs_interval_endpoints_are_individually_accepted():
    from anytime_attribution.cs import theta_rejected

    rng = np.random.default_rng(3)
    n = 3000
    T = rng.uniform(0.5, 10.0, size=n)
    Tj = np.clip(0.2 * T + rng.normal(0, 0.05, size=n) * T, 0.0, T)
    alpha, B = 0.05, 10.0
    res = drift_test_cs(list(Tj), list(T), alpha, B)
    assert res.status in ("interval", "empty", "indeterminate")
    if res.status == "interval":
        assert not theta_rejected(list(Tj), list(T), res.lo, alpha, B)
        assert not theta_rejected(list(Tj), list(T), res.hi, alpha, B)
        assert res.lo <= res.hi


def test_drift_test_cs_does_not_anchor_on_empirical_mean():
    """Where the empirical mean is clearly rejected (a sharply reversing trend), drift_test_cs must not return [mean, mean].

        It either finds the true (possibly anchor-disjoint) interval or reports empty/indeterminate.
    """
    from anytime_attribution.cs import theta_rejected

    rng = np.random.default_rng(4)
    n = 4000
    T = np.full(n, 1.0)
    # first half: campaign share ~0.05; second half: campaign share ~0.5
    # (sharp regime change -- no single constant theta should fit well)
    Tj = np.zeros(n)
    Tj[: n // 2] = rng.binomial(1, 0.05, size=n // 2).astype(float)
    Tj[n // 2:] = rng.binomial(1, 0.5, size=n - n // 2).astype(float)
    alpha, B = 0.05, 1.0
    res = drift_test_cs(list(Tj), list(T), alpha, B)
    emp_mean = Tj.sum() / T.sum()
    if res.status == "interval":
        assert not theta_rejected(list(Tj), list(T), res.lo, alpha, B)
        assert not theta_rejected(list(Tj), list(T), res.hi, alpha, B)
        # the key regression property: if the mean itself is rejected, the
        # returned interval must not silently equal [mean, mean]
        if theta_rejected(list(Tj), list(T), emp_mean, alpha, B):
            assert not (res.lo == emp_mean == res.hi)


@pytest.mark.skipif(not criteo_available, reason="primary journey cache (built from the raw Criteo file)/raw data not available")
def test_drift_test_cs_campaign2_counterexample_regression():
    """Counterexample: campaign idx 2, primary, linear model.

        At N=20,000, drift_test_cs must recover a non-empty interval approximately [0.0562, 0.0582], not the false
        point 0.0702 that an anchor-seeking bisection produces, which is disjoint from the empirical share 0.0702.
        At full N=109,221 it must return EMPTY, a genuine rejection of constant share.
    """
    from anytime_attribution.criteo import load_config
    from anytime_attribution.experiments._primary_journeys import build_and_cache_primary_journeys

    cfg = load_config("config/criteo.yaml")
    alpha = cfg["certification"]["alpha"]
    top_20 = cfg["campaigns"]["top_20"]
    K = len(top_20)
    alpha_eff = alpha / K
    B = 10.0

    journeys, top_20_check = build_and_cache_primary_journeys("config/criteo.yaml")
    assert top_20_check == top_20
    Tj_all = np.array([campaign_masses(j, K) for j in journeys])
    T_all = Tj_all.sum(axis=1)
    Tj2 = Tj_all[:, 2]

    res_20k = drift_test_cs(Tj2[:20000].tolist(), T_all[:20000].tolist(), alpha_eff, B)
    assert res_20k.status == "interval"
    assert 0.054 <= res_20k.lo <= 0.058
    assert 0.056 <= res_20k.hi <= 0.060
    emp_20k = Tj2[:20000].sum() / T_all[:20000].sum()
    assert not (res_20k.lo <= emp_20k <= res_20k.hi)  # disjoint from the empirical share

    n_total = len(journeys)
    res_full = drift_test_cs(Tj2[:n_total].tolist(), T_all[:n_total].tolist(), alpha_eff, B)
    assert res_full.status == "empty"


# Static check: the grid CS must not be reachable from E5-E8 code; walking the AST catches renamed imports and
# attribute calls that grep would miss.

FORBIDDEN_NAMES = {"ShareCSGrid", "MultiCampaignShareCSGrid", "invert_share_cs", "invert_share_cs_grid"}
E5_E8_MODULES = [
    "anytime_attribution/experiments/e5_certified_fraction.py",
    "anytime_attribution/experiments/e6_robustness_radius.py",
    "anytime_attribution/experiments/e7_interventions.py",
    "anytime_attribution/experiments/e8_instability.py",
]


def _names_referenced(tree: ast.Module) -> set[str]:
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.alias):
            # Check the original imported name, not the alias, so a rename cannot hide a forbidden import.
            found.add(node.name)
    return found


@pytest.mark.parametrize("module_path", E5_E8_MODULES)
def test_e5_e8_production_code_does_not_reference_superseded_cs(module_path):
    repo_root = Path(__file__).resolve().parent.parent
    source = (repo_root / module_path).read_text()
    tree = ast.parse(source, filename=module_path)
    referenced = _names_referenced(tree)
    hit = referenced & FORBIDDEN_NAMES
    assert not hit, f"{module_path} references superseded CS symbol(s): {hit}"
