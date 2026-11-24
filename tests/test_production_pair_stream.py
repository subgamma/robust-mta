"""Hybrid certification as the production path, Gamma* consistency, pairwise full-pool stream, driver call sites."""
import itertools
import os
import re

import numpy as np
import pytest

from anytime_attribution import certify as C
from anytime_attribution.certify import (Certification, UntrackedMassError, BoxSet, box_gamma_star, box_lower_gap, box_rule, certify_pair,
                                         cert_method_override, direct_lower_gap, hybrid_lower_gap, robust_lower_gap, snapshot)
from anytime_attribution.cs import DirectNumeratorCS, MultiCampaignCumulativeShareCS, boundary_override
from anytime_attribution.pair_cert import PairStream


def make_stream(K=5, n=3000, seed=0, B=10.0, untracked=0.0):
    rng = np.random.default_rng(seed)
    w = np.array([0.4, 0.25, 0.15, 0.12, 0.08])[:K]
    Tj = np.zeros((n, K))
    for i in range(n):
        for _ in range(rng.integers(1, 5)):
            Tj[i, rng.choice(K, p=w / w.sum())] += 1
    T = Tj.sum(1) + (rng.random(n) < untracked) * 2.0
    return Tj, T, B


def run_cs(Tj, T, B, alpha=0.05, boundary=None):
    cs = MultiCampaignCumulativeShareCS(Tj.shape[1], alpha, B, boundary=boundary)
    for i in range(len(T)):
        cs.update(Tj[i], T[i])
    return cs


def test_production_defaults_are_hybrid_exact():
    assert C.PRODUCTION_CERT_METHOD == "hybrid"
    Tj, T, B = make_stream()
    cs = run_cs(Tj, T, B)
    assert cs.boundary == "exact"
    b = snapshot(cs)
    for j, k in itertools.combinations(range(5), 2):
        assert certify_pair(cs, 1.25, 0.005, j, k) == certify_pair(cs, 1.25, 0.005, j, k, method="hybrid") == box_rule(b, 1.25, 0.005, j, k)
    d = DirectNumeratorCS(5, 0.05, B)           # numerator-only object has no ratio box -> direct is the default
    for i in range(len(T)):
        d.update(Tj[i], None)
    assert certify_pair(d, 1.25, 0.005, 0, 1) == certify_pair(d, 1.25, 0.005, 0, 1, method="direct")


@pytest.mark.parametrize("boundary", ["exact", "simplified"])
@pytest.mark.parametrize("Gamma,delta", [(1.0, 0.0), (1.0, 0.005), (1.5, 0.005), (2.0, 0.0)])
def test_hybrid_weakly_dominates_components_and_gamma_star_relation(boundary, Gamma, delta):
    Tj, T, B = make_stream(seed=2)
    cs = run_cs(Tj[:1500], T[:1500], B, boundary=boundary)
    b = snapshot(cs)
    for j, k in itertools.permutations(range(5), 2):
        h = box_lower_gap(b, Gamma, j, k, method="hybrid", delta=delta)
        d = direct_lower_gap(b.LN, b.UN, Gamma, j, k, delta=delta)
        lg = robust_lower_gap(b.L, b.U, Gamma, j, k) if (b.U[j] > 0 and b.U[k] > 0) else -np.inf
        assert h >= d - 1e-15 and h >= lg - 1e-15 and h == max(d, lg)
    # Gamma* relation: hybrid Gamma* = max of component Gamma*; the rule certifies just below, not just above
    for j, k in [(0, 1), (0, 3), (1, 4), (0, 4)]:
        gh = box_gamma_star(b, delta, j, k, method="hybrid", tol=1e-6)
        gl = box_gamma_star(b, delta, j, k, method="legacy_ratio_box", tol=1e-6)
        if gl is not None:
            assert gh is not None and gh >= gl - 1e-5            # hybrid radius >= legacy radius
        if gh is None:
            continue
        assert box_rule(b, gh * (1 - 1e-4), delta, j, k, method="hybrid") == Certification.J_OVER_K
        assert box_rule(b, gh * (1 + 1e-4) + 1e-4, delta, j, k, method="hybrid") != Certification.J_OVER_K


def test_tracked_pool_convention_and_untracked_guard():
    Tj, T, B = make_stream(untracked=0.3, seed=4)
    cs = run_cs(Tj[:800], T[:800], B)
    assert cs.untracked_mass_seen
    # tracked pool (production convention): untracked mass is outside the estimand -> allowed, both components share the pool
    assert certify_pair(cs, 1.0, 0.005, 0, 1, method="hybrid", pool="tracked") in set(Certification)
    # full pool with delta > 0 and untracked mass but no 'other' process -> refuse (never silently drop it)
    with pytest.raises(UntrackedMassError):
        certify_pair(cs, 1.0, 0.005, 0, 1, method="hybrid", pool="full")
    # delta = 0: the sign comparison is denominator-free, so the full-pool guard does not apply
    certify_pair(cs, 1.0, 0.0, 0, 1, method="hybrid", pool="full")


def test_gamma_above_one_over_plo_saturates():
    Tj, T, B = make_stream(seed=5)
    b = snapshot(run_cs(Tj[:2000], T[:2000], B))
    for j, k in [(0, 1), (0, 4)]:
        g_at_cap = box_lower_gap(b, 1 / 0.8, j, k, p_lo=0.8)
        g_beyond = box_lower_gap(b, 3.0, j, k, p_lo=0.8)
        assert g_beyond == pytest.approx(g_at_cap, abs=1e-12)               # Gamma_eff = min(Gamma, 1/p_lo)
        assert box_lower_gap(b, 3.0, j, k) <= g_beyond + 1e-12              # ratio-only (no floor) is weaker


def test_config_switches_are_scoped():
    Tj, T, B = make_stream(seed=6)
    b = snapshot(run_cs(Tj[:1000], T[:1000], B))
    with cert_method_override("legacy_ratio_box"):
        assert C.active_cert_method() == "legacy_ratio_box"
        assert box_lower_gap(b, 1.3, 0, 1) == robust_lower_gap(b.L, b.U, 1.3, 0, 1)
    assert C.active_cert_method() == "hybrid"
    with boundary_override("simplified"):
        bs = snapshot(run_cs(Tj[:1000], T[:1000], B))
    assert np.all(bs.UN >= b.UN) and np.all(bs.LN <= b.LN)
    assert np.all(bs.UN - bs.LN > b.UN - b.LN - 1e-15)


def test_pair_stream_full_pool_uses_other_bucket_and_legacy_comparator():
    rng = np.random.default_rng(7)
    n, B = 4000, 10.0
    Tj = rng.integers(0, 3, n).astype(float); Tk = rng.integers(0, 2, n).astype(float)
    T = Tj + Tk + rng.integers(0, 5, n)        # untracked mass (other campaigns)
    ps = PairStream(0.05, B, pool="full", method="hybrid")
    assert ps.kind == "direct_other" and ps.cs.n_processes == 3 and ps.cs.beta == pytest.approx(0.05 / 3)
    for i in range(n):
        ps.update(Tj[i], Tk[i], T[i])
    s = ps.snap()
    LN, UN = ps.cs.numerator_boxes()
    assert ps.rule(s, 1.2, 0.005) == C.direct_robust_rule(LN, UN, 1.2, 0.005, 0, 1, other=ps.cs.other_box(), pool="full", untracked_mass=True)
    # legacy comparator = ratio-box rule on the same-budget MultiCampaign object
    pl = PairStream(0.05, B, pool="full", method="legacy_ratio_box")
    for i in range(n):
        pl.update(Tj[i], Tk[i], T[i])
    L, U = pl.cs.intervals()
    assert pl.rule(pl.snap(), 1.2, 0.005) == C.robust_rule(L, U, 1.2, 0.005, 0, 1)
    # tracked-pool hybrid stream
    pt = PairStream(0.05, B, pool="tracked", method="hybrid")
    assert pt.kind == "hybrid"


def test_no_production_driver_calls_legacy_rule_directly():
    """Legacy robust_rule / gamma_star may appear only at the allowed comparator call sites."""
    allowed = {"e4_confident_reversal.py": 1,       # population point-box (deterministic) certification, equal under both rules
               "pair_cert.py": 2}                   # explicit legacy comparator
    root = "anytime_attribution/experiments"
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".py") or fn.startswith("e10"):
            continue
        src = open(os.path.join(root, fn)).read()
        code = "\n".join(l for l in src.split("\n") if not l.strip().startswith("#"))
        # strip docstrings/strings roughly: count real call sites
        n = len(re.findall(r"(?<![\w.])(robust_rule|gamma_star)\(", re.sub(r'""".*?"""', "", code, flags=re.S)))
        assert n <= allowed.get(fn, 0), f"{fn}: {n} direct legacy-rule call sites"


def test_empirical_models_are_not_routed_through_hybrid():
    import json
    p = "results/e5/e5_results.json"
    if not os.path.exists(p):
        pytest.skip("results/e5 not generated")
    r = json.load(open(p))
    assert r["cert_method_empirical"].startswith("N/A") and r["boundary"] == "exact"
    assert r["cert_method_guaranteed"] == "hybrid"
