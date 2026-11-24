import numpy as np
import pytest

from anytime_attribution.baselines import bootstrap_ci, fixed_eb_ci, csranks_style_ci


def _synthetic_stream(rng, n, true_ratio, B=4.0):
    T = rng.uniform(0.5, B, size=n)
    Tj = np.clip(true_ratio * T + rng.normal(0, 0.05, size=n) * T, 0.0, T)
    return list(Tj), list(T)


def test_bootstrap_ci_covers_true_ratio_typically():
    rng = np.random.default_rng(0)
    Tj, T = _synthetic_stream(rng, 500, true_ratio=0.3)
    lo, hi = bootstrap_ci(Tj, T, alpha=0.05, B=500, rng=rng)
    assert lo < hi
    assert lo <= 0.3 <= hi


def test_fixed_eb_ci_covers_true_ratio_typically():
    rng = np.random.default_rng(1)
    Tj, T = _synthetic_stream(rng, 500, true_ratio=0.3)
    lo, hi = fixed_eb_ci(Tj, T, alpha=0.05, bound=4.0)
    assert lo < hi
    assert lo <= 0.3 <= hi


def test_csranks_style_ci_covers_true_ratio_typically():
    rng = np.random.default_rng(2)
    Tj, T = _synthetic_stream(rng, 500, true_ratio=0.3)
    lo, hi = csranks_style_ci(Tj, T, alpha=0.05, n_campaigns=10)
    assert lo < hi
    assert lo <= 0.3 <= hi


def test_csranks_style_wider_than_fixed_eb_without_slack_due_to_bonferroni():
    # csranks-style uses alpha/K (K=10) so should be wider than a
    # single-campaign-level normal CI at the same alpha
    rng = np.random.default_rng(3)
    Tj, T = _synthetic_stream(rng, 500, true_ratio=0.3)
    lo_c, hi_c = csranks_style_ci(Tj, T, alpha=0.05, n_campaigns=10)
    lo_1, hi_1 = csranks_style_ci(Tj, T, alpha=0.05, n_campaigns=1)
    assert (hi_c - lo_c) > (hi_1 - lo_1)


def test_bootstrap_ci_shrinks_with_more_data():
    rng = np.random.default_rng(4)
    Tj_small, T_small = _synthetic_stream(rng, 50, true_ratio=0.3)
    Tj_big, T_big = _synthetic_stream(rng, 2000, true_ratio=0.3)
    lo_s, hi_s = bootstrap_ci(Tj_small, T_small, alpha=0.05, B=500, rng=rng)
    lo_b, hi_b = bootstrap_ci(Tj_big, T_big, alpha=0.05, B=500, rng=rng)
    assert (hi_b - lo_b) < (hi_s - lo_s)


def test_bootstrap_ci_n1_is_degenerate_point_interval():
    """diagnosis: at n=1 there is exactly one achievable percentile
    resample (the point itself), so the CI is provably zero-width -- not a
    small-sample noise issue but an algebraic certainty. Regression guard
    for the diagnosis in e1_coverage.py::BASELINE_N_MIN, not a coverage
    claim."""
    rng = np.random.default_rng(5)
    lo, hi = bootstrap_ci([0.3], [1.0], alpha=0.05, B=500, rng=rng)
    assert lo == hi == 0.3


def test_bootstrap_ci_marginal_coverage_at_n30_is_near_nominal():
    """single-look (per-interval, marginal) bootstrap
    coverage must be ~0.95, not the 0.000 e1_coverage.py's literal-n=1
    column showed (a degenerate corner case, see BASELINE_N_MIN). Calibrated
    at n=30 (e1_coverage.py's BASELINE_N_MIN) against the E1 population
    (K=10, campaign weights linspace(1.5,0.5,10), A4 cap 10) -- average
    marginal coverage across all 10 campaigns measured 0.9115 over 400
    replications (seed=7); 600 replications at seed=0 used here for a
    stable estimate (0.909), safely inside the band."""
    from anytime_attribution.estimands import campaign_masses
    from anytime_attribution.journeys import apply_cap
    from anytime_attribution.synth import generate_synthetic_journeys, true_aggregate_share

    K = 10
    weights = np.linspace(1.5, 0.5, K)
    A_true = true_aggregate_share(weights)
    n, n_reps, alpha = 30, 600, 0.05
    rng = np.random.default_rng(0)

    covered = np.zeros(K)
    for _ in range(n_reps):
        journeys = generate_synthetic_journeys(n, weights, rng)
        journeys = [apply_cap(j, lookback_days=7.0, max_touches=10) for j in journeys]
        Tj = np.array([campaign_masses(j, K) for j in journeys])
        T = Tj.sum(axis=1)
        for j in range(K):
            lo, hi = bootstrap_ci(Tj[:, j], T, alpha, B=1000, rng=rng)
            if lo <= A_true[j] <= hi:
                covered[j] += 1

    avg_coverage = (covered / n_reps).mean()
    assert 0.90 <= avg_coverage <= 0.99, f"average marginal bootstrap coverage at n=30 was {avg_coverage}"
