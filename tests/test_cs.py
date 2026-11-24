"""Confidence-sequence unit tests (i)-(iii) from the confidence-sequence design decisions:
(i) 2,000 replications x 1,000 monitoring points, empirical simultaneous
    coverage >= 0.95 at alpha=0.05, for both boundaries;
(ii) peeking 1,000 times does not change the coverage;
(iii) test inversion returns an interval; the statistic is monotone in theta
     on a grid.
"""
import numpy as np
import pytest

from anytime_attribution.cs import (
    covers_throughout,
    coverage_at_looks,
    ConfidenceSequence,
    theta_rejected,
    invert_share_cs,
    invert_share_cs_grid,
    ShareCSGrid,
    MultiCampaignShareCSGrid,
)

ALPHA = 0.05
N_REPS = 2000
N_POINTS = 1000
BOUNDARIES = ["wsr", "howard"]


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_i_simultaneous_coverage(boundary):
    rng = np.random.default_rng(12345)
    mu = 0.3
    covered = 0
    for _ in range(N_REPS):
        xs = (rng.random(N_POINTS) < mu).astype(float)  # i.i.d. Bernoulli(mu) in [0,1]
        if covers_throughout(xs, mu, ALPHA, boundary=boundary):
            covered += 1
    coverage = covered / N_REPS
    assert coverage >= 0.95 - 0.02, f"{boundary}: coverage={coverage}"


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_ii_peeking_1000_times_does_not_change_coverage(boundary):
    rng = np.random.default_rng(999)
    mu = 0.3
    look_points = [1, 10, 100, 1000]
    n_reps = 800
    covered_at = {L: 0 for L in look_points}
    for _ in range(n_reps):
        xs = (rng.random(N_POINTS) < mu).astype(float)
        result = coverage_at_looks(xs, mu, ALPHA, look_points, boundary=boundary)
        for L in look_points:
            if result[L]:
                covered_at[L] += 1
    coverages = {L: covered_at[L] / n_reps for L in look_points}
    for L, cov in coverages.items():
        assert cov >= 0.95 - 0.03, f"{boundary} look={L} coverage={cov}"
    # peeking at 1000 intermediate points instead of just once should not
    # meaningfully erode coverage relative to a single early look (sticky
    # rejection makes the anytime and fixed-time verdicts coincide)
    assert coverages[1000] >= coverages[1] - 0.05


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_iii_theta_inversion_is_monotone_and_interval_valued(boundary):
    rng = np.random.default_rng(7)
    true_a = 0.35
    B = 5.0
    n = 300
    Tj_hist, T_hist = [], []
    for _ in range(n):
        T_i = rng.uniform(0.5, B)
        Tj_i = true_a * T_i + rng.normal(0, 0.05 * B)
        Tj_i = min(max(Tj_i, 0.0), T_i)
        Tj_hist.append(Tj_i)
        T_hist.append(T_i)

    grid = np.linspace(0.01, 0.99, 50)
    rejects = [theta_rejected(Tj_hist, T_hist, th, alpha=0.05, B=B, boundary=boundary) for th in grid]

    anchor = sum(Tj_hist) / sum(T_hist)
    anchor_idx = int(np.argmin(np.abs(grid - anchor)))

    def _single_flip(seq):
        seen_reject = False
        for r in seq:
            if r:
                seen_reject = True
            elif seen_reject:
                return False
        return True

    left = rejects[: anchor_idx + 1][::-1]  # walking away from anchor towards 0
    right = rejects[anchor_idx:]  # walking away from anchor towards 1
    assert _single_flip(left), f"{boundary}: non-monotone rejection walking down from anchor"
    assert _single_flip(right), f"{boundary}: non-monotone rejection walking up from anchor"

    lo, hi = invert_share_cs(Tj_hist, T_hist, alpha=0.05, B=B, boundary=boundary)
    assert 0.0 <= lo <= hi <= 1.0


@pytest.mark.parametrize("boundary", BOUNDARIES)
def test_confidence_sequence_interval_matches_manual_bisection(boundary):
    rng = np.random.default_rng(3)
    mu = 0.4
    cs = ConfidenceSequence(alpha=0.05, lo=0.0, hi=1.0, boundary=boundary)
    xs = (rng.random(500) < mu).astype(float)
    for x in xs:
        cs.update(x)
    lo, hi = cs.interval()
    assert 0.0 <= lo <= hi <= 1.0
    assert lo <= mu <= hi  # single realisation at alpha=0.05, expected to cover


def test_unknown_boundary_rejected():
    with pytest.raises(ValueError):
        ConfidenceSequence(alpha=0.05, boundary="not-a-boundary")


def test_grid_cs_agrees_with_bisection_on_100_random_streams():
    """the grid-based WSR share CS (`ShareCSGrid` /
    `invert_share_cs_grid`) and the bisection reference (`invert_share_cs`)
    are the same test evaluated two different ways -- they must agree up to
    the grid's resolution (spacing 1/(grid_size-1)) plus the bisection's own
    `tol`, on both endpoints, across a range of random streams and lengths."""
    B = 10.0
    alpha = 0.05
    grid_size = 4001
    tol = 1e-4
    slack = 1.0 / (grid_size - 1) + tol + 1e-6

    rng = np.random.default_rng(0)
    for trial in range(100):
        n = int(rng.integers(5, 300))
        true_share = rng.uniform(0.05, 0.95)
        T = rng.uniform(0.5, B, size=n)
        Tj = np.clip(true_share * T + rng.normal(0, 0.1, size=n) * T, 0.0, T)

        lo_b, hi_b = invert_share_cs(list(Tj), list(T), alpha, B, tol=tol)
        lo_g, hi_g = invert_share_cs_grid(list(Tj), list(T), alpha, B, grid_size=grid_size)

        assert abs(lo_b - lo_g) <= slack, f"trial {trial}: lo mismatch {lo_b} vs {lo_g} (n={n})"
        assert abs(hi_b - hi_g) <= slack, f"trial {trial}: hi mismatch {hi_b} vs {hi_g} (n={n})"


def test_share_cs_grid_incremental_matches_non_incremental():
    """`ShareCSGrid.update`/`.interval` called incrementally must match
    `invert_share_cs_grid` called once on the full history (the whole point
    of the incremental API is that it's the SAME computation, just able to
    report the interval at intermediate points without replaying)."""
    rng = np.random.default_rng(1)
    n = 200
    T = rng.uniform(0.5, 10.0, size=n)
    Tj = np.clip(0.3 * T + rng.normal(0, 0.1, size=n) * T, 0.0, T)

    grid = ShareCSGrid(alpha=0.05, B=10.0, grid_size=2001)
    for tj, t in zip(Tj, T):
        grid.update(tj, t)
    lo_incremental, hi_incremental = grid.interval()

    lo_batch, hi_batch = invert_share_cs_grid(list(Tj), list(T), alpha=0.05, B=10.0, grid_size=2001)
    assert lo_incremental == lo_batch
    assert hi_incremental == hi_batch


def test_multi_campaign_grid_matches_per_campaign_grid():
    """performance: MultiCampaignShareCSGrid batches K campaigns'
    updates into one vectorized call; must give BIT-IDENTICAL results to K
    independent ShareCSGrid's (same deterministic recursion, just batched)."""
    rng = np.random.default_rng(2)
    K, n = 5, 300
    # Draw within the bound (each T_ij <= B/K, so T_i <= B); the grids raise on T_i > B.
    Tj = rng.uniform(0, 10.0 / K, (n, K))
    T = Tj.sum(axis=1)
    alpha, B, grid_size = 0.05 / (2 * K), 10.0, 101

    grids = [ShareCSGrid(alpha, B, grid_size=grid_size) for _ in range(K)]
    for i in range(n):
        for c in range(K):
            grids[c].update(Tj[i, c], T[i])
    los_ref = np.array([g.interval()[0] for g in grids])
    his_ref = np.array([g.interval()[1] for g in grids])

    mg = MultiCampaignShareCSGrid(K, alpha, B, grid_size=grid_size)
    for i in range(n):
        mg.update(Tj[i], T[i])
    los_batch, his_batch = mg.intervals()

    assert np.array_equal(los_ref, los_batch)
    assert np.array_equal(his_ref, his_batch)
