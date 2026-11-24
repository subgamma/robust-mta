"""Sequential index regression: the clock must be the arrival (population) clock, not the survivor clock.

The index i and filtration F_{i-1} must count every conversion, with a thinned-away one contributing zero mass
at its own position, not renumber the surviving subsequence. The discriminator is the thinning identity
E[T_j] = p_j E[S_j]. The ratio a_j = sum T_j / sum T cannot discriminate: dropping T=0 rows leaves it unchanged.

K=2, single-touch journeys of weight 1; campaign 1 w.p. 0.5, campaign 2 w.p. 0.5; retention p_1=1.0, p_2=0.5
(Gamma=2).
Population clock: E[S_1] = E[S_2] = 0.5, E[T_1] = 0.5, E[T_2] = 0.25, so E[T_j] = p_j E[S_j] holds for both j.
Survivor clock (condition on T>0): E[T_2 | surv] = 1/3 but p_2 E[S_2 | surv] = 1/6, so the identity fails for
campaign 2 by a factor of 2. This is a fixed bias, not sampling noise, since conditioning on T>0 is informative
about S when p_1 != p_2.
"""
from __future__ import annotations

import numpy as np
import pytest

from anytime_attribution.journeys import Journey, Touch, thin
from anytime_attribution.cs import MultiCampaignCumulativeShareCS
from anytime_attribution.identification import prop3_gap_interval

SEED = 20260917
N = 2_000_000
P1, P2 = 1.0, 0.5
GAMMA = P1 / P2
PI = 0.5
B_BOUND = 1.0
ALPHA = 0.05
TOL = 0.01


def _make_population(n: int, rng: np.random.Generator):
    """n single-touch latent journeys; campaign 1 w.p. PI, campaign 2 w.p. 1-PI.

        Returns (journeys, latent_labels, retained_mask). The mask is returned directly, rather than re-derived
        via `thin`, so both clocks agree on which journeys survived.
    """
    labels = np.where(rng.random(n) < PI, 1, 2)
    p_of = np.where(labels == 1, P1, P2)
    retained = rng.random(n) < p_of
    journeys = [
        Journey(touches=(Touch(campaign=int(c), timestamp=0.0, weight=1.0),), conversion_time=1.0)
        for c in labels
    ]
    return journeys, labels, retained


def _S_T(labels: np.ndarray, retained: np.ndarray):
    S1 = (labels == 1).astype(float)
    S2 = (labels == 2).astype(float)
    T1 = S1 * retained
    T2 = S2 * retained
    return S1, S2, T1, T2


@pytest.fixture(scope="module")
def population():
    rng = np.random.default_rng(SEED)
    return _make_population(N, rng)


def test_population_clock_thinning_identity_holds(population):
    """Population clock: every journey is counted (thinned-away ones contribute 0), so E[T_j] = p_j E[S_j] holds for both
    campaigns.
    """
    labels, retained = population[1], population[2]
    S1, S2, T1, T2 = _S_T(labels, retained)

    ES1, ES2 = S1.mean(), S2.mean()
    ET1, ET2 = T1.mean(), T2.mean()

    assert ES1 == pytest.approx(0.5, abs=TOL)
    assert ES2 == pytest.approx(0.5, abs=TOL)
    assert ET1 == pytest.approx(0.5, abs=TOL)
    assert ET2 == pytest.approx(0.25, abs=TOL)

    assert ET1 == pytest.approx(P1 * ES1, abs=TOL)
    assert ET2 == pytest.approx(P2 * ES2, abs=TOL)


def test_naive_survivor_clock_thinning_identity_fails(population):
    """Survivor clock: conditioning on T>0 breaks E[T_j] = p_j E[S_j] for campaign 2 by exactly 1/p_2 = 2, because
    survival is informative about S when p_1 != p_2.
    """
    labels, retained = population[1], population[2]
    S1, S2, T1, T2 = _S_T(labels, retained)
    surv = retained

    ES1_s, ES2_s = S1[surv].mean(), S2[surv].mean()
    ET1_s, ET2_s = T1[surv].mean(), T2[surv].mean()

    assert ES2_s == pytest.approx(1 / 3, abs=TOL)
    assert ET2_s == pytest.approx(1 / 3, abs=TOL)
    assert (P2 * ES2_s) == pytest.approx(1 / 6, abs=TOL)

    # Campaign 1 is degenerate (p_1=1), which is why the counterexample needs unequal retention.
    assert ET1_s == pytest.approx(P1 * ES1_s, abs=TOL)

    # Campaign 2 is where the identity breaks.
    assert ET2_s != pytest.approx(P2 * ES2_s, abs=TOL)
    assert abs(ET2_s - P2 * ES2_s) > 0.1


def test_survivor_clock_bias_does_not_shrink_with_sample_size():
    """The identity failure is a fixed bias: at a much smaller N the gap stays near the analytic 1/6."""
    rng_small = np.random.default_rng(SEED + 1)
    _, labels, retained = _make_population(50_000, rng_small)
    _, S2, _, T2 = _S_T(labels, retained)
    surv = retained
    gap = T2[surv].mean() - P2 * S2[surv].mean()
    assert gap == pytest.approx(1 / 6, abs=0.02)


def test_a_hat_is_the_same_under_either_clock_non_discriminating(population):
    """a_j = sum T_j / sum T is the same under either clock, since the dropped rows contribute 0 to numerator and
    denominator.

        This is why a_j alone cannot discriminate between the clocks.
    """
    labels, retained = population[1], population[2]
    S1, S2, T1, T2 = _S_T(labels, retained)
    q = PI * P1 / (PI * P1 + (1 - PI) * P2)
    assert q == pytest.approx(2 / 3)

    a1_population = T1.sum() / (T1.sum() + T2.sum())          # all N rows
    a1_survivor = T1[retained].sum() / (T1[retained].sum() + T2[retained].sum())  # T=0 rows dropped
    assert a1_population == pytest.approx(q, abs=TOL)
    assert a1_survivor == pytest.approx(q, abs=TOL)
    assert a1_population == pytest.approx(a1_survivor, abs=1e-9)


def test_production_cs_a_hat_matches(population):
    """`cs.MultiCampaignCumulativeShareCS`, fed one population-indexed update per journey, gives the same a_hat ->
    (2/3, 1/3).

        An independent re-implementation, not a call into the experiment scripts.
    """
    rng = np.random.default_rng(SEED + 2)
    journeys, _, _ = _make_population(200_000, rng)
    p = {1: P1, 2: P2}
    mg = MultiCampaignCumulativeShareCS(2, ALPHA, B_BOUND)
    for j in journeys:
        obs = thin(j, p, rng)
        Tvec = np.zeros(2)
        if obs is not None:
            for t in obs.touches:
                Tvec[t.campaign - 1] += t.weight
        mg.update(Tvec, float(Tvec.sum()))
    L, U = mg.intervals()
    a_hat = (L + U) / 2
    assert a_hat[0] == pytest.approx(2 / 3, abs=0.02)
    assert a_hat[1] == pytest.approx(1 / 3, abs=0.02)


def test_prop3_interval_contains_true_delta_non_discriminating():
    """The identified interval for Delta_12 built from a_hat=[2/3,1/3] at Gamma=2 contains the true Delta_12=0.

        Built from a_hat alone, so it is clock-invariant; it checks the gap interval, not the indexing.
    """
    a_hat = [2 / 3, 1 / 3]
    lo, hi = prop3_gap_interval(a_hat, Gamma=GAMMA, j=0, k=1)
    true_delta = PI - PI  # = 0 by construction (equal true composition)
    assert lo <= true_delta <= hi
