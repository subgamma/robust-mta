"""Synthetic journey-type mixtures and exact/Monte-Carlo moments under thinning (verification only).

A journey "type" is a vector of touch counts per campaign (weight 1, linear
attribution only). A "mixture" is a finite set of types with probabilities summing
to 1. `P_LO_GRID` and `GAMMA_GRID` define the retention grid.

Q_j = E[S_j/S] and A_j = E[S_j]/E[S] are closed-form in the type counts. a_j is
also closed-form under independent per-campaign thinning (retention rescales first
moments), but q_j needs E[M_j/M | type, M>0]: computed by exact enumeration over
retention patterns for small state spaces and by Monte Carlo (>= 10**6 draws)
above `EXACT_ENUM_THRESHOLD`.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import binom

P_LO_GRID = (0.2, 0.5, 0.8)
GAMMA_GRID = (1.0, 1.25, 1.5, 2.0)

EXACT_ENUM_THRESHOLD = 200_000  # max prod(n_k+1) over active campaigns
DEFAULT_MC_SAMPLES = 1_000_000


@dataclass(frozen=True)
class JourneyType:
    counts: tuple[int, ...]  # n_j for each campaign j; must sum to >= 1

    def __post_init__(self):
        if sum(self.counts) < 1:
            raise ValueError("a converting journey type must have at least one touch")

    @property
    def n(self) -> int:
        return sum(self.counts)


@dataclass(frozen=True)
class Mixture:
    types: tuple[JourneyType, ...]
    weights: tuple[float, ...]

    def __post_init__(self):
        if len(self.types) != len(self.weights):
            raise ValueError("types and weights must have the same length")
        if abs(sum(self.weights) - 1.0) > 1e-8:
            raise ValueError("mixture weights must sum to 1")


def exact_A(mixture: Mixture, n_campaigns: int) -> np.ndarray:
    """A_j = E[S_j]/E[S], closed form in type counts (linear)."""
    total = np.zeros(n_campaigns)
    denom = 0.0
    for t, w in zip(mixture.types, mixture.weights):
        c = np.zeros(n_campaigns)
        c[: len(t.counts)] = t.counts
        total += w * c
        denom += w * c.sum()
    return total / denom


def exact_Q(mixture: Mixture, n_campaigns: int) -> np.ndarray:
    """Q_j = E[S_j/S], closed form in type counts (linear)."""
    total = np.zeros(n_campaigns)
    for t, w in zip(mixture.types, mixture.weights):
        c = np.zeros(n_campaigns)
        c[: len(t.counts)] = t.counts
        total += w * (c / c.sum())
    return total


def exact_a(mixture: Mixture, p: Sequence[float], n_campaigns: int) -> np.ndarray:
    """a_j = E[T_j]/E[T], closed form: E[T_j | type] = p_j * n_j."""
    p = np.asarray(p, dtype=float)
    total = np.zeros(n_campaigns)
    denom = 0.0
    for t, w in zip(mixture.types, mixture.weights):
        c = np.zeros(n_campaigns)
        c[: len(t.counts)] = t.counts
        contrib = p * c
        total += w * contrib
        denom += w * contrib.sum()
    return total / denom


def _binom_pmf(n: int, p: float) -> np.ndarray:
    """Numerically stable binomial pmf (math.comb(n, m) overflows float conversion for n in the low thousands)."""
    return binom.pmf(np.arange(n + 1), n, p)


def _type_state_space_size(counts: Sequence[int]) -> int:
    size = 1
    for c in counts:
        if c > 0:
            size *= c + 1
    return size


def _type_moments_exact(
    counts: Sequence[int], p: Sequence[float], n_campaigns: int
) -> tuple[np.ndarray, float]:
    """Exact E[M_j/M | type, M>0] (all j) and P(M>0), by enumerating retention patterns."""
    active = [k for k in range(len(counts)) if counts[k] > 0]
    pmfs = {k: _binom_pmf(counts[k], p[k]) for k in active}
    ranges = [range(counts[k] + 1) for k in active]
    ratio_sum = np.zeros(n_campaigns)
    p_positive = 0.0
    for combo in itertools.product(*ranges):
        M = sum(combo)
        if M == 0:
            continue
        pr = 1.0
        for idx, k in enumerate(active):
            pr *= pmfs[k][combo[idx]]
        for idx, k in enumerate(active):
            ratio_sum[k] += pr * combo[idx] / M
        p_positive += pr
    if p_positive <= 0:
        return np.zeros(n_campaigns), 0.0
    return ratio_sum / p_positive, p_positive


def _type_moments_mc(
    counts: Sequence[int],
    p: Sequence[float],
    n_campaigns: int,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    """Monte Carlo estimate of `_type_moments_exact`, for state spaces too large to enumerate."""
    counts_arr = np.asarray(counts)
    M = np.zeros((n_samples, n_campaigns))
    for k in range(n_campaigns):
        if counts_arr[k] > 0:
            M[:, k] = rng.binomial(counts_arr[k], p[k], size=n_samples)
    T = M.sum(axis=1)
    mask = T > 0
    p_positive = float(mask.mean())
    if not np.any(mask):
        return np.zeros(n_campaigns), 0.0
    ratio = (M[mask] / T[mask, None]).mean(axis=0)
    return ratio, p_positive


def type_moments(
    counts: Sequence[int],
    p: Sequence[float],
    n_campaigns: int,
    method: str = "auto",
    mc_samples: int = DEFAULT_MC_SAMPLES,
    exact_threshold: int = EXACT_ENUM_THRESHOLD,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, float]:
    """E[M_j/M | type, M>0] (all j) and P(M>0) for one type: exact below `exact_threshold` retention-pattern
    combinations, Monte Carlo (>= 10**6 draws by default) above.
    """
    use_exact = method == "exact" or (
        method == "auto" and _type_state_space_size(counts) <= exact_threshold
    )
    if use_exact:
        return _type_moments_exact(counts, p, n_campaigns)
    if rng is None:
        rng = np.random.default_rng()
    return _type_moments_mc(counts, p, n_campaigns, mc_samples, rng)


def exact_q(
    mixture: Mixture,
    p: Sequence[float],
    n_campaigns: int,
    method: str = "auto",
    mc_samples: int = DEFAULT_MC_SAMPLES,
    exact_threshold: int = EXACT_ENUM_THRESHOLD,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """q_j = E[T_j/T | T>0], population-weighted by each type's survival probability."""
    num = np.zeros(n_campaigns)
    den = 0.0
    for t, w in zip(mixture.types, mixture.weights):
        ratio, surv = type_moments(
            t.counts,
            p,
            n_campaigns,
            method=method,
            mc_samples=mc_samples,
            exact_threshold=exact_threshold,
            rng=rng,
        )
        num += w * surv * ratio
        den += w * surv
    if den <= 0:
        raise ValueError("mixture has zero probability of any observed touch")
    return num / den


def retention_grid() -> list[tuple[float, float]]:
    """(p_lo, Gamma) retention grid."""
    return list(itertools.product(P_LO_GRID, GAMMA_GRID))


def two_type_mixture(n_campaigns: int, j: int, k: int, n_short: int, n_long: int, pi: float) -> Mixture:
    """Short-vs-long family: a fraction `pi` of journeys has `n_short` touches on campaign j, the rest `n_long` on
    campaign k.

    Q_j = pi exactly, independent of n_long. Captures only the between-journey
    dropout effect; see `two_campaign_mixture` for both effects.
    """
    short = [0] * n_campaigns
    short[j] = n_short
    long_ = [0] * n_campaigns
    long_[k] = n_long
    return Mixture(
        types=(JourneyType(tuple(short)), JourneyType(tuple(long_))),
        weights=(pi, 1.0 - pi),
    )


def two_campaign_mixture(
    n_campaigns: int,
    j: int,
    k: int,
    n_j1: int,
    n_k1: int,
    n_j2: int,
    n_k2: int,
    pi: float,
) -> Mixture:
    """General two-type mixture over campaigns j and k.

    Type 1 has (n_j1, n_k1) touches (weight pi) and type 2 has (n_j2, n_k2)
    (weight 1-pi); each type needs at least one positive count. Unlike
    `two_type_mixture`, a type may contain both campaigns, so one type carries both
    the within-journey and the between-journey distortion.
    """
    t1 = [0] * n_campaigns
    t1[j], t1[k] = n_j1, n_k1
    t2 = [0] * n_campaigns
    t2[j], t2[k] = n_j2, n_k2
    return Mixture(
        types=(JourneyType(tuple(t1)), JourneyType(tuple(t2))),
        weights=(pi, 1.0 - pi),
    )


# Population generator for E1-E4: each touch picks campaign j with probability w_j/sum(w), so A_j = w_j/sum(w) exactly
# for any length distribution; lengths follow a hand-specified distribution approximating the Criteo warm-up
# quantiles.

LENGTH_DIST_BINS: tuple[tuple[int, int, float], ...] = (
    (1, 1, 0.78),
    (2, 2, 0.12),
    (3, 3, 0.05),
    (4, 6, 0.03),
    (7, 15, 0.019),
    (16, 40, 0.001),
)
"""(lo, hi, prob) bins approximating the Criteo warm-up journey-length
quantiles (docs/criteo_preprocessing.md); within a bin, length is uniform
over the integers [lo, hi]. Probabilities sum to 1.0."""


def sample_journey_lengths(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw `n` journey lengths from `LENGTH_DIST_BINS`."""
    los = np.array([b[0] for b in LENGTH_DIST_BINS])
    his = np.array([b[1] for b in LENGTH_DIST_BINS])
    probs = np.array([b[2] for b in LENGTH_DIST_BINS])
    bin_idx = rng.choice(len(LENGTH_DIST_BINS), size=n, p=probs)
    lo, hi = los[bin_idx], his[bin_idx]
    return lo + rng.integers(0, hi - lo + 1)


def true_aggregate_share(campaign_weights: Sequence[float]) -> np.ndarray:
    """A_j = w_j / sum(w) exactly for linear attribution, for any journey-length distribution (each touch's campaign is
    independent of length).
    """
    w = np.asarray(campaign_weights, dtype=float)
    return w / w.sum()


def generate_synthetic_journeys(
    n: int,
    campaign_weights: Sequence[float],
    rng: np.random.Generator,
    window_days: float = 7.0,
) -> list[Journey]:
    """`n` latent synthetic journeys.

    Length from `sample_journey_lengths`; each touch's campaign i.i.d. from
    Categorical(campaign_weights/sum); timestamps i.i.d. Uniform(0, window_days),
    sorted; conversion at `window_days` (so tau = window_days - timestamp). Weight
    is 1 per touch; time-decay and last-touch models recompute their own weights.
    """
    # Local import avoids a module-level cycle.
    from .journeys import Journey, Touch

    w = np.asarray(campaign_weights, dtype=float)
    p = w / w.sum()
    lengths = sample_journey_lengths(n, rng)
    K = len(w)
    out = []
    for length in lengths:
        campaigns = rng.choice(K, size=length, p=p)
        timestamps = np.sort(rng.uniform(0.0, window_days, size=length))
        touches = tuple(
            Touch(campaign=int(c), timestamp=float(t), weight=1.0)
            for c, t in zip(campaigns, timestamps)
        )
        out.append(Journey(touches=touches, conversion_time=window_days))
    return out
