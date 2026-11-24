"""Journey dataclass, weight functions, cap rule (A4) and thinning (A2).

A touch weight w depends only on its own campaign and timestamp, not on its position
among the observed touches: a retained touch keeps the weight it had in the full journey.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Touch:
    campaign: int
    timestamp: float  # same units as conversion_time (e.g. days since epoch)
    weight: float


@dataclass(frozen=True)
class Journey:
    touches: tuple[Touch, ...]
    conversion_time: float

    @property
    def mass(self) -> float:
        """S = sum_j S_j"""
        return sum(t.weight for t in self.touches)

    @property
    def n_touches(self) -> int:
        return len(self.touches)


def linear_weight(timestamp: float, conversion_time: float) -> float:
    """w = 1 for every touch (linear attribution)."""
    return 1.0


def time_decay_weight(timestamp: float, conversion_time: float, h: float = 7.0) -> float:
    """w_ell = 2**(-tau_ell/h), tau_ell = time to conversion, h = half-life in days."""
    tau = conversion_time - timestamp
    if tau < 0:
        raise ValueError("touch timestamp is after conversion_time")
    return 2.0 ** (-tau / h)


WeightFn = Callable[[float, float], float]


def build_journey(
    raw_touches: Sequence[tuple[int, float]],
    conversion_time: float,
    weight_fn: WeightFn,
) -> Journey:
    """Attach weights to raw (campaign, timestamp) touches using `weight_fn`.

    `weight_fn` sees each touch alone, so a retained touch keeps its full-journey
    weight regardless of which other touches survive thinning.
    """
    touches = tuple(
        Touch(campaign=c, timestamp=t, weight=weight_fn(t, conversion_time))
        for c, t in raw_touches
    )
    return Journey(touches=touches, conversion_time=conversion_time)


def apply_cap(journey: Journey, lookback_days: float, max_touches: int) -> Journey:
    """Assumption A4 (bounded journey mass): fixed lookback window and max-touch rule.

    Applied to the latent journey, before attribution and before thinning.
    Touches with tau > lookback_days are dropped; if more than max_touches remain,
    the max_touches closest to conversion are kept, then re-sorted by time.
    """
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")
    if max_touches <= 0:
        raise ValueError("max_touches must be positive")
    kept = [
        t for t in journey.touches
        if (journey.conversion_time - t.timestamp) <= lookback_days
    ]
    if len(kept) > max_touches:
        kept = sorted(kept, key=lambda t: journey.conversion_time - t.timestamp)[:max_touches]
        kept = sorted(kept, key=lambda t: t.timestamp)
    return Journey(touches=tuple(kept), conversion_time=journey.conversion_time)


def thin(
    journey: Journey,
    p: dict[int, float],
    rng: np.random.Generator,
) -> Optional[Journey]:
    """Observation model A2: retain each touch independently with probability p[campaign].

    Returns None if no touch is retained; such journeys are absent from the observed stream.
    """
    kept = tuple(t for t in journey.touches if rng.random() < p[t.campaign])
    if not kept:
        return None
    return Journey(touches=kept, conversion_time=journey.conversion_time)
