"""Time-decay attribution: weight 2**(-tau/h) by recency to conversion.

The half-life h is fixed at construction, not learned, so the aggregate share is
theorem-covered. Weights are recomputed from each touch's own timestamp at credit
time, so the same weight-1 `Journey` objects serve every model.
"""
from __future__ import annotations

import numpy as np

from typing import Optional

from ..journeys import Journey


class TimeDecay:
    def __init__(self, h: float = 7.0):
        self.h = h

    def fit(
        self,
        train_journeys: list[Journey],
        n_campaigns: int,
        seed: int = 0,
        negative_journeys: Optional[list[Journey]] = None,
    ) -> None:
        self.n_campaigns = n_campaigns  # h fixed at construction; negative_journeys unused

    def credit(self, journey: Journey) -> np.ndarray:
        masses = np.zeros(self.n_campaigns)
        for t in journey.touches:
            tau = max(journey.conversion_time - t.timestamp, 0.0)  # clamp float noise at 0
            masses[t.campaign] += 2.0 ** (-tau / self.h)
        total = masses.sum()
        return masses / total if total > 0 else masses
