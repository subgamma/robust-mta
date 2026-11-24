"""Last-touch attribution: all credit to the campaign of the most recent touch (empirical baseline)."""
from __future__ import annotations

import numpy as np

from typing import Optional

from ..journeys import Journey


class LastTouch:
    def fit(
        self,
        train_journeys: list[Journey],
        n_campaigns: int,
        seed: int = 0,
        negative_journeys: Optional[list[Journey]] = None,
    ) -> None:
        self.n_campaigns = n_campaigns  # no parameters to learn; negative_journeys unused

    def credit(self, journey: Journey) -> np.ndarray:
        z = np.zeros(self.n_campaigns)
        last = max(journey.touches, key=lambda t: t.timestamp)
        z[last.campaign] = 1.0
        return z
