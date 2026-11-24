"""Linear attribution: equal credit to every touch (theorem-covered for the aggregate share)."""
from __future__ import annotations

import numpy as np

from typing import Optional

from ..estimands import per_journey_shares
from ..journeys import Journey


class Linear:
    def fit(
        self,
        train_journeys: list[Journey],
        n_campaigns: int,
        seed: int = 0,
        negative_journeys: Optional[list[Journey]] = None,
    ) -> None:
        self.n_campaigns = n_campaigns  # no parameters to learn; negative_journeys unused

    def credit(self, journey: Journey) -> np.ndarray:
        return per_journey_shares(journey, self.n_campaigns)
