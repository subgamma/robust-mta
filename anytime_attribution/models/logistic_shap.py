"""Logistic conversion model with SHAP-based per-touch credit.

Empirical: no identification guarantee; use only the empirical envelope Gamma_hat_sens, never Gamma*.

`fit` trains a logistic regression on per-campaign touch counts, with warm-up
(days 1-7) converting journeys as positives and non-converting users' click
histories as negatives. `credit` explains the predicted log-odds with
`shap.LinearExplainer`, which gives exact Shapley values for a linear model; this
replaces KernelSHAP, which is intractable at this dataset's scale. SHAP values are
clipped to non-negative and normalised to sum to 1; a journey whose SHAP values are all
<= 0 falls back to last-touch credit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression

from ..estimands import campaign_masses
from ..journeys import Journey


def _features(journeys: list[Journey], n_campaigns: int) -> np.ndarray:
    return np.stack([campaign_masses(j, n_campaigns) for j in journeys])


class LogisticSHAP:
    def __init__(self, C: float = 1.0):
        self.C = C
        self.n_fallback_ = 0

    def fit(
        self,
        train_journeys: list[Journey],
        n_campaigns: int,
        seed: int = 0,
        negative_journeys: Optional[list[Journey]] = None,
    ) -> None:
        if not negative_journeys:
            raise ValueError("LogisticSHAP.fit requires negative_journeys")
        self.n_campaigns = n_campaigns
        X_pos = _features(train_journeys, n_campaigns)
        X_neg = _features(negative_journeys, n_campaigns)
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(len(train_journeys)), np.zeros(len(negative_journeys))])
        self.model = LogisticRegression(C=self.C, max_iter=1000, random_state=seed)
        self.model.fit(X, y)
        background = X_neg[np.random.default_rng(seed).choice(len(X_neg), size=min(200, len(X_neg)), replace=False)]
        self.explainer = shap.LinearExplainer(self.model, background)

    def credit(self, journey: Journey) -> np.ndarray:
        x = campaign_masses(journey, self.n_campaigns)
        shap_values = np.asarray(self.explainer.shap_values(x.reshape(1, -1))).reshape(-1)
        # Mask to touched campaigns: SHAP can assign positive value to an unseen campaign (phi_j = beta_j*(0-E[x_j])),
        # which is not a valid credit.
        touched = x > 0
        z = np.where(touched, np.clip(shap_values, 0.0, None), 0.0)
        total = z.sum()
        if total <= 0:
            self.n_fallback_ += 1
            z = np.zeros(self.n_campaigns)
            last = max(journey.touches, key=lambda t: t.timestamp)
            z[last.campaign] = 1.0
            return z
        return z / total
