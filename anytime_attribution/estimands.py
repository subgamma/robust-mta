"""Aggregate share A/a and dashboard share Q/q from a stream of journeys.

One pair of functions serves both the full (latent) and observed versions. A journey
with no retained touch contributes 0/0 to both sums and can be absent, so conditioning
on T > 0 is immaterial: a stream of latent journeys yields A_j, Q_j and a stream of
observed (thinned, non-empty) journeys yields a_j, q_j.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .journeys import Journey


def campaign_masses(journey: Journey, n_campaigns: int) -> np.ndarray:
    """S_j (or T_j for an observed journey) for j = 0..n_campaigns-1."""
    masses = np.zeros(n_campaigns)
    for t in journey.touches:
        masses[t.campaign] += t.weight
    return masses


def per_journey_shares(journey: Journey, n_campaigns: int) -> np.ndarray:
    """Z_j = S_j / S for one journey. Raises if S = 0; zero-mass journeys are absent from observed streams."""
    masses = campaign_masses(journey, n_campaigns)
    total = masses.sum()
    if total <= 0:
        raise ValueError("per-journey share undefined for a zero-mass journey")
    return masses / total


def aggregate_share(stream: Iterable[Journey], n_campaigns: int) -> np.ndarray:
    """A_j = E[S_j]/E[S], or a_j = E[T_j]/E[T] when `stream` is the observed (non-empty) stream."""
    total_masses = np.zeros(n_campaigns)
    total = 0.0
    for journey in stream:
        m = campaign_masses(journey, n_campaigns)
        total_masses += m
        total += m.sum()
    if total <= 0:
        raise ValueError("aggregate_share undefined for an empty or all-zero-mass stream")
    return total_masses / total


def dashboard_share(stream: Iterable[Journey], n_campaigns: int) -> np.ndarray:
    """Q_j = E[S_j/S], or q_j = E[T_j/T | T>0] for an observed stream. Every journey must have S > 0."""
    total = np.zeros(n_campaigns)
    count = 0
    for journey in stream:
        total += per_journey_shares(journey, n_campaigns)
        count += 1
    if count == 0:
        raise ValueError("dashboard_share undefined for an empty stream")
    return total / count
