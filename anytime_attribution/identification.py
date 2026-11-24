"""Identified sets under bounded differential observability.

Covers the aggregate-share interval, its ratio form, the vertex reduction for the
pairwise-gap objective (shared with the certification rules in `certify.py`), the
ordering corollary, the within-journey linear bound and the dashboard envelope.

Every function takes an optional `p_lo`:
* `p_lo=None`: the ratio-only model P(Gamma) = P(Gamma, 0). The aggregate-share,
  gap and ordering results are sharp there, since p can be scaled down so that
  any ratio Gamma is feasible.
* `p_lo` given: the floored model P(Gamma, p_lo) = P(Gamma_eff, p_lo) with
  Gamma_eff = min(Gamma, 1/p_lo), because a ratio above 1/p_lo is infeasible for
  p in [p_lo, 1]. The same formulas at Gamma_eff are sharp; at Gamma they are a
  valid but non-sharp superset when Gamma > 1/p_lo.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def gamma_eff(Gamma: float, p_lo: float) -> float:
    """Gamma_eff = min(Gamma, 1/p_lo): retention ratios cannot exceed 1/p_lo."""
    if not (0 < p_lo <= 1):
        raise ValueError("p_lo must be in (0, 1]")
    return min(Gamma, 1.0 / p_lo)


def _effective_gamma(Gamma: float, p_lo: float | None) -> float:
    """Gamma if p_lo is None (ratio-only model), else min(Gamma, 1/p_lo)."""
    if Gamma < 1:
        raise ValueError("Gamma must be >= 1")
    return Gamma if p_lo is None else gamma_eff(Gamma, p_lo)


def gamma_star_floor_saturated(gamma_star_value: float, p_lo: float) -> bool:
    """Interpretation of a ratio-only Gamma* under an absolute floor.

    If Gamma* > 1/p_lo, certification already holds at Gamma_eff = 1/p_lo, i.e. for
    every retention vector in P(Gamma, p_lo) whatever Gamma, so the floored radius is
    unbounded. Gamma* is a sup over {Gamma: lower gap > delta} and the lower gap is
    continuous and non-increasing in Gamma, so the comparison is strict.
    """
    if not (0 < p_lo <= 1):
        raise ValueError("p_lo must be in (0, 1]")
    return gamma_star_value > 1.0 / p_lo


def prop2_interval(a_j: float, Gamma: float, p_lo: float | None = None) -> tuple[float, float]:
    """Identified set Theta_j(Gamma) for the aggregate share A_j.

    Theta_j(Gamma) = [a_j / (Gamma - (Gamma-1) a_j), Gamma a_j / (1 + (Gamma-1) a_j)].
    Sharp over P(Gamma) (p_lo=None); with a floor it is evaluated at Gamma_eff.
    """
    Gamma = _effective_gamma(Gamma, p_lo)
    lo = a_j / (Gamma - (Gamma - 1) * a_j)
    hi = Gamma * a_j / (1 + (Gamma - 1) * a_j)
    return lo, hi


def robust_A_interval(L_t: float, U_t: float, Gamma: float, p_lo: float | None = None) -> tuple[float, float]:
    """Interval for the true aggregate share A_j from a confidence box [L_t, U_t] on a_j.

    lo(a) and hi(a) of `prop2_interval` are strictly increasing in a for Gamma >= 1,
    so if a_j lies in [L_t, U_t] then A_j lies in [lo(L_t), hi(U_t)]. Hence
    P(for all t: A_j in [lo(L^j_t), hi(U^j_t)]) >= 1 - alpha whenever the boxes
    are simultaneously valid at level alpha.
    """
    lo, _ = prop2_interval(L_t, Gamma, p_lo)
    _, hi = prop2_interval(U_t, Gamma, p_lo)
    return lo, hi


def ratio_bounds(a_j: float, a_k: float, Gamma: float, p_lo: float | None = None) -> tuple[float, float]:
    """Ratio form: A_j/A_k in [a_j/(Gamma a_k), Gamma a_j/a_k], attained at p_j/p_k = Gamma^{+-1}.

    The normalising denominator cancels, so this is not the product of two marginal
    bounds. With a floor, Gamma is replaced by Gamma_eff.
    """
    Gamma = _effective_gamma(Gamma, p_lo)
    return a_j / (Gamma * a_k), Gamma * a_j / a_k


def ordering_identified(a_j: float, a_k: float, Gamma: float, p_lo: float | None = None) -> bool:
    """j > k is identified iff a_j/a_k > Gamma (not Gamma**2) over P(Gamma); with a floor the threshold is Gamma_eff."""
    return a_j / a_k > _effective_gamma(Gamma, p_lo)


def gap_vertex_extrema(
    l: Sequence[float],
    h: Sequence[float],
    j: int,
    k: int,
) -> tuple[float, float]:
    """Vertex reduction for the gap objective.

    inf/sup of g_jk(z) = (z_j - z_k) / sum(z) over the box prod_m [l_m, h_m] with
    l_m >= 0, h_j > 0, h_k > 0. The minimiser is z_j=l_j, z_k=h_k, with every other
    coordinate at h_m if l_j - h_k > 0 else l_m; the maximiser is z_j=h_j, z_k=l_k,
    with every other coordinate at l_m if h_j - l_k > 0 else h_m. Both extremal
    points have positive denominator.

    Used for the identified gap (l=a, h=Gamma*a) and for the robust certification
    rule (l=L_t, h=Gamma*U_t).
    """
    l = np.asarray(l, dtype=float)
    h = np.asarray(h, dtype=float)
    if np.any(l < 0):
        raise ValueError("lower bounds must be nonnegative")
    if not (h[j] > 0 and h[k] > 0):
        raise ValueError("h_j and h_k must be strictly positive")

    other = [m for m in range(len(l)) if m not in (j, k)]

    z_min = l.copy()
    z_min[j], z_min[k] = l[j], h[k]
    fill_min = h if (l[j] - h[k]) > 0 else l
    for m in other:
        z_min[m] = fill_min[m]
    d_min = z_min.sum()
    g_min = (z_min[j] - z_min[k]) / d_min

    z_max = l.copy()
    z_max[j], z_max[k] = h[j], l[k]
    fill_max = l if (h[j] - l[k]) > 0 else h
    for m in other:
        z_max[m] = fill_max[m]
    d_max = z_max.sum()
    g_max = (z_max[j] - z_max[k]) / d_max

    return g_min, g_max


def prop3_gap_interval(
    a: Sequence[float], Gamma: float, j: int, k: int, p_lo: float | None = None
) -> tuple[float, float]:
    """Identified set of the gap Delta_jk = A_j - A_k: vertex reduction on the box [a_m, Gamma*a_m].

    With a floor, Gamma is replaced by Gamma_eff.
    """
    a = np.asarray(a, dtype=float)
    Gamma = _effective_gamma(Gamma, p_lo)
    return gap_vertex_extrema(a, Gamma * a, j, k)


def lemma1_bounds(s_j: float, Gamma: float) -> tuple[float, float]:
    """Within-journey thinning bound (linear attribution only): E[M_j/M | journey, M>0] in [s_j/Gamma, Gamma s_j], s_j =
    n_j/n.
    """
    if Gamma < 1:
        raise ValueError("Gamma must be >= 1")
    return s_j / Gamma, Gamma * s_j


def prop7_envelope(q_j: float, Gamma: float, p_lo: float, n_min: int) -> tuple[float, float]:
    """Dashboard-share envelope (linear attribution only).

    p_lo <= q_j/Q_j <= Gamma_eff/s_low, equivalently
    Q_j in [(s_low/Gamma_eff) q_j, q_j/p_lo], s_low = 1 - (1-p_lo)**n_min.

    The endpoint Q_j <= q_j/p_lo follows from Jensen's inequality on
    sum_l p_j E[1/(1+M_-l)] and needs only p_lo. Both endpoints are attained in
    the limit for n_min = 1; attainment of the upper endpoint for n_min > 1 is not
    claimed.
    """
    if n_min < 1:
        raise ValueError("n_min must be >= 1")
    g_eff = gamma_eff(Gamma, p_lo)
    s_low = 1.0 - (1.0 - p_lo) ** n_min
    lo = (s_low / g_eff) * q_j
    hi = q_j / p_lo
    return lo, hi
