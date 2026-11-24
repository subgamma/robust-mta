"""Fixed-sample baselines for E1/E2.

Each is valid at one pre-specified sample size and is recomputed at every look
without a peeking correction, so simultaneous coverage degrades under continuous
monitoring. None is time-uniform.

- `bootstrap_ci`: percentile bootstrap of the (T_ij, T_i) pairs for a_j = E[T_ij]/E[T_i].
- `fixed_eb_ci`: delta-method normal interval plus an empirical-Bernstein slack term.
- `csranks_style_ci`: the delta-method interval at Bonferroni level alpha/K, standing in
  for the fixed-time simultaneous rank sets of Mogstad et al. (2024). It is a
  Python approximation of the target, labelled "csranks-style", not the R package.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import norm


def bootstrap_ci(
    Tj_history: Sequence[float],
    T_history: Sequence[float],
    alpha: float,
    B: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Fixed-sample percentile bootstrap CI for a_j = E[T_ij]/E[T_i]."""
    Tj = np.asarray(Tj_history)
    T = np.asarray(T_history)
    n = len(Tj)
    idx = rng.integers(0, n, size=(B, n))
    ratios = Tj[idx].sum(axis=1) / T[idx].sum(axis=1)
    lo = np.quantile(ratios, alpha / 2)
    hi = np.quantile(ratios, 1 - alpha / 2)
    return float(lo), float(hi)


def _delta_method_se(Tj: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """Ratio estimate and delta-method standard error, Var(Tj - ratio*T) / (n * mean(T)^2)."""
    n = len(Tj)
    ratio_hat = Tj.sum() / T.sum()
    resid = Tj - ratio_hat * T
    var_resid = resid.var(ddof=1) if n > 1 else resid[0] ** 2
    se = math.sqrt(var_resid / n) / T.mean()
    return ratio_hat, se


def fixed_eb_ci(
    Tj_history: Sequence[float],
    T_history: Sequence[float],
    alpha: float,
    bound: float,
) -> tuple[float, float]:
    """Fixed-sample delta-method interval plus a Bernstein slack term at level alpha.

    `bound` is the per-journey mass bound B, not the bootstrap resample count.
    """
    Tj = np.asarray(Tj_history)
    T = np.asarray(T_history)
    n = len(Tj)
    ratio_hat, se = _delta_method_se(Tj, T)
    z = _normal_quantile(1 - alpha / 2)
    slack = bound * math.log(2 / alpha) / (3 * max(n - 1, 1))
    half_width = z * se + slack
    return ratio_hat - half_width, ratio_hat + half_width


def csranks_style_ci(
    Tj_history: Sequence[float],
    T_history: Sequence[float],
    alpha: float,
    n_campaigns: int,
) -> tuple[float, float]:
    """Fixed-sample delta-method interval at Bonferroni level alpha/K (see module docstring)."""
    Tj = np.asarray(Tj_history)
    T = np.asarray(T_history)
    ratio_hat, se = _delta_method_se(Tj, T)
    z = _normal_quantile(1 - alpha / (2 * n_campaigns))
    half_width = z * se
    return ratio_hat - half_width, ratio_hat + half_width


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF."""
    return float(norm.ppf(p))
