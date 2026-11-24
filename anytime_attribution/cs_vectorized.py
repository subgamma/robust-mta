"""Vectorised empirical-Bernstein confidence sequence (Howard et al., 2021, Thm. 4) at every t of a stored stream.

Does not replace `cs.py`: the incremental classes there are the reference, and the
tests check that this module reproduces them exactly (m = 1) for both boundary
forms. It lets Monte Carlo studies check coverage at every t without a Python loop
per observation. The intrinsic-time scale `m` is exposed for width diagnostics only.

Definitions as in `cs.py`: X_i in [0,1]; forecast Xhat_1 = 0.5, Xhat_i = mean(X_1..X_{i-1});
V_t = sum_{i<=t} (X_i - Xhat_i)^2; radius r_t = S_alpha(max(V_t, m)) / t with
S_alpha(v) = sqrt(k1^2 v l + k2^2 c^2 l^2) + c k2 l (exact; "simplified" is k1 sqrt(v l) + 2 c k2 l),
l(v) = s loglog(eta v / m) + log(zeta(s) / (alpha (log eta)^s)),
k1 = (eta^{1/4} + eta^{-1/4})/sqrt(2), k2 = (sqrt(eta) + 1)/2, c = 1, eta = 2, s = 1.4.
`alpha` is the per-boundary crossing probability (beta/2 for two-sided per-process error beta).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import zeta as _zeta

from .cs import _check_form

ETA, S_PAR, C_SCALE = 2.0, 1.4, 1.0
_K1 = (ETA ** 0.25 + ETA ** -0.25) / math.sqrt(2.0)
_K2 = (math.sqrt(ETA) + 1.0) / 2.0
_ZETA_S = float(_zeta(S_PAR))
_LOG_ETA_S = math.log(ETA) ** S_PAR


def lagged_mean_variance_process(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Running mean and V_t for a stream of shape (N) or (N,K) in [0,1], using the lagged-mean forecast (Xhat_1 = 0.5)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    csum = np.cumsum(x, axis=0)
    t = np.arange(1, n + 1, dtype=float).reshape((-1,) + (1,) * (x.ndim - 1))
    mean = csum / t
    fore = np.empty_like(x)
    fore[0] = 0.5
    if n > 1:
        tm1 = np.arange(1, n, dtype=float).reshape((-1,) + (1,) * (x.ndim - 1))
        fore[1:] = csum[:-1] / tm1
    V = np.cumsum((x - fore) ** 2, axis=0)
    return mean, V


def stitched_constants(eta: float, s: float):
    """(k1, k2, zeta(s), (log eta)^s) for general (eta, s), Howard et al. (2021) eq. (8)."""
    return ((eta ** 0.25 + eta ** -0.25) / math.sqrt(2.0), (math.sqrt(eta) + 1.0) / 2.0, float(_zeta(s)), math.log(eta) ** s)


def howard_radius_vec(V: np.ndarray, t: np.ndarray, alpha_used: float, m: float = 1.0,
                      eta: float = ETA, s: float = S_PAR, form=None) -> np.ndarray:
    """Vectorised `cs.howard_eb_radius` with an explicit intrinsic-time scale m.

    m = 1, eta = 2, s = 1.4 reproduces the production boundary.
    form = "exact": S(v) = sqrt(k1^2 v l + k2^2 c^2 l^2) + c k2 l (Thm. 1, eq. (8)); never larger than "simplified".
    form = "simplified": S(v) = k1 sqrt(v l) + 2 c k2 l, the upper bound stated after eq. (8).
    """
    if m <= 0:
        raise ValueError("m must be positive")
    form = _check_form(form)      # None -> cs.BOUNDARY_DEFAULT ("exact") or the active boundary_override
    k1, k2, zs, lg = stitched_constants(eta, s)
    v = np.maximum(V, m)
    L = s * np.log(np.log(eta * v / m)) + math.log(zs / (alpha_used * lg))
    if form == "simplified":
        S = k1 * np.sqrt(v * L) + 2.0 * C_SCALE * k2 * L
    else:
        S = np.sqrt(k1 ** 2 * v * L + (k2 * C_SCALE * L) ** 2) + C_SCALE * k2 * L
    return S / t


def cumulative_share_boxes(Tj: np.ndarray, T: np.ndarray, B: float, alpha_used: float, m: float = 1.0, **radius_kw):
    """Boxes at every t for the K numerator means and the shared denominator, all at level `alpha_used`, plus the
    [0,1]-clipped ratio box.

    Returns a dict with LN, UN (N,K), LD, UD (N), L, U (N,K), mean_num (N,K), mean_den (N).
    """
    Tj = np.asarray(Tj, dtype=float); T = np.asarray(T, dtype=float)
    tol = 1e-9 * B        # floating-point rounding slack only, same as cs._MASS_TOL; anything larger raises
    if not (np.isfinite(Tj).all() and np.isfinite(T).all()) or np.any(Tj < -tol) or np.any(Tj > B + tol) or np.any(T < -tol) or np.any(T > B + tol):
        raise ValueError("masses outside [0, B] (nothing is clipped; fix the generator)")
    Tj = np.clip(Tj, 0.0, B); T = np.clip(T, 0.0, B)   # snaps only the <= tol rounding slack verified above
    n = len(T)
    t = np.arange(1, n + 1, dtype=float)
    mn, Vn = lagged_mean_variance_process(Tj / B)
    md, Vd = lagged_mean_variance_process(T / B)
    rn = howard_radius_vec(Vn, t[:, None], alpha_used, m, **radius_kw)
    rd = howard_radius_vec(Vd, t, alpha_used, m, **radius_kw)
    LN, UN = np.clip(mn - rn, 0, 1), np.clip(mn + rn, 0, 1)
    LD, UD = np.clip(md - rd, 0, 1), np.clip(md + rd, 0, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(LD[:, None] > 0, np.clip(LN / UD[:, None], 0, 1), 0.0)
        U = np.where(LD[:, None] > 0, np.clip(UN / np.where(LD > 0, LD, 1.0)[:, None], 0, 1), 1.0)
    return {"LN": LN, "UN": UN, "LD": LD, "UD": UD, "L": L, "U": U, "mean_num": mn, "mean_den": md}
