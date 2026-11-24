"""Time-uniform confidence sequences for bounded means.

Production sequences (`CumulativeShareCS`, `MultiCampaignCumulativeShareCS`,
`DirectNumeratorCS`) use the empirical-Bernstein bound of Howard et al. (2021),
Thm. 4, with the polynomial stitched boundary (eta=2, s=1.4, m=1, c=1).

`ConfidenceSequence` offers two boundaries for a fixed mean:
- "wsr" (default): Waudby-Smith & Ramdas (2023) predictable plug-in betting
  martingale. For any predictable bet lambda_i with 1 + lambda_i (X_i - m) >= 0,
  the wealth is a nonnegative martingale at the true mean, so Ville's
  inequality gives validity; the variance-adaptive bet only affects tightness.
- "howard": conservative comparator using a fixed grid of exponential tilts and
  the deterministic variance bound 1/4 (Popoviciu). Reports must label it
  "howard (conservative comparator)" (see `HOWARD_LABEL`).

Each boundary is applied one-sided at alpha/2 per side (sign=+1 rejects means
that are too small, sign=-1 too large) to form a two-sided sequence.

`drift_test_cs` is a diagnostic only, not a certification mechanism.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy.special import zeta as _scipy_zeta


class MassBoundError(ValueError):
    """An observation lies outside the bounded-mass support (T_ij <= T_i <= B).
    Clipping would change the covered process and void the coverage proof, so
    every update path raises instead. Subclasses ValueError.
    """


# Floating-point slack only; larger violations raise MassBoundError, smaller ones are snapped to the boundary.
_MASS_TOL = 1e-9


def _validated_unit(x, what: str):
    """Return `x` (scalar or array) after checking every entry is finite and
    in [0, 1] up to `_MASS_TOL`; raise `MassBoundError` otherwise. Entries in
    the rounding slack are snapped onto [0, 1]."""
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise MassBoundError(f"{what}: non-finite observation")
    if np.any(arr < -_MASS_TOL) or np.any(arr > 1.0 + _MASS_TOL):
        bad = float(arr.flat[int(np.argmax(np.abs(arr - 0.5)))])
        raise MassBoundError(
            f"{what}: normalised observation {bad!r} outside [0, 1]; the mass bound B "
            "(Assumption a:cap) is violated. Re-apply the cap rule or raise B before "
            "running the sequence; the CS never clips."
        )
    out = np.clip(arr, 0.0, 1.0)
    return float(out) if np.ndim(out) == 0 else out


def _validated_mass(x, B: float, what: str):
    """Raw-mass version of `_validated_unit`: every entry finite and in [0, B]
    up to `_MASS_TOL * B`; raises `MassBoundError` otherwise; rounding slack is
    snapped. Used by the betting/diagnostic paths that take raw masses."""
    if not (B > 0 and math.isfinite(B)):
        raise ValueError("B must be positive and finite")
    arr = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise MassBoundError(f"{what}: non-finite observation")
    tol = _MASS_TOL * B
    if np.any(arr < -tol) or np.any(arr > B + tol):
        bad = float(arr.flat[int(np.argmax(np.abs(arr - 0.5 * B)))])
        raise MassBoundError(
            f"{what}: observation {bad!r} outside [0, B={B}]; Assumption a:cap is violated. "
            "Nothing is clipped or rescaled."
        )
    out = np.clip(arr, 0.0, B)
    return float(out) if np.ndim(out) == 0 else out


class _WSRSide:
    """One-sided sticky-rejection WSR betting test for H0: rescaled mean = m_tilde."""

    def __init__(self, m_tilde: float, alpha: float, sign: int):
        self.m = m_tilde
        self.alpha = alpha
        self.sign = sign
        self.t = 0
        self.sum_x = 0.5
        self.sum_sq_dev = 0.25
        self.log_wealth = 0.0
        self.max_log_wealth = 0.0
        self.rejected = False
        self._log_threshold = math.log(1.0 / alpha)

    def update(self, x_tilde: float) -> None:
        if self.rejected:
            self.t += 1
            self.sum_x += x_tilde
            return
        i = self.t + 1
        mu_prev = self.sum_x / i
        var_prev = min(max(self.sum_sq_dev / i, 1e-6), 0.25)
        if i > 1:
            lam_mag = math.sqrt(2.0 * math.log(1.0 / self.alpha) / (var_prev * i * math.log(i + 1)))
        else:
            lam_mag = 0.0
        lam_mag = min(lam_mag, 1.0)
        lam = self.sign * lam_mag
        eps = 1e-9
        # Keep 1 + lam*(x-m) >= 0 on [0,1]; the cap is vacuous when m is near 0 (lam>0) or near 1 (lam<0).
        if lam > 0 and self.m > eps:
            lam = min(lam, (1.0 - eps) / self.m)
        elif lam < 0 and self.m < 1 - eps:
            lam = max(lam, -(1.0 - eps) / (1.0 - self.m))

        factor = max(1.0 + lam * (x_tilde - self.m), 1e-300)
        self.log_wealth += math.log(factor)
        if self.log_wealth > self.max_log_wealth:
            self.max_log_wealth = self.log_wealth
        if self.max_log_wealth >= self._log_threshold:
            self.rejected = True

        self.sum_sq_dev += (x_tilde - mu_prev) ** 2
        self.sum_x += x_tilde
        self.t += 1


class _HowardSide:
    """Conservative comparator (see module docstring): one-sided sticky-rejection grid-stitched Bernstein test of H0:
    rescaled mean = m_tilde, using the deterministic variance bound.
    """

    N_GRID = 12

    def __init__(self, m_tilde: float, alpha: float, sign: int):
        self.m = m_tilde
        self.sign = sign
        self.t = 0
        b = 1.0
        lam_max = 2.9 / b
        lams = np.geomspace(0.02, lam_max, self.N_GRID)
        self.lams = sign * lams
        var_bound = 0.25  # Popoviciu: Var(X) <= 1/4 for X in [0,1], deterministic
        self.psi = (self.lams ** 2) * var_bound / (2.0 * (1.0 - b * np.abs(self.lams) / 3.0))
        self.log_wealth = np.zeros(self.N_GRID)
        self.max_log_wealth = np.zeros(self.N_GRID)
        self.log_threshold = math.log(self.N_GRID / alpha)
        self.rejected = False

    def update(self, x_tilde: float) -> None:
        if not self.rejected:
            xi = x_tilde - self.m
            self.log_wealth += self.lams * xi - self.psi
            np.maximum(self.max_log_wealth, self.log_wealth, out=self.max_log_wealth)
            if np.any(self.max_log_wealth >= self.log_threshold):
                self.rejected = True
        self.t += 1


_SIDE_CLASSES = {"wsr": _WSRSide, "howard": _HowardSide}

# Reports broken out by boundary must use these labels, not the bare key; "howard" is a conservative comparator.
BOUNDARY_LABELS = {
    "wsr": "wsr (primary)",
    "howard": "howard (conservative comparator)",
}


def _two_sided_rejected(xs_tilde: Iterable[float], m_tilde: float, alpha: float, boundary: str) -> bool:
    """Streams already-rescaled-to-[0,1] increments through a two-sided sticky
    test for H0: mean = m_tilde, total error `alpha`; returns True as soon as
    m_tilde is ever excluded (Ville / sticky rejection)."""
    side_cls = _SIDE_CLASSES[boundary]
    alpha_side = alpha / 2.0
    lower = side_cls(m_tilde, alpha_side, sign=+1)
    upper = side_cls(m_tilde, alpha_side, sign=-1)
    for x in xs_tilde:
        if not lower.rejected:
            lower.update(x)
        if not upper.rejected:
            upper.update(x)
        if lower.rejected or upper.rejected:
            return True
    return False


def covers_throughout(
    xs: Iterable[float],
    m: float,
    alpha: float,
    lo: float = 0.0,
    hi: float = 1.0,
    boundary: str = "wsr",
) -> bool:
    """True iff `m` is never excluded by the two-sided sequence while streaming `xs` (O(1) memory)."""
    m_tilde = (m - lo) / (hi - lo)
    xs_tilde = ((x - lo) / (hi - lo) for x in xs)
    return not _two_sided_rejected(xs_tilde, m_tilde, alpha, boundary)


def first_directional_exclusion(
    xs: Iterable[float],
    m: float,
    alpha: float,
    lo: float = 0.0,
    hi: float = 1.0,
    boundary: str = "wsr",
) -> tuple[Optional[int], Optional[int]]:
    """First look (1-indexed) at which `m` is excluded, and the direction.

    Direction is +1 if the running mean is above `m` and -1 if below;
    (None, None) if `m` is never excluded.
    """
    m_tilde = (m - lo) / (hi - lo)
    side_cls = _SIDE_CLASSES[boundary]
    alpha_side = alpha / 2.0
    lower = side_cls(m_tilde, alpha_side, sign=+1)
    upper = side_cls(m_tilde, alpha_side, sign=-1)
    for i, x in enumerate(xs, start=1):
        x_tilde = (x - lo) / (hi - lo)
        if not lower.rejected:
            lower.update(x_tilde)
        if not upper.rejected:
            upper.update(x_tilde)
        if lower.rejected:
            return i, +1
        if upper.rejected:
            return i, -1
    return None, None


def coverage_at_looks(
    xs: Sequence[float],
    m: float,
    alpha: float,
    look_points: Sequence[int],
    lo: float = 0.0,
    hi: float = 1.0,
    boundary: str = "wsr",
) -> dict[int, bool]:
    """Single-pass sweep reporting, for each t in `look_points`, whether `m`
    was still covered after exactly t updates. Because rejection is sticky,
    this is also whether m was covered at EVERY look <= t -- used to check
    that peeking at intermediate looks does not change the final coverage
    verdict (App. B unit test (ii))."""
    m_tilde = (m - lo) / (hi - lo)
    side_cls = _SIDE_CLASSES[boundary]
    alpha_side = alpha / 2.0
    lower = side_cls(m_tilde, alpha_side, sign=+1)
    upper = side_cls(m_tilde, alpha_side, sign=-1)
    look_set = set(look_points)
    max_look = max(look_points)
    result: dict[int, bool] = {}
    for i, x in enumerate(xs, start=1):
        if i > max_look:
            break
        x_tilde = (x - lo) / (hi - lo)
        if not lower.rejected:
            lower.update(x_tilde)
        if not upper.rejected:
            upper.update(x_tilde)
        if i in look_set:
            result[i] = not (lower.rejected or upper.rejected)
    return result


class ConfidenceSequence:
    """Time-uniform confidence sequence for the mean of bounded increments in
    [lo, hi]. Usage: ConfidenceSequence(alpha).update(x1).update(x2)....interval.
    """

    def __init__(self, alpha: float, lo: float = 0.0, hi: float = 1.0, boundary: str = "wsr"):
        if boundary not in _SIDE_CLASSES:
            raise ValueError(f"unknown boundary {boundary!r}")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        if hi <= lo:
            raise ValueError("hi must be > lo")
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.boundary = boundary
        self._history: list[float] = []

    def update(self, x: float) -> "ConfidenceSequence":
        if not (isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))):
            raise MassBoundError(f"ConfidenceSequence.update: non-finite observation {x!r}")
        if not (self.lo - 1e-9 <= x <= self.hi + 1e-9):
            raise MassBoundError(f"x={x} outside declared bounds [{self.lo}, {self.hi}]")
        self._history.append(min(max(x, self.lo), self.hi))
        return self

    def rejects(self, m: float) -> bool:
        """Whether `m` has ever been excluded, replaying the full history."""
        m_tilde = (m - self.lo) / (self.hi - self.lo)
        xs_tilde = ((x - self.lo) / (self.hi - self.lo) for x in self._history)
        return _two_sided_rejected(xs_tilde, m_tilde, self.alpha, self.boundary)

    def interval(self, tol: float = 1e-4) -> tuple[float, float]:
        """Current interval [L_t, U_t] by bisection.

        The accepted set is an interval: the sign=+1 wealth is non-increasing in m
        and the sign=-1 wealth is non-decreasing, so each rejection indicator flips
        at most once moving away from the empirical mean.
        """
        if not self._history:
            return self.lo, self.hi
        mean = sum(self._history) / len(self._history)
        mean = min(max(mean, self.lo), self.hi)
        width = self.hi - self.lo

        lo_bound = self.lo
        if self.rejects(self.lo):
            a, b = self.lo, mean
            for _ in range(60):
                mid = (a + b) / 2
                if self.rejects(mid):
                    a = mid
                else:
                    b = mid
                if b - a < tol * width:
                    break
            lo_bound = b

        hi_bound = self.hi
        if self.rejects(self.hi):
            a, b = mean, self.hi
            for _ in range(60):
                mid = (a + b) / 2
                if self.rejects(mid):
                    b = mid
                else:
                    a = mid
                if b - a < tol * width:
                    break
            hi_bound = a

        return lo_bound, hi_bound


def theta_rejected(
    Tj_history: Sequence[float],
    T_history: Sequence[float],
    theta: float,
    alpha: float,
    B: float,
    boundary: str = "wsr",
) -> bool:
    """Single-theta test underlying `invert_share_cs`.

    G_i(theta) = T_ij - theta*T_i lies in [-theta*B, B], and H0: theta = a_j iff E[G_i(theta)] = 0. It is rescaled
    to x_i(theta) = (G_i(theta)+theta*B)/((1+theta)*B) in [0,1] with null mean theta/(1+theta).
    """
    if theta < 0:
        raise ValueError("theta must be >= 0")
    # Reject (never clip) out-of-support raw masses; the min/max below only guards floating-point rounding.
    Tj_history = _validated_mass(Tj_history, B, "theta_rejected T_ij")
    T_history = _validated_mass(T_history, B, "theta_rejected T_i")
    Tj_history = np.atleast_1d(Tj_history); T_history = np.atleast_1d(T_history)
    denom = (1.0 + theta) * B
    m_tilde = theta / (1.0 + theta)
    xs_tilde = []
    for tj, t in zip(Tj_history, T_history):
        g = tj - theta * t
        x_tilde = (g + theta * B) / denom
        xs_tilde.append(min(max(x_tilde, 0.0), 1.0))
    return _two_sided_rejected(xs_tilde, m_tilde, alpha, boundary)


def invert_share_cs(
    Tj_history: Sequence[float],
    T_history: Sequence[float],
    alpha: float,
    B: float,
    boundary: str = "wsr",
    tol: float = 1e-4,
) -> tuple[float, float]:
    """[L_t, U_t] for a_j = E[T_ij]/E[T_i] by bisecting `theta_rejected` over [0, 1].

    Assumes the empirical ratio is accepted and the accepted set is a single
    interval containing it. This does not always hold (see `drift_test_cs`), so
    the production experiments use `CumulativeShareCS` instead.
    """
    if len(Tj_history) != len(T_history):
        raise ValueError("Tj_history and T_history must be the same length")
    if not Tj_history:
        return 0.0, 1.0

    total_Tj = sum(Tj_history)
    total_T = sum(T_history)
    anchor = total_Tj / total_T if total_T > 0 else 0.5
    anchor = min(max(anchor, 1e-6), 1 - 1e-6)

    def rej(theta: float) -> bool:
        return theta_rejected(Tj_history, T_history, theta, alpha, B, boundary)

    lo_bound = 0.0
    if rej(0.0):
        a, b = 0.0, anchor
        for _ in range(60):
            mid = (a + b) / 2
            if rej(mid):
                a = mid
            else:
                b = mid
            if b - a < tol:
                break
        lo_bound = b

    hi_bound = 1.0
    if rej(1.0):
        a, b = anchor, 1.0
        for _ in range(60):
            mid = (a + b) / 2
            if rej(mid):
                b = mid
            else:
                a = mid
            if b - a < tol:
                break
        hi_bound = a

    return lo_bound, hi_bound


class DriftTestResult:
    """Outcome of `drift_test_cs`. `status` is one of:
    - "interval": one contiguous accepted region, refined to `tol`; `lo`/`hi`
      are each accepted by `theta_rejected`.
    - "empty": no accepted theta at two scan resolutions; `lo`/`hi` are None.
      This rejects constant conditional share.
    - "indeterminate": more than one disjoint accepted region; `lo`/`hi` are
      None and no verdict is returned.
    """

    __slots__ = ("status", "lo", "hi", "n_scan_points")

    def __init__(self, status: str, lo: Optional[float], hi: Optional[float], n_scan_points: int):
        self.status = status
        self.lo = lo
        self.hi = hi
        self.n_scan_points = n_scan_points

    def __repr__(self) -> str:
        return f"DriftTestResult(status={self.status!r}, lo={self.lo}, hi={self.hi}, n_scan_points={self.n_scan_points})"


def _drift_scan(Tj_history, T_history, alpha, B, boundary, n_points) -> tuple[np.ndarray, np.ndarray]:
    """Rejected/accepted status at `n_points` evenly spaced theta in one vectorized pass."""
    if boundary != "wsr":
        thetas = np.linspace(0.0, 1.0, n_points)
        rejected = np.array([
            theta_rejected(Tj_history, T_history, float(th), alpha, B, boundary) for th in thetas
        ])
        return thetas, rejected
    grid = ShareCSGrid(alpha, B, grid_size=n_points)
    for tj, t in zip(Tj_history, T_history):
        grid.update(tj, t)
    rejected = grid._lower.rejected | grid._upper.rejected
    return grid.thetas, rejected


def _accepted_runs(rejected: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous (start_idx, end_idx) index ranges (inclusive) where
    `rejected` is False, i.e. accepted runs."""
    accepted = ~rejected
    runs = []
    i, n = 0, len(accepted)
    while i < n:
        if accepted[i]:
            j = i
            while j + 1 < n and accepted[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _bisect_to_boundary(rej, a: float, b: float, tol: float, accepted_side: str) -> float:
    """Bisects [a, b] where exactly one of the two endpoints is accepted
    (`accepted_side` says which), returning a point on the accepted side of
    the crossing, narrowed to `tol`. Does not assume which of a/b is larger
    in rejection status beyond what `accepted_side` states."""
    lo, hi = a, b
    for _ in range(80):
        mid = (lo + hi) / 2.0
        mid_rejected = rej(mid)
        if accepted_side == "b":
            if mid_rejected:
                lo = mid
            else:
                hi = mid
        else:
            if mid_rejected:
                hi = mid
            else:
                lo = mid
        if hi - lo < tol:
            break
    return hi if accepted_side == "b" else lo


def _nudge_to_accepted(rej, x: float, interior: float, tol: float) -> float:
    """Safety net: if root-search rounding leaves `x` marginally rejected,
    step toward a known-accepted interior point until accepted, or raise if
    that fails within a bounded number of steps (should not happen; this
    would indicate the accepted region is narrower than floating-point
    precision can resolve)."""
    if not rej(x):
        return x
    step = interior - x
    for frac in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0):
        candidate = x + frac * step
        if not rej(candidate):
            return candidate
    raise RuntimeError(
        "drift_test_cs: could not find an accepted endpoint near a claimed "
        "interval boundary within a bounded nudge search -- the accepted "
        "region may be narrower than floating-point precision can resolve; "
        "report as indeterminate rather than silently returning a rejected "
        "endpoint."
    )


def drift_test_cs(
    Tj_history: Sequence[float],
    T_history: Sequence[float],
    alpha: float,
    B: float,
    boundary: str = "wsr",
    tol: float = 1e-4,
    n_coarse: int = 2001,
    n_refine: int = 20001,
) -> DriftTestResult:
    """Drift diagnostic: inverts the fixed-mean WSR betting martingale over
    H0: mu_ij/mu_i == a_j (constant for all i).

    An empty accepted set is an anytime-valid level-alpha rejection of constant
    conditional share. It is not a certification mechanism.

    Procedure: scan theta on `n_coarse` points in [0, 1] and find the maximal
    contiguous accepted runs. With no run, rescan on `n_refine` points; if that
    also finds none, status is "empty". With exactly one run, refine both ends by
    bisection to `tol` and confirm each is accepted by `theta_rejected`
    (`_nudge_to_accepted` handles rounding). With several runs, status is
    "indeterminate". Verdicts rest on finite-resolution scans, not an analytic
    monotonicity proof.
    """
    if len(Tj_history) != len(T_history):
        raise ValueError("Tj_history and T_history must be the same length")
    _validated_mass(np.asarray(Tj_history, dtype=float), B, "drift_test_cs T_ij")
    _validated_mass(np.asarray(T_history, dtype=float), B, "drift_test_cs T_i")
    if not Tj_history:
        return DriftTestResult("interval", 0.0, 1.0, 0)

    def rej(theta: float) -> bool:
        return theta_rejected(Tj_history, T_history, theta, alpha, B, boundary)

    def resolve_at(n_points: int) -> Optional[DriftTestResult]:
        thetas, rejected = _drift_scan(Tj_history, T_history, alpha, B, boundary, n_points)
        runs = _accepted_runs(rejected)
        if len(runs) >= 2:
            return DriftTestResult("indeterminate", None, None, n_points)
        if len(runs) == 0:
            return None
        i0, i1 = runs[0]
        interior = (thetas[i0] + thetas[i1]) / 2.0
        lo = 0.0 if i0 == 0 else _bisect_to_boundary(rej, thetas[i0 - 1], thetas[i0], tol, accepted_side="b")
        hi = 1.0 if i1 == n_points - 1 else _bisect_to_boundary(rej, thetas[i1], thetas[i1 + 1], tol, accepted_side="a")
        lo = _nudge_to_accepted(rej, lo, interior, tol)
        hi = _nudge_to_accepted(rej, hi, interior, tol)
        return DriftTestResult("interval", lo, hi, n_points)

    result = resolve_at(n_coarse)
    if result is not None:
        return result

    result = resolve_at(n_refine)
    if result is not None:
        return result

    return DriftTestResult("empty", None, None, n_refine)


# Grid-based WSR share CS: benchmark infrastructure; the experiments use CumulativeShareCS.


class _GridSide:
    """Vectorized WSR betting wealth for one side (sign=+1 or -1), evaluated
    simultaneously at every theta on the grid -- the exact per-theta
    recursion of `_WSRSide.update`, just carrying a `(grid_size)` state
    instead of a scalar."""

    def __init__(self, m_tilde: np.ndarray, alpha_side: float, sign: int):
        self.m = m_tilde
        self.sign = sign
        G = len(m_tilde)
        self.t = 0
        self.sum_x = np.full(G, 0.5)
        self.sum_sq_dev = np.full(G, 0.25)
        self.log_wealth = np.zeros(G)
        self.max_log_wealth = np.zeros(G)
        self.rejected = np.zeros(G, dtype=bool)
        self._log_threshold = math.log(1.0 / alpha_side)

    def update(self, x_tilde: float) -> None:
        active = ~self.rejected
        if not active.any():
            self.t += 1
            return
        i = self.t + 1
        m = self.m
        mu_prev = self.sum_x / i
        var_prev = np.clip(self.sum_sq_dev / i, 1e-6, 0.25)
        lam_mag = np.zeros(len(m))
        if i > 1:
            log_inv_alpha = self._log_threshold
            lam_mag = np.sqrt(2.0 * log_inv_alpha / (var_prev * i * math.log(i + 1)))
        lam_mag = np.minimum(lam_mag, 1.0)
        lam = self.sign * lam_mag

        eps = 1e-9
        pos = (lam > 0) & (m > eps)
        lam = np.where(pos, np.minimum(lam, (1.0 - eps) / np.where(m > eps, m, 1.0)), lam)
        neg = (lam < 0) & (m < 1 - eps)
        lam = np.where(neg, np.maximum(lam, -(1.0 - eps) / np.where(m < 1 - eps, 1.0 - m, 1.0)), lam)

        factor = np.maximum(1.0 + lam * (x_tilde - m), 1e-300)
        new_log_wealth = np.where(active, self.log_wealth + np.log(factor), self.log_wealth)
        new_max_log_wealth = np.where(active, np.maximum(self.max_log_wealth, new_log_wealth), self.max_log_wealth)
        newly_rejected = active & (new_max_log_wealth >= self._log_threshold)

        self.log_wealth = new_log_wealth
        self.max_log_wealth = new_max_log_wealth
        self.rejected = self.rejected | newly_rejected
        self.sum_sq_dev = np.where(active, self.sum_sq_dev + (x_tilde - mu_prev) ** 2, self.sum_sq_dev)
        self.sum_x = np.where(active, self.sum_x + x_tilde, self.sum_x)
        self.t += 1


class ShareCSGrid:
    """Grid-based, incrementally updatable WSR confidence sequence for a_j = E[T_ij]/E[T_i].

    grid = ShareCSGrid(alpha, B)
    for tj, t in zip(Tj_history, T_history):
        grid.update(tj, t)
        lo, hi = grid.interval  # valid at every step
    """

    def __init__(self, alpha: float, B: float, grid_size: int = 4001):
        self.B = B
        self.thetas = np.linspace(0.0, 1.0, grid_size)
        m_tilde = self.thetas / (1.0 + self.thetas)
        alpha_side = alpha / 2.0
        self._lower = _GridSide(m_tilde, alpha_side, sign=+1)
        self._upper = _GridSide(m_tilde, alpha_side, sign=-1)

    def update(self, Tj_i: float, T_i: float) -> None:
        Tj_i = _validated_mass(Tj_i, self.B, "ShareCSGrid.update T_ij")
        T_i = _validated_mass(T_i, self.B, "ShareCSGrid.update T_i")
        thetas = self.thetas
        g = Tj_i - thetas * T_i
        denom = (1.0 + thetas) * self.B
        x_tilde = np.clip((g + thetas * self.B) / denom, 0.0, 1.0)
        self._lower.update(x_tilde)
        self._upper.update(x_tilde)

    def interval(self) -> tuple[float, float]:
        accepted = ~(self._lower.rejected | self._upper.rejected)
        idx = np.nonzero(accepted)[0]
        if len(idx) == 0:
            return 0.0, 1.0
        return float(self.thetas[idx[0]]), float(self.thetas[idx[-1]])


class _MultiGridSide:
    """Like `_GridSide`, but carries a (K, grid_size) state so K campaigns'
    wealth processes update in ONE vectorized numpy call per observation
    (needed for E7/E8's many intervention configs x replications,
    where K separate `ShareCSGrid`s' per-campaign Python-level call
    overhead dominates -- see MultiCampaignShareCSGrid)."""

    def __init__(self, K: int, m_tilde: np.ndarray, alpha_side: float, sign: int):
        self.m = m_tilde
        self.sign = sign
        G = len(m_tilde)
        self.t = 0
        self.sum_x = np.full((K, G), 0.5)
        self.sum_sq_dev = np.full((K, G), 0.25)
        self.log_wealth = np.zeros((K, G))
        self.max_log_wealth = np.zeros((K, G))
        self.rejected = np.zeros((K, G), dtype=bool)
        self._log_threshold = math.log(1.0 / alpha_side)

    def update(self, x_tilde: np.ndarray) -> None:
        """x_tilde: (K, G)."""
        active = ~self.rejected
        i = self.t + 1
        m = np.broadcast_to(self.m[None, :], x_tilde.shape)
        mu_prev = self.sum_x / i
        var_prev = np.clip(self.sum_sq_dev / i, 1e-6, 0.25)
        lam_mag = np.zeros_like(self.sum_x)
        if i > 1:
            log_inv_alpha = self._log_threshold
            lam_mag = np.sqrt(2.0 * log_inv_alpha / (var_prev * i * math.log(i + 1)))
        lam_mag = np.minimum(lam_mag, 1.0)
        lam = self.sign * lam_mag

        eps = 1e-9
        pos = (lam > 0) & (m > eps)
        lam = np.where(pos, np.minimum(lam, (1.0 - eps) / np.where(m > eps, m, 1.0)), lam)
        neg = (lam < 0) & (m < 1 - eps)
        lam = np.where(neg, np.maximum(lam, -(1.0 - eps) / np.where(m < 1 - eps, 1.0 - m, 1.0)), lam)

        factor = np.maximum(1.0 + lam * (x_tilde - m), 1e-300)
        new_log_wealth = np.where(active, self.log_wealth + np.log(factor), self.log_wealth)
        new_max_log_wealth = np.where(active, np.maximum(self.max_log_wealth, new_log_wealth), self.max_log_wealth)
        newly_rejected = active & (new_max_log_wealth >= self._log_threshold)

        self.log_wealth = new_log_wealth
        self.max_log_wealth = new_max_log_wealth
        self.rejected = self.rejected | newly_rejected
        self.sum_sq_dev = np.where(active, self.sum_sq_dev + (x_tilde - mu_prev) ** 2, self.sum_sq_dev)
        self.sum_x = np.where(active, self.sum_x + x_tilde, self.sum_x)
        self.t += 1


class MultiCampaignShareCSGrid:
    """`ShareCSGrid` for K campaigns at once, one vectorized (K, grid_size)
    update per observation instead of K independent `ShareCSGrid.update`
    calls -- same per-theta WSR recursion, batched over campaigns to
    amortize Python-level call overhead (performance need)."""

    def __init__(self, K: int, alpha: float, B: float, grid_size: int = 4001):
        self.K = K
        self.B = B
        self.thetas = np.linspace(0.0, 1.0, grid_size)
        m_tilde = self.thetas / (1.0 + self.thetas)
        alpha_side = alpha / 2.0
        self._lower = _MultiGridSide(K, m_tilde, alpha_side, sign=+1)
        self._upper = _MultiGridSide(K, m_tilde, alpha_side, sign=-1)

    def update(self, Tj_vec: np.ndarray, T_i: float) -> None:
        """Tj_vec: (K) masses for this observation, one per campaign;
        T_i: the shared total mass (same for every campaign's test)."""
        Tj_vec = _validated_mass(np.asarray(Tj_vec, dtype=float), self.B, "MultiCampaignShareCSGrid.update T_ij")
        T_i = _validated_mass(T_i, self.B, "MultiCampaignShareCSGrid.update T_i")
        thetas = self.thetas
        g = Tj_vec[:, None] - thetas[None, :] * T_i
        denom = (1.0 + thetas) * self.B
        x_tilde = np.clip((g + thetas[None, :] * self.B) / denom[None, :], 0.0, 1.0)
        self._lower.update(x_tilde)
        self._upper.update(x_tilde)

    def intervals(self) -> tuple[np.ndarray, np.ndarray]:
        accepted = ~(self._lower.rejected | self._upper.rejected)
        los = np.zeros(self.K)
        his = np.ones(self.K)
        for c in range(self.K):
            idx = np.nonzero(accepted[c])[0]
            if len(idx) == 0:
                continue
            los[c] = self.thetas[idx[0]]
            his[c] = self.thetas[idx[-1]]
        return los, his


def invert_share_cs_grid(
    Tj_history: Sequence[float],
    T_history: Sequence[float],
    alpha: float,
    B: float,
    grid_size: int = 4001,
) -> tuple[float, float]:
    """Non-incremental wrapper with the signature of `invert_share_cs`.

    For repeated calls on a growing prefix, use `ShareCSGrid` incrementally.
    """
    if len(Tj_history) != len(T_history):
        raise ValueError("Tj_history and T_history must be the same length")
    if not Tj_history:
        return 0.0, 1.0
    grid = ShareCSGrid(alpha, B, grid_size)
    for tj, t in zip(Tj_history, T_history):
        grid.update(tj, t)
    return grid.interval()


# Cumulative-share confidence sequence: Howard et al. (2021) Thm. 4 with the polynomial stitched boundary (eta=2,
# s=1.4, m=1, c=1).

_HOWARD_ETA = 2.0   # eq. (24)'s own choice
_HOWARD_S = 1.4     # eq. (24)'s own choice (h(k) ~ (k+1)^s, zeta(s))
_HOWARD_M = 1.0     # eq. (24)'s own choice (v v m = v v 1)
_HOWARD_ZETA_S = float(_scipy_zeta(_HOWARD_S))          # zeta(1.4)
_HOWARD_K1 = (_HOWARD_ETA ** 0.25 + _HOWARD_ETA ** -0.25) / math.sqrt(2.0)
_HOWARD_K2 = (math.sqrt(_HOWARD_ETA) + 1.0) / 2.0
_HOWARD_LOG_ETA_POW_S = math.log(_HOWARD_ETA) ** _HOWARD_S  # (log eta)^s, NOT log(eta^s)


def _howard_eb_L(v: float, alpha_used: float) -> float:
    """l(v) inside Howard et al. (2021) eq. (10): l(v) = s*loglog(eta*v/m) +
    log(zeta(s)/(alpha_used*(log eta)^s)), v already clamped to >= m by the
    caller."""
    return _HOWARD_S * math.log(math.log(_HOWARD_ETA * v / _HOWARD_M)) + math.log(
        _HOWARD_ZETA_S / (alpha_used * _HOWARD_LOG_ETA_POW_S)
    )


BOUNDARY_DEFAULT = "exact"  # production default: Howard et al. (2021) Thm. 1 eq. (8), proved at eq. (43)
_BOUNDARY_FORMS = ("exact", "simplified")


_ACTIVE_BOUNDARY = [BOUNDARY_DEFAULT]


class boundary_override:
    """Context manager that switches the default stitched-boundary form for every Howard-EB
    object CONSTRUCTED inside the block (the A/B/C/D controlled comparison
    re-runs identical streams under the "simplified" boundary). Production code never
    uses it; objects built without an explicit `boundary` take BOUNDARY_DEFAULT = "exact"."""

    def __init__(self, form: str):
        self.form = _check_form(form)

    def __enter__(self):
        self._prev = _ACTIVE_BOUNDARY[0]
        _ACTIVE_BOUNDARY[0] = self.form
        return self

    def __exit__(self, *exc):
        _ACTIVE_BOUNDARY[0] = self._prev
        return False


def _check_form(form: Optional[str]) -> str:
    if form is None:
        form = _ACTIVE_BOUNDARY[0]
    if form == "production":  # older name of the simplified form, kept as an alias
        form = "simplified"
    if form not in _BOUNDARY_FORMS:
        raise ValueError(f"boundary form must be one of {_BOUNDARY_FORMS}, got {form!r}")
    return form


def howard_stitched_S(v: float, alpha_used: float, form: Optional[str] = None) -> float:
    """Stitched boundary S_alpha(v) of Howard et al. (2021) with eta=2, s=1.4, m=1, c=1, for v >= m.

    form="exact": eq. (8) with the stitching bound of eq. (43),
    S = sqrt(k1^2 v l + k2^2 c^2 l^2) + c k2 l.
    form="simplified": the upper bound S <= k1 sqrt(v l) + 2 c k2 l. Since
    sqrt(a^2+b^2) <= a+b, exact <= simplified pointwise, so both are valid.
    """
    form = _check_form(form)
    L = _howard_eb_L(v, alpha_used)
    if form == "simplified":
        return _HOWARD_K1 * math.sqrt(v * L) + 2.0 * _HOWARD_K2 * L
    return math.sqrt(_HOWARD_K1 ** 2 * v * L + (_HOWARD_K2 * L) ** 2) + _HOWARD_K2 * L


def howard_eb_radius(V_hat: float, t: int, alpha_used: float, form: Optional[str] = None) -> float:
    """Confidence radius r_t = S_alpha_used(max(V_hat, m)) / t for the empirical-Bernstein bound (Thm. 4).

    `alpha_used` is the boundary crossing probability of one process: the
    two-sided miscoverage is 2*alpha_used, so pass beta/2 for two-sided level beta.
    """
    if t <= 0:
        return 1.0
    v = max(V_hat, _HOWARD_M)
    return howard_stitched_S(v, alpha_used, form) / t


class HowardEBProcess:
    """Incremental empirical-Bernstein confidence sequence for the running average
    mu_t = (1/t) sum_{i<=t} E[X_i | F_{i-1}] of a predictable-mean process in [0,1].

    The predictable forecast is the lagged running mean, with 0.5 as the first
    forecast. The boundary is closed-form in the empirical variance process.
    """

    def __init__(self, alpha_used: float, form: Optional[str] = None):
        self.alpha_used = alpha_used
        self.form = _check_form(form)
        self.t = 0
        self.sum_x = 0.0
        self.V_hat = 0.0
        self._forecast = 0.5

    def update(self, x: float) -> None:
        x = _validated_unit(x, "HowardEBProcess.update")
        self.V_hat += (x - self._forecast) ** 2
        self.sum_x += x
        self.t += 1
        self._forecast = self.sum_x / self.t

    def interval(self) -> tuple[float, float]:
        if self.t == 0:
            return 0.0, 1.0
        mean = self.sum_x / self.t
        r = howard_eb_radius(self.V_hat, self.t, self.alpha_used, self.form)
        return max(mean - r, 0.0), min(mean + r, 1.0)


class MultiHowardEBProcess:
    """K independent `HowardEBProcess` sequences (one per campaign numerator) batched into vectorized numpy state.

    Updates all K campaigns in one call per observation; no state is shared between campaigns.
    """

    def __init__(self, K: int, alpha_used: float, form: Optional[str] = None):
        self.K = K
        self.alpha_used = alpha_used
        self.form = _check_form(form)
        self.t = 0
        self.sum_x = np.zeros(K)
        self.V_hat = np.zeros(K)
        self._forecast = np.full(K, 0.5)

    def update(self, x_vec: np.ndarray) -> None:
        x_vec = _validated_unit(x_vec, "MultiHowardEBProcess.update")
        self.V_hat += (x_vec - self._forecast) ** 2
        self.sum_x += x_vec
        self.t += 1
        self._forecast = self.sum_x / self.t

    def intervals(self) -> tuple[np.ndarray, np.ndarray]:
        if self.t == 0:
            return np.zeros(self.K), np.ones(self.K)
        mean = self.sum_x / self.t
        r = np.array([howard_eb_radius(float(v), self.t, self.alpha_used, self.form) for v in self.V_hat])
        return np.clip(mean - r, 0.0, 1.0), np.clip(mean + r, 0.0, 1.0)


class MultiCampaignCumulativeShareCS:
    """Cumulative-share confidence sequence for K campaigns (K >= 1).

    K `HowardEBProcess` numerator sequences (vectorized in `MultiHowardEBProcess`)
    share one denominator sequence, each at per-process error beta = alpha/(K+1)
    and boundary crossing probability beta/2. Each interval is a closed-form ratio
    of two empirical-Bernstein intervals; it contains the empirical cumulative
    share and is never empty once L^D_t > 0.
    """

    def __init__(self, K: int, alpha: float, B: float, boundary: Optional[str] = None):
        self.K = K
        self.B = B
        self.alpha = alpha
        self.boundary = _check_form(boundary)
        self.beta = alpha / (K + 1)
        self.alpha_used = self.beta / 2.0
        self._num = MultiHowardEBProcess(K, self.alpha_used, self.boundary)
        self._den = HowardEBProcess(self.alpha_used, self.boundary)
        self.n_processes = K + 1
        self.untracked_mass_seen = False

    def update(self, Tj_vec: np.ndarray, T_i: float) -> None:
        """Tj_vec: (K) masses for this observation, one per campaign;
        T_i: the shared total mass (same for every campaign's numerator).
        Raises `MassBoundError` if any mass is non-finite, negative, or
        exceeds B (Assumption a:cap); nothing is clipped. Validation happens
        before either sub-process is touched, so a rejected observation
        leaves the sequence state unchanged."""
        Tj_vec = np.asarray(Tj_vec, dtype=float)
        x_num = _validated_unit(Tj_vec / self.B, "MultiCampaignCumulativeShareCS numerator T_ij/B")
        x_den = _validated_unit(T_i / self.B, "MultiCampaignCumulativeShareCS denominator T_i/B")
        self._num.update(x_num)
        self._den.update(x_den)
        if float(T_i) - float(Tj_vec.sum()) > _MASS_TOL * self.B:
            self.untracked_mass_seen = True

    def numerator_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        """Normalised (units of 1/B) confidence boxes [L^N_m, U^N_m] for the K numerator running means.

        Simultaneously valid with the denominator box at budget beta = alpha/(K+1)
        per process. Truncation to [0, 1] intersects with the known support; it is
        not an observation clip. Before any observation: ([0]*K, [1]*K).
        """
        return self._num.intervals()

    def intervals(self) -> tuple[np.ndarray, np.ndarray]:
        LN, UN = self._num.intervals()
        LD, UD = self._den.interval()
        if LD <= 0.0:
            return np.zeros(self.K), np.ones(self.K)
        L = np.clip(LN / UD, 0.0, 1.0)
        U = np.clip(UN / LD, 0.0, 1.0)
        return L, U


class DirectNumeratorCS:
    """Numerator-only confidence sequences for direct-numerator certification.

    The gap g_jk(z) = (z_j - z_k)/sum(z) is degree-0 homogeneous, so the common
    denominator cancels (see `certify.direct_lower_gap`) and only numerator boxes
    are needed. Union-bound allocation, two-sided crossing probability beta/2 each:
      * include_other=False: K processes, beta = alpha/K. Valid for the share within
        the tracked pool (`pool="tracked"`) or when T_i = sum_j T_ij.
      * include_other=True: K+1 processes (K campaigns plus an "other" bucket
        T_iO = T_i - sum_j T_ij >= 0), beta = alpha/(K+1). Required for the
        full-pool estimand with delta > 0 when untracked mass exists.
    No dashboard-share intervals: use `MultiCampaignCumulativeShareCS` for those.
    """

    def __init__(self, K: int, alpha: float, B: float, include_other: bool = False,
                 boundary: Optional[str] = None):
        if K < 1:
            raise ValueError("K must be >= 1")
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0, 1)")
        if not (B > 0 and math.isfinite(B)):
            raise ValueError("B must be positive and finite")
        self.K, self.B, self.alpha, self.include_other = K, B, alpha, include_other
        self.n_processes = K + 1 if include_other else K
        self.beta = alpha / self.n_processes
        self.alpha_used = self.beta / 2.0
        self.boundary = _check_form(boundary)
        self._num = MultiHowardEBProcess(self.n_processes, self.alpha_used, self.boundary)
        self.untracked_mass_seen = False

    def update(self, Tj_vec: np.ndarray, T_i: Optional[float] = None) -> None:
        """Tj_vec: (K) tracked masses. T_i: total mass of the conversion; needed
        only when include_other (then T_iO = T_i - sum_j T_ij must be >= 0, else
        MassBoundError). With include_other=False a T_i that exceeds the tracked
        sum is recorded in `untracked_mass_seen` (it does not affect the
        tracked-pool estimand). Validation precedes any state change."""
        Tj_vec = np.asarray(Tj_vec, dtype=float)
        if Tj_vec.shape != (self.K,):
            raise ValueError(f"Tj_vec must have shape ({self.K},)")
        _validated_mass(Tj_vec, self.B, "DirectNumeratorCS numerator T_ij")
        vec = Tj_vec
        if T_i is not None:
            _validated_mass(T_i, self.B, "DirectNumeratorCS total T_i")
            other = float(T_i) - float(Tj_vec.sum())
            if other < -_MASS_TOL * self.B:
                raise MassBoundError(
                    f"DirectNumeratorCS: T_i={T_i!r} is below sum_j T_ij={Tj_vec.sum()!r}; "
                    "tracked masses cannot exceed the total"
                )
            other = max(other, 0.0)  # rounding slack only (|violation| <= _MASS_TOL*B, checked above)
            if other > _MASS_TOL * self.B:
                self.untracked_mass_seen = True
            if self.include_other:
                vec = np.append(Tj_vec, other)
        elif self.include_other:
            raise ValueError("include_other=True requires T_i")
        self._num.update(vec / self.B)

    def numerator_boxes(self) -> tuple[np.ndarray, np.ndarray]:
        """Normalised boxes for the K tracked numerators (first K entries)."""
        L, U = self._num.intervals()
        return L[: self.K], U[: self.K]

    def other_box(self) -> Optional[tuple[float, float]]:
        """Normalised box [L^N_O, U^N_O] for the "other" bucket, or None."""
        if not self.include_other:
            return None
        L, U = self._num.intervals()
        return float(L[self.K]), float(U[self.K])
