"""Robust and naive certification rules, certified graph and Gamma*.

`robust_rule` and `gamma_star` are guaranteed for additive fixed-weight rules
(linear, timestamp-fixed time-decay) given valid confidence sequences.
`gamma_hat_sens` is an empirical re-thinning envelope for models without an
identification guarantee (last-touch, logistic+SHAP, DARNN); never report it as Gamma*.
"""
from __future__ import annotations

import math
from enum import Enum
from typing import Callable, NamedTuple, Optional, Sequence

import numpy as np

from .identification import gamma_eff, gap_vertex_extrema
from .journeys import Journey, thin


class Certification(Enum):
    J_OVER_K = "cert(j>k)"
    K_OVER_J = "cert(k>j)"
    UNRESOLVED = "unresolved"


def robust_lower_gap(L: Sequence[float], U: Sequence[float], Gamma: float, j: int, k: int) -> float:
    """Delta_jk,t(Gamma): inf of the gap objective over the box prod_m [L_m, Gamma*U_m] (vertex reduction)."""
    L = np.asarray(L, dtype=float)
    U = np.asarray(U, dtype=float)
    g_min, _ = gap_vertex_extrema(L, Gamma * U, j, k)
    return g_min


def robust_rule(L: Sequence[float], U: Sequence[float], Gamma: float, delta: float, j: int, k: int) -> Certification:
    """certify(j>k) iff Delta_jk,t(Gamma) > delta; certify(k>j) iff Delta_kj,t(Gamma) > delta; else UNRESOLVED.

    For delta=0 this reduces to L_t^j / U_t^k > Gamma.
    """
    if robust_lower_gap(L, U, Gamma, j, k) > delta:
        return Certification.J_OVER_K
    if robust_lower_gap(L, U, Gamma, k, j) > delta:
        return Certification.K_OVER_J
    return Certification.UNRESOLVED


# Direct-numerator certification: g_jk is degree-0 homogeneous in z, so the common denominator cancels and the gap
# bound is the minimum over the numerator box. delta is an absolute share gap over the pool in the denominator sum, so
# the pool matters (see `pool`).

class UntrackedMassError(ValueError):
    """delta > 0 with a full-pool estimand and untracked mass but no "other" numerator process.

    Dropping the other mass from the denominator would inflate the share gap (anti-conservative).
    """


def _check_boxes(LN, UN, what: str):
    LN = np.asarray(LN, dtype=float)
    UN = np.asarray(UN, dtype=float)
    if LN.shape != UN.shape or LN.ndim != 1:
        raise ValueError(f"{what}: boxes must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(LN)) and np.all(np.isfinite(UN))):
        raise ValueError(f"{what}: non-finite box endpoint")
    if np.any(LN < 0) or np.any(UN < LN):
        raise ValueError(f"{what}: need 0 <= L <= U")
    return LN, UN


def direct_lower_gap(
    LN: Sequence[float],
    UN: Sequence[float],
    Gamma: float,
    j: int,
    k: int,
    *,
    p_lo: Optional[float] = None,
    other: Optional[tuple[float, float]] = None,
    pool: str = "tracked",
    untracked_mass: bool = False,
    delta: float = 0.0,
) -> float:
    """Direct-numerator lower bound on Delta_jk.

    Minimum of g_jk over prod_m [L^N_m, Gamma_eff U^N_m] (plus the "other" box if
    given), by vertex reduction. LN, UN are the K tracked numerator boxes in any
    common unit (the objective is scale-free).

    pool="tracked": the estimand is the share within the K tracked campaigns;
        untracked mass is ignored.
    pool="full": the denominator includes untracked mass. For delta > 0 an `other`
        box is required if `untracked_mass` is True (raises UntrackedMassError);
        at delta = 0 only the sign of z_j - z_k matters.
    Returns -inf (not certifiable) when the objective is undefined or a required
    upper endpoint is zero. p_lo applies the floor via Gamma_eff = min(Gamma, 1/p_lo).
    """
    if pool not in ("tracked", "full"):
        raise ValueError("pool must be 'tracked' or 'full'")
    if Gamma < 1:
        raise ValueError("Gamma must be >= 1")
    G = Gamma if p_lo is None else gamma_eff(Gamma, p_lo)
    LN, UN = _check_boxes(LN, UN, "direct_lower_gap")
    if pool == "full" and untracked_mass and other is None and delta > 0:
        raise UntrackedMassError(
            "delta > 0 with a full-pool estimand and untracked campaign mass requires an explicit "
            "'other' numerator process (DirectNumeratorCS(include_other=True)); refusing to drop it"
        )
    l = LN.copy()
    h = G * UN
    if other is not None:
        lo_o, hi_o = float(other[0]), float(other[1])
        if not (np.isfinite(lo_o) and np.isfinite(hi_o)) or lo_o < 0 or hi_o < lo_o:
            raise ValueError("invalid 'other' box")
        l = np.append(l, lo_o)
        h = np.append(h, G * hi_o)
    if h[j] <= 0.0 or h[k] <= 0.0:
        return float("-inf")  # Lemma 1 needs h_j, h_k > 0; zero upper endpoint -> no certificate
    g_min, _ = gap_vertex_extrema(l, h, j, k)
    return float(g_min)


def direct_robust_rule(
    LN: Sequence[float],
    UN: Sequence[float],
    Gamma: float,
    delta: float,
    j: int,
    k: int,
    **kwargs,
) -> "Certification":
    """certify(j>k) iff direct_lower_gap(j,k) > delta; certify(k>j) iff direct_lower_gap(k,j) > delta; else UNRESOLVED.

    At delta = 0 this is L^N_j / U^N_k > Gamma_eff.
    """
    if direct_lower_gap(LN, UN, Gamma, j, k, delta=delta, **kwargs) > delta:
        return Certification.J_OVER_K
    if direct_lower_gap(LN, UN, Gamma, k, j, delta=delta, **kwargs) > delta:
        return Certification.K_OVER_J
    return Certification.UNRESOLVED


def direct_gamma_star(
    LN: Sequence[float],
    UN: Sequence[float],
    delta: float,
    j: int,
    k: int,
    gamma_max: float = 1000.0,
    tol: float = 1e-4,
    **kwargs,
) -> Optional[float]:
    """Gamma* under the direct-numerator rule: sup{Gamma >= 1: lower gap > delta}.

    The lower gap is non-increasing in Gamma, so bisect. Ratio-only unless `p_lo` is
    passed through kwargs, in which case it saturates at 1/p_lo (see
    `identification.gamma_star_floor_saturated`). Returns None if not certified at Gamma = 1.
    """
    if direct_lower_gap(LN, UN, 1.0, j, k, delta=delta, **kwargs) <= delta:
        return None
    if direct_lower_gap(LN, UN, gamma_max, j, k, delta=delta, **kwargs) > delta:
        return gamma_max
    a, b = 1.0, gamma_max
    for _ in range(60):
        mid = (a + b) / 2
        if direct_lower_gap(LN, UN, mid, j, k, delta=delta, **kwargs) > delta:
            a = mid
        else:
            b = mid
        if b - a < tol:
            break
    return a


def hybrid_lower_gap(
    LN: Sequence[float],
    UN: Sequence[float],
    L_ratio: Sequence[float],
    U_ratio: Sequence[float],
    Gamma: float,
    j: int,
    k: int,
    *,
    p_lo: Optional[float] = None,
    other: Optional[tuple[float, float]] = None,
    pool: str = "tracked",
    untracked_mass: bool = False,
    delta: float = 0.0,
) -> float:
    """Hybrid lower bound max(L_direct, L_legacy_clipped) on the tracked-pool gap Delta_jk,t = A_j - A_k.

    Inputs come from one `MultiCampaignCumulativeShareCS` (K numerators plus the
    shared denominator at beta = alpha/(K+1)): (LN, UN) = numerator_boxes,
    (L_ratio, U_ratio) = intervals. On the event that all K+1 processes cover
    (probability >= 1-alpha) both components bound the same quantity, so their
    maximum is valid with no extra budget.
    Both use the K tracked campaigns as the pool, so `other` is not supported
    (NotImplementedError), and the full-pool/untracked/delta>0 guard of
    `direct_lower_gap` applies to the hybrid as a whole. p_lo applies the floor
    to both components.
    """
    if other is not None:
        raise NotImplementedError("hybrid rule supports the tracked-pool estimand only (no 'other' bucket)")
    d = direct_lower_gap(LN, UN, Gamma, j, k, p_lo=p_lo, other=None, pool=pool,
                         untracked_mass=untracked_mass, delta=delta)
    G = Gamma if p_lo is None else gamma_eff(Gamma, p_lo)
    Lr, Ur = _check_boxes(L_ratio, U_ratio, "hybrid_lower_gap ratio box")
    if np.any(Ur > 1.0 + 1e-12):
        raise ValueError("ratio box must be the [0,1]-clipped box from cs.intervals()")
    leg = robust_lower_gap(Lr, Ur, G, j, k) if (Ur[j] > 0 and Ur[k] > 0) else float("-inf")
    return float(max(d, leg))


def hybrid_robust_rule(LN, UN, L_ratio, U_ratio, Gamma, delta, j, k, **kwargs) -> "Certification":
    """Certify with the hybrid lower bound (valid at the shared alpha/(K+1))."""
    if hybrid_lower_gap(LN, UN, L_ratio, U_ratio, Gamma, j, k, delta=delta, **kwargs) > delta:
        return Certification.J_OVER_K
    if hybrid_lower_gap(LN, UN, L_ratio, U_ratio, Gamma, k, j, delta=delta, **kwargs) > delta:
        return Certification.K_OVER_J
    return Certification.UNRESOLVED


def hybrid_gamma_star(
    LN, UN, L_ratio, U_ratio, delta: float, j: int, k: int,
    gamma_max: float = 1000.0, tol: float = 1e-4, **kwargs,
) -> Optional[float]:
    """Gamma* under the hybrid rule.

    Each component's lower gap is non-increasing in Gamma, so is their maximum, and
    {Gamma: max > delta} is the union of the component down-sets:
    Gamma*_hybrid = max(Gamma*_direct, Gamma*_legacy_clipped).
    """
    f = lambda G: hybrid_lower_gap(LN, UN, L_ratio, U_ratio, G, j, k, delta=delta, **kwargs)
    if f(1.0) <= delta:
        return None
    if f(gamma_max) > delta:
        return gamma_max
    a, b = 1.0, gamma_max
    for _ in range(60):
        mid = (a + b) / 2
        if f(mid) > delta:
            a = mid
        else:
            b = mid
        if b - a < tol:
            break
    return a


def certify_pair(
    cs,
    Gamma: float,
    delta: float,
    j: int,
    k: int,
    *,
    method: Optional[str] = None,
    p_lo: Optional[float] = None,
    pool: str = "tracked",
) -> "Certification":
    """Production certification entry point.

    method=None uses the default: "hybrid" for a `MultiCampaignCumulativeShareCS`,
    "direct" for a `DirectNumeratorCS`. "hybrid" needs a
    `MultiCampaignCumulativeShareCS`; "direct" applies the direct-numerator rule to
    `numerator_boxes` (plus the other bucket when `cs` carries one);
    "legacy_ratio_box" runs `robust_rule(*cs.intervals)` for comparison and ignores
    p_lo and pool.
    """
    if method is None:  # hybrid is the production default whenever K+1 shared processes exist
        method = "hybrid" if hasattr(cs, "intervals") else "direct"
    if method == "legacy_ratio_box":
        L, U = cs.intervals()
        return robust_rule(L, U, Gamma, delta, j, k)
    if method == "hybrid":
        LN, UN = cs.numerator_boxes()
        L, U = cs.intervals()
        return hybrid_robust_rule(
            LN, UN, L, U, Gamma, delta, j, k, p_lo=p_lo, pool=pool,
            untracked_mass=bool(getattr(cs, "untracked_mass_seen", False)),
        )
    if method != "direct":
        raise ValueError("method must be 'direct', 'hybrid' or 'legacy_ratio_box'")
    LN, UN = cs.numerator_boxes()
    other = cs.other_box() if hasattr(cs, "other_box") else None
    return direct_robust_rule(
        LN, UN, Gamma, delta, j, k, p_lo=p_lo, other=other, pool=pool,
        untracked_mass=bool(getattr(cs, "untracked_mass_seen", False)),
    )


# Method switch shared by every production driver.
PRODUCTION_CERT_METHOD = "hybrid"
CERT_METHODS = ("hybrid", "legacy_ratio_box")


class BoxSet(NamedTuple):
    """Confidence boxes of one `MultiCampaignCumulativeShareCS` at one time t.

    The [0,1]-clipped ratio box (L, U) and the K numerator boxes (LN, UN), all valid
    simultaneously at alpha/(K+1) per process.
    """
    L: np.ndarray
    U: np.ndarray
    LN: np.ndarray
    UN: np.ndarray


def snapshot(cs) -> BoxSet:
    L, U = cs.intervals()
    LN, UN = cs.numerator_boxes()
    return BoxSet(np.asarray(L, float), np.asarray(U, float), np.asarray(LN, float), np.asarray(UN, float))


class cert_method_override:
    """Scoped switch of the default certification method used by the `box_*` helpers (for controlled comparisons)."""

    def __init__(self, method: str):
        if method not in CERT_METHODS:
            raise ValueError(f"method must be one of {CERT_METHODS}")
        self.method = method

    def __enter__(self):
        global PRODUCTION_CERT_METHOD
        self._prev, PRODUCTION_CERT_METHOD = PRODUCTION_CERT_METHOD, self.method
        return self

    def __exit__(self, *exc):
        global PRODUCTION_CERT_METHOD
        PRODUCTION_CERT_METHOD = self._prev
        return False


def active_cert_method() -> str:
    return PRODUCTION_CERT_METHOD


def _method(method: Optional[str]) -> str:
    method = PRODUCTION_CERT_METHOD if method is None else method
    if method not in CERT_METHODS:
        raise ValueError(f"method must be one of {CERT_METHODS}")
    return method


def box_lower_gap(b: BoxSet, Gamma: float, j: int, k: int, *, method: Optional[str] = None,
                  p_lo: Optional[float] = None, delta: float = 0.0) -> float:
    """Lower gap Delta_jk,t(Gamma) under the chosen method (default hybrid); "legacy_ratio_box" ignores p_lo."""
    if _method(method) == "legacy_ratio_box":
        return robust_lower_gap(b.L, b.U, Gamma, j, k)
    return hybrid_lower_gap(b.LN, b.UN, b.L, b.U, Gamma, j, k, p_lo=p_lo, delta=delta)


def box_rule(b: BoxSet, Gamma: float, delta: float, j: int, k: int, *, method: Optional[str] = None,
             p_lo: Optional[float] = None) -> "Certification":
    if box_lower_gap(b, Gamma, j, k, method=method, p_lo=p_lo, delta=delta) > delta:
        return Certification.J_OVER_K
    if box_lower_gap(b, Gamma, k, j, method=method, p_lo=p_lo, delta=delta) > delta:
        return Certification.K_OVER_J
    return Certification.UNRESOLVED


def box_gamma_star(b: BoxSet, delta: float, j: int, k: int, *, method: Optional[str] = None,
                   gamma_max: float = 1000.0, tol: float = 1e-4, p_lo: Optional[float] = None) -> Optional[float]:
    """Gamma* matching `box_rule` (same method and p_lo)."""
    if _method(method) == "legacy_ratio_box":
        return gamma_star(b.L, b.U, delta, j, k, gamma_max=gamma_max, tol=tol)
    return hybrid_gamma_star(b.LN, b.UN, b.L, b.U, delta, j, k, gamma_max=gamma_max, tol=tol, p_lo=p_lo)


def naive_certify(lo: float, hi: float, delta: float) -> Certification:
    """Baseline: certify(j>k) iff lo > delta, certify(k>j) iff hi < -delta, else UNRESOLVED.

    `[lo, hi]` is a pairwise CS on a_j - a_k (or q_j - q_k) at alpha/C(K,2). Valid for
    the observed-data gap only; it can confidently reverse the true ordering.
    """
    if lo > delta:
        return Certification.J_OVER_K
    if hi < -delta:
        return Certification.K_OVER_J
    return Certification.UNRESOLVED


def certified_graph(
    certifications: dict[tuple[int, int], Certification],
    n_campaigns: int,
) -> dict[int, set[int]]:
    """Directed graph with an edge j -> k whenever CERT(j>k) has been issued."""
    graph: dict[int, set[int]] = {c: set() for c in range(n_campaigns)}
    for (j, k), cert in certifications.items():
        if cert == Certification.J_OVER_K:
            graph[j].add(k)
        elif cert == Certification.K_OVER_J:
            graph[k].add(j)
    return graph


def is_acyclic(graph: dict[int, set[int]]) -> bool:
    """DFS cycle detection. The certified graph is acyclic on the simultaneous-coverage event; this checks the property,
    it does not enforce it.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in graph}

    def visit(node: int) -> bool:
        color[node] = GRAY
        for nxt in graph.get(node, ()):
            c = color.get(nxt, WHITE)
            if c == GRAY:
                return False
            if c == WHITE and not visit(nxt):
                return False
        color[node] = BLACK
        return True

    return all(visit(node) for node in list(graph) if color[node] == WHITE)


def gamma_star(
    L: Sequence[float],
    U: Sequence[float],
    delta: float,
    j: int,
    k: int,
    gamma_max: float = 1000.0,
    tol: float = 1e-4,
) -> Optional[float]:
    """Gamma*_jk,t = sup{Gamma >= 1: robust rule certifies j > k}.

    Delta_jk,t(Gamma) is non-increasing in Gamma, so bisect. Returns None if not
    certified at Gamma=1.
    """
    if robust_lower_gap(L, U, 1.0, j, k) <= delta:
        return None
    if robust_lower_gap(L, U, gamma_max, j, k) > delta:
        return gamma_max
    a, b = 1.0, gamma_max
    for _ in range(60):
        mid = (a + b) / 2
        if robust_lower_gap(L, U, mid, j, k) > delta:
            a = mid
        else:
            b = mid
        if b - a < tol:
            break
    return a


def gamma_hat_sens(
    journeys: Sequence[Journey],
    credit_fn: Callable[[Journey], np.ndarray],
    n_campaigns: int,
    j: int,
    k: int,
    gamma_grid: Sequence[float],
    p_lo: float,
    rng: np.random.Generator,
    n_resample: int = 1,
) -> dict[float, float]:
    """Empirical sensitivity envelope for models without an identification guarantee.

    Re-thins held-out journeys at retention ratios up to Gamma and records the
    extremal shift in the credit difference Z_j - Z_k. Carries no coverage guarantee;
    never plot as Gamma*.
    """
    base_diffs = [credit_fn(journey)[j] - credit_fn(journey)[k] for journey in journeys]
    base_mean = float(np.mean(base_diffs)) if base_diffs else 0.0

    shifts: dict[float, float] = {}
    for Gamma in gamma_grid:
        p_hi = min(1.0, Gamma * p_lo)
        p = {c: p_lo for c in range(n_campaigns)}
        p[j] = p_hi
        diffs = []
        for _ in range(n_resample):
            for journey in journeys:
                obs = thin(journey, p, rng)
                if obs is None:
                    continue
                z = credit_fn(obs)
                diffs.append(z[j] - z[k])
        shifts[Gamma] = abs(float(np.mean(diffs)) - base_mean) if diffs else float("nan")
    return shifts
