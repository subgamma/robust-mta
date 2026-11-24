"""Independent checks of proof steps and counterexamples.

Standalone: imports nothing from the production package, reads no data and writes no files. Each check prints
its result. The checks illustrate or confirm specific steps and counterexamples; they are not proofs.

Run: python scripts/proof_checks/independent_proof_audit_checks.py
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction as F

import numpy as np

RNG = np.random.default_rng(20261120)


def theta(a: float, G: float) -> tuple[float, float]:
    """Theorem 1(ii) closed form."""
    return a / (G - (G - 1) * a), G * a / (1 + (G - 1) * a)


def gvec(z, j, k):
    s = float(np.sum(z))
    return (z[j] - z[k]) / s


def lemma2_min(l, h, j, k):
    """Lemma 2 vertex rule for inf of g_jk over prod [l_m, h_m]."""
    z = np.array(l, float).copy()
    z[k] = h[k]
    num = l[j] - h[k]
    for m in range(len(l)):
        if m not in (j, k):
            z[m] = h[m] if num > 0 else l[m]
    return gvec(z, j, k)


def brute_min(l, h, j, k):
    best = math.inf
    for v in itertools.product(*[(l[m], h[m]) for m in range(len(l))]):
        if sum(v) > 0:
            best = min(best, gvec(np.array(v), j, k))
    return best


# ---------------------------------------------------------------- Lemma 2
def check_lemma2(n=20000):
    worst = 0.0
    for _ in range(n):
        K = RNG.integers(2, 6)
        l = RNG.uniform(0, 1, K) * (RNG.uniform(size=K) < 0.7)  # some zero lower ends
        h = l + RNG.uniform(0, 1, K)
        j, k = RNG.choice(K, 2, replace=False)
        v = lemma2_min(l, h, j, k)
        b = brute_min(l, h, j, k)
        # interior points never go below the vertex minimum
        for _ in range(5):
            z = RNG.uniform(l, h)
            if z.sum() > 0:
                assert gvec(z, j, k) >= v - 1e-12
        worst = max(worst, abs(v - b))
    print(f"[Lemma 2] vertex rule vs all 2^K vertices, {n} random boxes: max |diff| = {worst:.2e}")


# ---------------------------------------------------------------- Theorem 1
def check_thm1_closed_forms(n=5000):
    worst = 0.0
    for _ in range(n):
        K = RNG.integers(2, 7)
        a = RNG.dirichlet(np.ones(K))
        G = RNG.uniform(1, 4)
        j = 0
        lo, hi = theta(a[j], G)
        # vertex extremes
        u = np.full(K, G); u[j] = 1.0
        A_lo = a[j] * u[j] / (a @ u)
        u = np.ones(K); u[j] = G
        A_hi = a[j] * u[j] / (a @ u)
        worst = max(worst, abs(A_lo - lo), abs(A_hi - hi))
        for _ in range(20):
            u = RNG.uniform(1, G, K)
            A = a[j] * u[j] / (a @ u)
            assert lo - 1e-12 <= A <= hi + 1e-12
    print(f"[Thm 1(ii)] closed form vs vertex evaluation: max |diff| = {worst:.2e}; random interior u inside")
    a = np.array([0.4, 0.3, 0.3])
    print("[Thm 1(iii)] example Gamma=2, p_lo=0.8: Theta_1(1.25) = (%.4f, %.4f), Theta_1(2) = (%.4f, %.4f)"
          % (*theta(0.4, 1.25), *theta(0.4, 2.0)))


def check_thm1_full_law_counterexamples():
    """Identified set w.r.t. the FULL observed law can be strictly smaller than
    Theta_j, and Theorem 1(v)'s 'no sample size can establish it' can fail.

    C1 (ratio-only model, count cap n_max = 1, so every capped latent journey
    has exactly one touchpoint, S = 1 = B; linear rule): observed law =
    P(single retained touch on j) = o_j, on k = o_k, zero-mass conversion = pi0
    (observable on the conversion-arrival clock). Then A_j p_j = o_j and
    (1 - A_j) p_k = o_k exactly, with p in (0,1]^2 and p_j/p_k in [1/G, G].
    C2 (floored model, any within-journey dependence): if no observed journey
    ever has two or more retained touchpoints, then for every latent journey
    E[#retained | x*] = sum_l p_{c_l} <= 1 while each p >= p_lo = 0.8, so every
    latent journey has a single touchpoint, and the same equations apply."""

    def feasible_A(o_j, o_k, G, p_lo, grid=2_000_001):
        A = np.linspace(1e-9, 1 - 1e-9, grid)
        pj = o_j / A
        pk = o_k / (1 - A)
        ok = (pj <= 1) & (pk <= 1) & (pj >= p_lo) & (pk >= p_lo)
        r = pj / pk
        ok &= (r <= G + 1e-12) & (r >= 1 / G - 1e-12)
        return A[ok].min(), A[ok].max()

    # C1
    o_j, o_k, G = 0.55, 0.30, 2.0
    a_j = o_j / (o_j + o_k)
    lo, hi = feasible_A(o_j, o_k, G, 0.0)
    print(f"[Thm 1, C1 ratio-only, n_max=1] o=({o_j},{o_k}), pi0={1-o_j-o_k:.2f}, a_j={a_j:.4f}, "
          f"a_j/a_k={o_j/o_k:.4f} <= Gamma={G}")
    print(f"    Theta_j(2) = ({theta(a_j, G)[0]:.4f}, {theta(a_j, G)[1]:.4f});"
          f" full-law identified set = [{lo:.4f}, {hi:.4f}] (exact: [0.55, 0.70])")
    print(f"    every admissible structure has A_j >= 0.55 > 0.5 > A_k: ordering IS identified")
    # C2
    G, p_lo = 2.0, 0.8
    Ge = min(G, 1 / p_lo)
    for o_j, o_k in [(0.48, 0.42), (0.52, 0.43)]:
        a_j = o_j / (o_j + o_k)
        lo, hi = feasible_A(o_j, o_k, G, p_lo)
        print(f"[Thm 1, C2 floored p_lo=0.8, Gamma_eff=1.25] o=({o_j},{o_k}), a_j/a_k={o_j/o_k:.4f}: "
              f"Theta_j(1.25)=({theta(a_j, Ge)[0]:.4f},{theta(a_j, Ge)[1]:.4f}), "
              f"full-law set=[{lo:.4f},{hi:.4f}]")
    # C1 depends on the cap: with n_max = 10 and dependent retention the same
    # observed law is reproduced with A_j < 1/2 (p_j = 0.2, p_k = 0.1).
    # type A: 5 j-touches, retain exactly one (marginal 0.2 each), prob 0.55
    # type B: 10 k-touches, retain exactly one (marginal 0.1), prob PB
    # type D: 1 k-touch retained w.p. 0.1, prob PD
    PD = 0.15 / 0.9
    PB = 0.30 - 0.1 * PD
    assert abs(0.55 + PB + PD - 1) < 1e-12
    A_j = 0.55 * 5 / (0.55 * 5 + PB * 10 + PD * 1)
    print(f"    [C1 without the n_max=1 cap] n_max=10 dependent-retention structure reproduces the same law "
          f"with A_j = {A_j:.4f} < 0.5 (so C1 needs the binding cap)")


# ---------------------------------------------------------------- Lemma 3
def check_lemma3_dependence():
    """Comonotone retention (R_l = 1{U < p_{c_l}}, one U per journey) satisfies
    Assumption 4. Journey: one touch on j, one on k, r_j = 2, r_k = 1."""
    rj, rk = 2.0, 1.0
    claimed = rj / (rj + rk)
    for eps in [0.1, 0.01, 0.001]:
        pj, pk = eps * rj, eps * rk
        # P(T>0) = pj; given T>0 both retained w.p. pk/pj (share 1/2), else only j (share 1)
        q = (1 - pk / pj) * 1.0 + (pk / pj) * 0.5
        print(f"[Lemma 3] comonotone retention eps={eps}: q_j = {q:.4f}; claimed limit {claimed:.4f}")
    # independent retention: converges to the claimed limit
    for eps in [0.1, 0.01, 0.001]:
        pj, pk = eps * rj, eps * rk
        PT = 1 - (1 - pj) * (1 - pk)
        q = (pj * (1 - pk) + 0.5 * pj * pk) / PT
        print(f"[Lemma 3] independent retention eps={eps}: q_j = {q:.4f}")


# ---------------------------------------------------------------- Lemma 4
def e_share_indep(counts, probs, j):
    """E[M_j/M | M>0] for independent Bernoulli retention, campaign counts."""
    pmfs = [np.array([math.comb(n, r) * p ** r * (1 - p) ** (n - r) for r in range(n + 1)])
            for n, p in zip(counts, probs)]
    num = 0.0
    pos = 0.0
    for rs in itertools.product(*[range(n + 1) for n in counts]):
        pr = np.prod([pmfs[c][r] for c, r in enumerate(rs)])
        M = sum(rs)
        if M > 0:
            pos += pr
            num += pr * rs[j] / M
    return num / pos


def q_touch(pis, ell):
    """q_ell = P(ell retained and selected | M>0) with per-touch probabilities."""
    n = len(pis)
    num = 0.0
    pos = 0.0
    for R in itertools.product((0, 1), repeat=n):
        pr = np.prod([pi if r else 1 - pi for pi, r in zip(pis, R)])
        M = sum(R)
        if M:
            pos += pr
            if R[ell]:
                num += pr / M
    return num / pos


def check_lemma4(n=3000):
    worst_up = 0.0
    worst_lo = 0.0
    for _ in range(n):
        K = RNG.integers(2, 4)
        counts = list(RNG.integers(1, 4, K))
        G = RNG.uniform(1, 3)
        base = RNG.uniform(0.02, 1 / G)
        probs = list(base * RNG.uniform(1, G, K))
        probs = [min(p, 1.0) for p in probs]
        G_real = max(probs) / min(probs)
        s = counts[0] / sum(counts)
        e = e_share_indep(counts, probs, 0)
        worst_up = max(worst_up, e / (G_real * s))
        worst_lo = max(worst_lo, (s / G_real) / e)
    print(f"[Lemma 4] {n} random journeys: max e/(Gamma s) = {worst_up:.4f} (<=1), "
          f"max (s/Gamma)/e = {worst_lo:.4f} (<=1)")

    # Per-touch competing monotonicity (independent retention): q_ell non-increasing
    # in every other touch's probability, non-decreasing in its own.
    viol_comp = viol_own = 0
    for _ in range(3000):
        nn = RNG.integers(2, 6)
        pis = list(RNG.uniform(0.01, 1, nn))
        ell = 0
        other = RNG.integers(1, nn)
        base = q_touch(pis, ell)
        up = pis.copy(); up[other] = min(1.0, up[other] + RNG.uniform(0.001, 0.3))
        if q_touch(up, ell) > base + 1e-12:
            viol_comp += 1
        up = pis.copy(); up[ell] = min(1.0, up[ell] + RNG.uniform(0.001, 0.3))
        if q_touch(up, ell) < base - 1e-12:
            viol_own += 1
    print(f"[Lemma 4 / App. A.1] per-touch monotonicity, 3000 draws: competing violations={viol_comp}, own violations={viol_own}")

    # App. A.1 says c = E[1/(1+M_-l)] does not depend on p = p_{c_l}. False when
    # the journey has another touch on the same campaign. Journey (j, j, k).
    pj, pk = 0.4, 0.3

    def c_of(pj):
        # M_-l = R_{other j} + R_k
        return sum(pr / (1 + m) for m, pr in
                   [(0, (1 - pj) * (1 - pk)), (1, pj * (1 - pk) + (1 - pj) * pk), (2, pj * pk)])
    print(f"[App. A.1] journey (j,j,k): c(p_j=0.4)={c_of(0.4):.4f}, c(p_j=0.8)={c_of(0.8):.4f} "
          "-> c depends on own-campaign p")
    # numerical derivative of campaign-level q_ell vs the appendix formula holding c, d fixed
    h = 1e-6
    ql = lambda pj: q_touch([pj, pj, pk], 0)
    num_der = (ql(pj + h) - ql(pj - h)) / (2 * h)
    c = c_of(pj); d = (1 - pj) * (1 - pk)
    formula = c * (1 - d) / (1 - (1 - pj) * d) ** 2
    print(f"    d q_l / d p_j (campaign level) = {num_der:.4f}; appendix formula c(1-d)/[.]^2 = {formula:.4f}")

    # Is campaign-level q_ell / e_j monotone in own-campaign p_j? (not claimed by the
    # lemma statement; scanned for information)
    nonmono = 0
    for nj in range(1, 5):
        for nk in range(0, 5):
            for pk in [0.05, 0.3, 0.7, 1.0]:
                vals = [e_share_indep([nj, nk], [p, pk], 0) for p in np.linspace(0.01, 1, 60)]
                if np.any(np.diff(vals) < -1e-12):
                    nonmono += 1
    print(f"[Lemma 4] campaign-level e_j monotone in p_j on grid (n_j<=4, n_k<=4): non-monotone cases = {nonmono}")


# ---------------------------------------------------------------- Theorem 2
def q_and_Q(types, probs, w=None, dependent=False):
    """types: list of (prob, counts per campaign); linear rule, campaign 0 = j.
    Returns (q_j, Q_j, P(T>0)). dependent=True uses comonotone retention."""
    num = den = Q = 0.0
    for pi, counts in types:
        n = sum(counts)
        Q += pi * counts[0] / n
        if not dependent:
            e = e_share_indep(counts, probs, 0) if counts[0] else 0.0
            s = 1 - np.prod([(1 - p) ** c for p, c in zip(probs, counts)])
        else:
            # comonotone: touches with p_c > U are retained
            present = [(p, c, idx) for idx, (p, c) in enumerate(zip(probs, counts)) if c > 0]
            present.sort(reverse=True)
            s = present[0][0]
            se = 0.0
            ps = [x[0] for x in present] + [0.0]
            for r in range(len(present)):
                band = ps[r] - ps[r + 1]
                kept = present[: r + 1]
                Mj = sum(c for p, c, idx in kept if idx == 0)
                M = sum(c for p, c, idx in kept)
                se += band * Mj / M
            e = se / s
        num += pi * s * e
        den += pi * s
    return num / den, Q, den


def check_thm2():
    # (i) identity on random distributions
    worst = 0.0
    for _ in range(2000):
        S = RNG.uniform(0.1, 5, 6)
        Z = RNG.uniform(0, 1, 6)
        pr = RNG.dirichlet(np.ones(6))
        ES = pr @ S
        A = pr @ (S * Z) / ES
        Q = pr @ Z
        cov = pr @ (S * Z) - ES * Q
        worst = max(worst, abs((Q - A) + cov / ES))
        sd = math.sqrt(pr @ S ** 2 - ES ** 2)
        assert abs(Q - A) <= 0.5 * sd / ES + 1e-12
    print(f"[Thm 2(i)] identity max error {worst:.1e}; |Q-A| <= SD(S)/(2E S) held")

    # (ii) under a count cap n_max the paper's construction cannot drive q_j below Q_j/(Gamma n_max)
    n_max = 10
    worst = math.inf
    for _ in range(400):
        ntypes = RNG.integers(1, 4)
        types = []
        pr = RNG.dirichlet(np.ones(ntypes))
        for t in range(ntypes):
            counts = [int(RNG.integers(0, 4)), int(RNG.integers(0, 4))]
            if sum(counts) == 0:
                counts[0] = 1
            types.append((pr[t], counts))
        if all(c[0] == 0 for _, c in types):
            continue
        G = RNG.uniform(1, 3)
        pmin = RNG.uniform(0.001, 1 / G)
        probs = [pmin * RNG.uniform(1, G), pmin * RNG.uniform(1, G)]
        Greal = max(probs) / min(probs)
        nm = max(sum(c) for _, c in types)
        for dep in (False, True):
            q, Q, _ = q_and_Q(types, probs, dependent=dep)
            worst = min(worst, q / (Q / (Greal * nm)))
    print(f"[Thm 2(ii)] random capped mixtures: min q_j / (Q_j/(Gamma n_max)) = {worst:.4f} (>= 1 expected)")
    for N in (2, 5, 10):
        pi = 0.99
        lim = pi / (pi + N * (1 - pi))  # p -> 0, equal retention
        print(f"    paper's construction with N={N}, Q_j=0.99: q_j -> {lim:.4f} as p->0 (cannot be < 0.01 when N<=10)")

    # (iii)
    w, p = 100.0, 0.5
    ratio = (1 + w * (1 - p)) / (2 - p)
    direct = ((p ** 2 / (1 + w) + p * (1 - p)) / (1 - (1 - p) ** 2)) / (1 / (1 + w))
    print(f"[Thm 2(iii)] omega=100, p=0.5: closed form {ratio:.4f}, direct {direct:.4f}")

    # (iv) attaining sequences: with a cap N+1 <= n_max the limits are not reached
    p_lo, Ge = 0.4, 2.5
    for N in (9, 50, 500):
        lo = p_lo / (1 - (1 - p_lo) ** N)                      # pi -> 0 limit
        up = Ge * (1 - (1 - p_lo) ** (N + 1)) / p_lo           # pi -> 0 limit
        print(f"[Thm 2(iv)] p_lo=0.4, Gamma_eff=2.5, N={N}: q/Q -> {lo:.6f} (bound {p_lo}), "
              f"-> {up:.6f} (bound {Ge/p_lo})")
    # envelope endpoint q_j/p_lo exceeds 1 for q_j > p_lo
    print("    note: Q_j <= 1, so the endpoint q_j/p_lo is unattainable whenever q_j > p_lo")


# ---------------------------------------------------------------- Theorem 3 / Corollary 1
def check_corollary_floor():
    """Corollary 1 says Gamma* < a_j/a_k on E, and also that under a floor the
    radius can be unbounded. Both cannot hold together."""
    p_lo = 0.5
    LN = np.array([0.49, 0.19, 0.30]); UN = np.array([0.51, 0.21, 0.32])
    m = np.array([0.50, 0.20, 0.31])  # true running means, inside the boxes
    rho = m[0] / m[1]
    certified_all = True
    for G in [1.0, 2.0, 2.5, 3.0, 10.0, 1000.0]:
        Ge = min(G, 1 / p_lo)
        lg = lemma2_min(LN, Ge * UN, 0, 1)
        certified_all &= lg > 0
    print(f"[Cor. 1] floor p_lo=0.5, a_j/a_k={rho:.2f}: certified at every Gamma up to 1000 -> "
          f"Gamma* = inf > a_j/a_k: {certified_all}")


def check_hybrid_clipping(n=200000):
    """Thm 3(iii): 'the clipped legacy component can exceed the direct one only
    through clipping the upper endpoint of a third campaign.' Check at the
    level of certification decisions (delta >= 0) and of raw values."""
    dec_diff = dec_not_third = val_diff = val_k = 0
    for _ in range(n):
        K = int(RNG.integers(2, 5))
        xbar = RNG.dirichlet(np.ones(K)) * RNG.uniform(0.05, 1)
        rN = RNG.uniform(0.0005, 0.3, K)
        LN = np.clip(xbar - rN, 0, 1); UN = np.clip(xbar + rN, 0, 1)
        ybar = xbar.sum(); rD = RNG.uniform(0.0005, 0.3)
        LD, UD = ybar - rD, ybar + rD
        if LD <= 0:
            continue
        Lr = np.clip(LN / UD, 0, 1); Ur = np.clip(UN / LD, 0, 1)
        G = RNG.uniform(1, 3)
        j, k = RNG.choice(K, 2, replace=False)
        delta = RNG.choice([0.0, 0.005, 0.02])
        d = lemma2_min(LN, G * UN, j, k)
        lg = lemma2_min(Lr, G * Ur, j, k)
        if lg > d + 1e-12:
            val_diff += 1
            # unclip k's upper only
            Ur_k = Ur.copy(); Ur_k[k] = UN[k] / LD
            if lemma2_min(Lr, G * Ur_k, j, k) <= d + 1e-12:
                val_k += 1
        if lg > delta >= d:
            dec_diff += 1
            Ur3 = Ur.copy()
            for mm in range(K):
                if mm not in (j, k):
                    Ur3[mm] = UN[mm] / LD
            if lemma2_min(Lr, G * Ur3, j, k) > delta:
                dec_not_third += 1
    print(f"[Thm 3(iii)] {n} random box draws: legacy>direct in value {val_diff} times "
          f"({val_k} explained by clipping of k's own upper end); decision differences {dec_diff}, "
          f"of which not explained by third-campaign clipping: {dec_not_third}")


# ---------------------------------------------------------------- Howard et al.
def check_howard():
    lam = np.linspace(1e-6, 0.999, 2000)
    for c in (0.5, 1.0, 2.0):
        l = lam / c
        psiE = (-np.log(1 - c * l) - c * l) / c ** 2
        psiG = l ** 2 / (2 * (1 - c * l))
        assert np.all(psiE <= psiG + 1e-15)
    print("[Howard] psi_E,c <= psi_G,c on [0,1/c) (so a sub-gamma boundary is a sub-exponential boundary)")
    eta, s, alpha = 2.0, 1.4, 0.025
    from math import log, sqrt
    zeta14 = sum(1 / n ** s for n in range(1, 2_000_000)) + (2_000_000 ** (1 - s)) / (s - 1)
    k1 = (eta ** 0.25 + eta ** -0.25) / sqrt(2)
    k2 = (sqrt(eta) + 1) / 2
    const = log(zeta14 / (alpha * log(eta) ** s))
    print(f"[Howard eq. (24)] k1*sqrt(s)={k1*sqrt(s):.3f} (1.7), const/s={const/s:.3f} (3.8), "
          f"2k2*s={2*k2*s:.3f} (3.4), 2k2*const={2*k2*const:.2f} (13)")
    # l(v) >= 0 at v = m for every alpha_u <= 1 -> sqrt(a^2+b^2) <= a+b applies
    print(f"    l(m) at alpha_u=1: {s*log(log(eta)) + log(zeta14/log(eta)**s):.3f} (>0)")


# ---------------------------------------------------------------- G3 fragmentation
def check_g3():
    worst = 1.0
    for rho in (0.2, 0.4, 0.6):
        lo, hi = 1.0, 0.0
        for n in range(1, 41):
            for ell in range(1, n + 1):
                pr = 1 - rho * (n - ell + 1) / n  # cut after touch g~U{1..n} orphans 1..g
                lo, hi = min(lo, pr), max(hi, pr)
        print(f"[G3] rho={rho}: per-touch retention probabilities in [{lo:.3f}, {hi:.3f}] "
              f"-> joint ratio <= 1/(1-rho) = {1/(1-rho):.3f}")


if __name__ == "__main__":
    check_lemma2()
    check_thm1_closed_forms()
    check_thm1_full_law_counterexamples()
    check_lemma3_dependence()
    check_lemma4()
    check_thm2()
    check_corollary_floor()
    check_hybrid_clipping()
    check_howard()
    check_g3()
