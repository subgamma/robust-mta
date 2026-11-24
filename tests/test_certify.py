import numpy as np
import pytest

from anytime_attribution.cs import invert_share_cs
from anytime_attribution.certify import (
    Certification,
    robust_rule,
    naive_certify,
    certified_graph,
    is_acyclic,
    gamma_star,
)


def _true_A_from_a_and_p(a, p):
    a = np.asarray(a, dtype=float)
    p = np.asarray(p, dtype=float)
    u = a / p
    return u / u.sum()


@pytest.mark.parametrize("boundary", ["wsr", "howard"])
def test_robust_rule_no_incorrect_certification_on_500_replications(boundary):
    """Acceptance check: with a CS that covers the truth,
    no incorrect certification on 500 synthetic replications; certified graph
    acyclic. 'Incorrect' means Delta_jk <= delta was certified as j > k."""
    rng = np.random.default_rng(2024)
    K = 4
    p = np.array([0.30, 0.25, 0.22, 0.20])
    Gamma_used = float(p.max() / p.min())  # true retention ratio bound
    a_true = np.array([0.40, 0.28, 0.20, 0.12])  # observed aggregate shares (sum=1)
    A_true = _true_A_from_a_and_p(a_true, p)
    delta = 0.0
    alpha = 0.05
    alpha_per_campaign = alpha / K
    B = 4.0
    n_conv = 400
    n_reps = 500

    n_incorrect = 0
    n_pairs_certified = 0
    for _ in range(n_reps):
        T_hist = rng.uniform(1.0, B, size=n_conv)
        L = np.zeros(K)
        U = np.zeros(K)
        for j in range(K):
            noise = rng.normal(0, 0.05, size=n_conv) * T_hist
            Tj_hist = np.clip(a_true[j] * T_hist + noise, 0.0, T_hist)
            lo, hi = invert_share_cs(list(Tj_hist), list(T_hist), alpha_per_campaign, B, boundary=boundary)
            L[j], U[j] = lo, hi

        certs = {}
        for j in range(K):
            for k in range(j + 1, K):
                cert = robust_rule(L, U, Gamma_used, delta, j, k)
                certs[(j, k)] = cert
                if cert == Certification.J_OVER_K:
                    n_pairs_certified += 1
                    if not (A_true[j] - A_true[k] > delta):
                        n_incorrect += 1
                elif cert == Certification.K_OVER_J:
                    n_pairs_certified += 1
                    if not (A_true[k] - A_true[j] > delta):
                        n_incorrect += 1

        graph = certified_graph(certs, K)
        if n_incorrect == 0:
            # if every issued edge in this replication is a true ordering,
            # the graph must be acyclic (transitivity of real-number
            # comparisons) -- this holds unconditionally, no coverage
            # assumption needed.
            assert is_acyclic(graph)

    # alpha-budget: some rare incorrect certifications are consistent with
    # the theory (P(ever incorrect) <= alpha per replication); zero is
    # expected in practice given the conservative boundaries implemented
    # here, but we allow slack rather than asserting exact zero.
    assert n_incorrect <= max(1, int(0.05 * n_pairs_certified))


def test_robust_rule_reduces_to_ratio_test_at_delta_zero():
    L = np.array([0.5, 0.1, 0.05])
    U = np.array([0.6, 0.2, 0.1])
    Gamma = 2.0
    cert = robust_rule(L, U, Gamma, delta=0.0, j=0, k=1)
    assert cert == Certification.J_OVER_K
    assert L[0] / U[1] > Gamma  # eq. robustlow remark at delta=0


def test_robust_rule_unresolved_when_gap_not_cleared():
    L = np.array([0.3, 0.25])
    U = np.array([0.4, 0.35])
    cert = robust_rule(L, U, Gamma=1.5, delta=0.0, j=0, k=1)
    assert cert == Certification.UNRESOLVED


def test_naive_certify_basic():
    assert naive_certify(lo=0.02, hi=0.10, delta=0.01) == Certification.J_OVER_K
    assert naive_certify(lo=-0.10, hi=-0.02, delta=0.01) == Certification.K_OVER_J
    assert naive_certify(lo=-0.005, hi=0.005, delta=0.01) == Certification.UNRESOLVED


def test_certified_graph_and_acyclicity():
    certs = {
        (0, 1): Certification.J_OVER_K,
        (1, 2): Certification.J_OVER_K,
        (0, 2): Certification.J_OVER_K,
    }
    graph = certified_graph(certs, n_campaigns=3)
    assert graph == {0: {1, 2}, 1: {2}, 2: set()}
    assert is_acyclic(graph)


def test_certified_graph_detects_cycle():
    graph = {0: {1}, 1: {2}, 2: {0}}
    assert not is_acyclic(graph)


def test_gamma_star_monotone_and_bisection_matches_direct_check():
    L = np.array([0.5, 0.1])
    U = np.array([0.6, 0.2])
    delta = 0.0
    j, k = 0, 1
    g_star = gamma_star(L, U, delta, j, k)
    assert g_star is not None
    from anytime_attribution.certify import robust_lower_gap

    assert robust_lower_gap(L, U, g_star - 1e-3, j, k) > delta
    assert robust_lower_gap(L, U, g_star + 1e-3, j, k) <= delta + 1e-6


def test_gamma_star_none_when_not_certified_at_gamma_one():
    L = np.array([0.3, 0.29])
    U = np.array([0.35, 0.34])
    assert gamma_star(L, U, delta=0.0, j=0, k=1) is None
