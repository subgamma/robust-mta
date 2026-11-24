"""E10 controlled multi-campaign synthetic benchmark: generator, retention, masses, certification budget,
decision-loss formulas, leakage, 45-pair coverage. See anytime_attribution/experiments/e10_multicampaign.py."""
import importlib.util
import itertools
import sys

import numpy as np
import pytest

from anytime_attribution.certify import Certification, certify_pair
from anytime_attribution.cs import MultiCampaignCumulativeShareCS
from anytime_attribution.cs_vectorized import cumulative_share_boxes
from anytime_attribution.experiments import e10_multicampaign as E


def _runner():
    spec = importlib.util.spec_from_file_location("run_e10", "scripts/run_e10_multicampaign.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


@pytest.fixture(scope="module")
def stream():
    rng = np.random.default_rng(0)
    trip, rate = E.sample_converting(20000, rng)
    return trip, rate


def test_generator_basic_validity_and_cap(stream):
    trip, rate = stream
    N = trip["N"]; jid, camp, tau, n = trip["jid"], trip["camp"], trip["tau"], trip["n"]
    assert len(n) == N and n.min() >= 1 and n.max() <= E.CAP          # latent cap B = 10 on touches
    assert np.array_equal(np.bincount(jid, minlength=N), n)
    assert camp.min() >= 0 and camp.max() < E.K and tau.min() >= 0 and tau.max() <= E.WINDOW
    # within a journey touches run earliest -> latest (tau non-increasing)
    same = jid[1:] == jid[:-1]
    assert np.all(tau[1:][same] <= tau[:-1][same] + 1e-15)
    assert 0.02 < rate < 0.2


def test_composition_target_met_and_reported_not_assumed(stream):
    trip, _ = stream
    N = trip["N"]; u = np.unique(trip["jid"] * E.K + trip["camp"]); nd = np.bincount(u // E.K, minlength=N)
    assert 0.45 <= (nd >= 2).mean() <= 0.55
    assert (nd >= 3).mean() > 0.25
    assert (trip["n"] == 1).mean() > 0.2                                # heavy-tailed: many short, some capped at 10
    assert (trip["n"] == E.CAP).mean() > 0.05


def test_role_structure_early_upper_late_lower(stream):
    trip, _ = stream
    first = trip["camp"][np.concatenate([[0], np.cumsum(trip["n"])[:-1]])][trip["n"] >= 4]
    last = trip["camp"][np.cumsum(trip["n"]) - 1][trip["n"] >= 4]
    assert np.mean(E.ROLE_OF[first] == 0) > np.mean(E.ROLE_OF[last] == 0)
    assert np.mean(E.ROLE_OF[last] == 2) > np.mean(E.ROLE_OF[first] == 2)


@pytest.mark.parametrize("r", E.R_TRUE_GRID)
def test_retention_vector_ratio_floor_admissible(r):
    p = E.retention_vector(r)
    chk = E.check_retention(p, r, p.min())
    assert chk["in_(0,1]"] and chk["ratio_matches"] and chk["floor_respected"]
    assert p.max() / p.min() == pytest.approx(r, rel=1e-12)
    assert p.min() >= 0.3 - 1e-12                                        # above the configured floor 0.3 even at r = 2.5


def test_in_model_retention_is_independent_bernoulli_by_campaign(stream):
    """Marginal retention given the latent journey equals p_{c}: independent of position, tau and journey length."""
    trip, _ = stream
    p = E.retention_vector(2.0)
    keep = E.thin(trip, p, np.random.default_rng(1))
    camp, tau = trip["camp"], trip["tau"]
    for c in range(E.K):
        sel = camp == c
        assert abs(keep[sel].mean() - p[c]) < 4 * np.sqrt(p[c] * (1 - p[c]) / sel.sum())
        early = sel & (tau > 3.5); late = sel & (tau <= 3.5)
        assert abs(keep[early].mean() - keep[late].mean()) < 5 * np.sqrt(p[c] * (1 - p[c]) * (1 / early.sum() + 1 / late.sum()))
    long_ = np.repeat(trip["n"] >= 6, trip["n"])
    assert abs(keep[long_ & (camp == 0)].mean() - keep[~long_ & (camp == 0)].mean()) < 0.03


def test_masses_bounded_zero_rows_kept_and_weights_fixed(stream):
    trip, _ = stream
    N = trip["N"]; p = E.retention_vector(2.5)
    keep = E.thin(trip, p, np.random.default_rng(2))
    shap = E.LinearSHAPModel(); shap.beta = np.linspace(-0.05, 0.1, E.K); shap.mu = np.full(E.K, 1.0)
    for m in E.MODELS:
        S, B = E.model_masses(m, trip, None, N, shap); T, B2 = E.model_masses(m, trip, keep, N, shap)
        assert S.shape == (N, E.K) == T.shape and B == B2
        assert S.min() >= -1e-12 and S.sum(1).max() <= B + 1e-9 and T.sum(1).max() <= B + 1e-9
        if m in ("linear", "time_decay"):
            assert np.all(T <= S + 1e-12)                                # thinning only removes mass
    # zero-observed-mass conversions exist and are KEPT as rows (arrival clock preserved)
    T, _ = E.model_masses("linear", trip, keep, N)
    assert (T.sum(1) == 0).sum() > 0 and T.shape[0] == N
    # time-decay weight is a property of the latent touch: a retained touch keeps exactly 2^{-tau/7}
    one = np.nonzero(keep)[0][:2000]
    Td, _ = E.model_masses("time_decay", trip, keep, N)
    manual = np.zeros(N); np.add.at(manual, trip["jid"][one], 2.0 ** (-trip["tau"][one] / E.H_DECAY))
    sub = np.unique(trip["jid"][one]); full_keep = keep.copy(); full_keep[:] = False; full_keep[one] = True
    Td2, _ = E.model_masses("time_decay", trip, full_keep, N)
    assert np.allclose(Td2.sum(1), manual)                               # no renormalisation after deletion


def test_last_touch_is_last_retained_touch():
    trip = {"jid": np.array([0, 0, 0, 1]), "camp": np.array([1, 2, 3, 4]), "tau": np.array([5.0, 3.0, 1.0, 2.0]), "N": 3}
    keep = np.array([True, True, False, True])
    T, B = E.model_masses("last_touch", trip, keep, 3)
    assert B == 1.0 and T[0, 2] == 1 and T[0].sum() == 1 and T[1, 4] == 1 and T[2].sum() == 0
    S, _ = E.model_masses("last_touch", trip, None, 3)
    assert S[0, 3] == 1                                                  # latent reference: last latent touch


def test_exact_linear_shap_matches_shap_library():
    import shap
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(3)
    X = rng.poisson(1.0, size=(500, E.K)).astype(float); y = (rng.random(500) < 0.3).astype(int)
    lr = LogisticRegression(max_iter=1000).fit(X, y)
    mu = X[y == 0].mean(0)
    mine = E.LinearSHAPModel(); mine.beta = lr.coef_.ravel(); mine.mu = mu
    ex = shap.LinearExplainer(lr, mu.reshape(1, -1))                      # interventional, single background row = the mean
    ref = np.asarray(ex.shap_values(X[:50]))
    assert np.allclose(mine.beta[None, :] * (X[:50] - mu[None, :]), ref, atol=1e-8)


def test_logistic_model_trained_only_on_warmup_and_frozen():
    rng = np.random.default_rng(4)
    warm = E.sample_warmup(60000, np.random.default_rng(np.random.SeedSequence([20263, 1])))
    m = E.LinearSHAPModel().fit(warm, seed=0)
    beta0, mu0 = m.beta.copy(), m.mu.copy()
    trip, _ = E.sample_converting(2000, np.random.default_rng(np.random.SeedSequence([20263, 3, 0])))
    E.model_masses("logistic_shap", trip, E.thin(trip, E.retention_vector(2.0), rng), 2000, m)
    assert np.array_equal(m.beta, beta0) and np.array_equal(m.mu, mu0)   # A1: never touched by monitored data
    assert m.n_train == 60000 and warm["y"].sum() > 1000 and (~warm["y"]).sum() > 1000   # includes non-converting journeys
    # the warm-up and monitoring random streams are different seed paths (no shared draws)
    a = np.random.default_rng(np.random.SeedSequence([20263, 1])).random(5); b = np.random.default_rng(np.random.SeedSequence([20263, 3, 0])).random(5)
    assert not np.allclose(a, b)
    import inspect
    assert list(inspect.signature(E.LinearSHAPModel.fit).parameters)[:2] == ["self", "warm"]   # fit sees only the warm-up dict


def test_loss_formulas_hand_example():
    pl = E.pair_list()
    Delta = {p: 0.0 for p in pl}
    Delta[(0, 1)] = 0.10; Delta[(0, 2)] = -0.05; Delta[(0, 3)] = 0.02; Delta[(0, 4)] = 0.003
    cert = {p: "unres" for p in pl}
    cert[(0, 1)] = "k>j"     # wrong on an actionable pair, |Delta| = 0.10
    cert[(0, 2)] = "k>j"     # correct
    cert[(0, 3)] = "unres"   # unresolved actionable, |Delta| = 0.02
    cert[(0, 4)] = "j>k"     # |Delta| <= delta: certification is incorrect (theorem sense) but NOT actionable
    absD, act, corr, wrong, unres = E.decision_outcomes(cert, Delta, 0.005)
    s = E.loss_summary(absD, act, corr, wrong, unres)
    assert s["n_actionable"] == 3 and s["wrong_actionable"] == 1 and s["wrong_nonactionable"] >= 1
    assert s["resolved_opportunity_fraction"] == pytest.approx(1 / 3)
    assert s["directional_error_loss"] == pytest.approx(0.10)
    assert s["abstention_adjusted_loss_lambda=0.0"] == pytest.approx(0.10)
    assert s["abstention_adjusted_loss_lambda=0.5"] == pytest.approx(0.10 + 0.5 * 0.02)
    assert s["abstention_adjusted_loss_lambda=1.0"] == pytest.approx(0.10 + 0.02)
    assert s["abstention_adjusted_loss_normalised_lambda=1.0"] == pytest.approx((0.10 + 0.02) / 0.17)


def test_adapter_equals_production_cs_and_certify_pair_hybrid():
    R = _runner()
    rng = np.random.default_rng(5)
    N = 3000
    Tj = np.zeros((N, E.K)); Tj[np.arange(N), rng.choice(E.K, N, p=E.CAMPAIGN_POP / E.CAMPAIGN_POP.sum())] = rng.integers(1, 11, N)
    T = Tj.sum(1); cs = MultiCampaignCumulativeShareCS(E.K, 0.05, 10.0)
    for i in range(N): cs.update(Tj[i], T[i])
    bx = cumulative_share_boxes(Tj, T, 10.0, 0.05 / (E.K + 1) / 2)
    ad = R.BoxesCS(bx, N)
    assert np.allclose(ad.intervals()[0], cs.intervals()[0], atol=1e-12) and np.allclose(ad.numerator_boxes()[1], cs.numerator_boxes()[1], atol=1e-12)
    assert cs.beta == pytest.approx(0.05 / (E.K + 1))                    # shared alpha/(K+1) budget
    for G in (1.0, 1.5):
        for (j, k) in E.pair_list():
            assert certify_pair(ad, G, 0.005, j, k, method="hybrid") == certify_pair(cs, G, 0.005, j, k, method="hybrid")


def test_all_45_pairs_evaluated_and_determinism():
    R = _runner()
    assert len(E.pair_list()) == 45 == len(set(E.pair_list())) and len(R.PAIRS) == 45
    shap = E.LinearSHAPModel(); shap.beta = np.linspace(-0.02, 0.08, E.K); shap.mu = np.ones(E.K)
    A_pop = {m: np.full(E.K, 0.1) + 0.01 * np.arange(E.K) / E.K for m in E.MODELS}
    for m in E.MODELS: A_pop[m] = A_pop[m] / A_pop[m].sum()
    H = (1500,)
    rep, rate, res_h = R.one_rep((0, 1500, A_pop, (shap.beta, shap.mu), H, "D"))
    rep2, rate2, res2_h = R.one_rep((0, 1500, A_pop, (shap.beta, shap.mu), H, "D"))
    assert rate == rate2
    res = {k: v[1500] for k, v in res_h.items()}; res2 = {k: v[1500] for k, v in res2_h.items()}
    for key, v in res.items():
        assert len(v["codes"]) == 45 and len(v["gap_jk"]) == 45 and len(v["gap_kj"]) == 45
        assert v["n_correct"] + v["n_wrong"] + v["n_unres"] == 45
        assert v == res2[key] or all(v[k] == res2[key][k] for k in ("n_correct", "n_wrong", "n_unres", "ever_wrong"))
    assert {k[1] for k in res} == set(E.R_TRUE_GRID)
    names = {k[2] for k in res if k[1] == 2.0}
    assert any(n.startswith("naive") for n in names) and any(n.startswith("wellspec") for n in names) and any(n.startswith("misspec") for n in names)
    assert not any(n.startswith("misspec") for k in res for n in [k[2]] if k[1] in (1.25, 1.5))


def test_prop6_identity_on_generated_population(stream):
    trip, _ = stream
    N = trip["N"]; S, _ = E.model_masses("linear", trip, None, N)
    Sl = S.sum(1); Z = S / Sl[:, None]
    Q = Z.mean(0); A = S.sum(0) / Sl.sum()
    cov = ((Sl - Sl.mean())[:, None] * (Z - Z.mean(0))).mean(0)
    assert np.allclose(Q - A, -cov / Sl.mean(), atol=1e-12)


def test_config_A_to_D_use_requested_engine_and_hybrid_dominates_legacy():
    R = _runner()
    shap = E.LinearSHAPModel(); shap.beta = np.linspace(-0.02, 0.08, E.K); shap.mu = np.ones(E.K)
    A_pop = {m: np.full(E.K, 0.1) for m in E.MODELS}
    out = {c: R.one_rep((0, 2500, A_pop, (shap.beta, shap.mu), (2500,), c))[2] for c in "ABCD"}
    for key in out["D"]:
        a, b, c_, d = (np.array(out[x][key][2500]["gap_jk"]) for x in "ABCD")
        assert np.all(b >= a - 1e-12) and np.all(d >= c_ - 1e-12)             # hybrid >= legacy at equal boundary
        assert np.all(c_ >= a - 1e-12) and np.all(d >= b - 1e-12)             # exact boundary >= simplified at equal rule
