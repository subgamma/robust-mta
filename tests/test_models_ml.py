"""Tests for the ML-based models (logistic+SHAP, DARNN). Both are smoke
tests on tiny synthetic data -- confirming the frozen interface works
end-to-end -- not accuracy/reproduction tests against real Criteo data
(that training happens outside this test suite;)."""
import numpy as np
import pytest

from anytime_attribution.journeys import build_journey, linear_weight
from anytime_attribution.models.logistic_shap import LogisticSHAP
from anytime_attribution.models.darnn import DARNN

N_CAMPAIGNS = 3


def _journey(raw, conversion_time=10.0):
    return build_journey(raw, conversion_time, linear_weight)


def _synthetic_pos_neg(rng, n=60):
    pos, neg = [], []
    for _ in range(n):
        c = rng.integers(0, N_CAMPAIGNS)
        n_touches = rng.integers(1, 4)
        raw = [(int(c), float(t)) for t in sorted(rng.uniform(0, 9, size=n_touches))]
        pos.append(_journey(raw))
    for _ in range(n):
        c = rng.integers(0, N_CAMPAIGNS)
        n_touches = rng.integers(1, 4)
        raw = [(int(c), float(t)) for t in sorted(rng.uniform(0, 9, size=n_touches))]
        neg.append(_journey(raw))
    return pos, neg


def test_logistic_shap_credit_sums_to_one_and_touches_only():
    rng = np.random.default_rng(0)
    pos, neg = _synthetic_pos_neg(rng)
    m = LogisticSHAP()
    m.fit(pos, n_campaigns=N_CAMPAIGNS, seed=0, negative_journeys=neg)
    for j in pos[:10]:
        z = m.credit(j)
        assert z.sum() == pytest.approx(1.0)
        touched = {t.campaign for t in j.touches}
        for c in range(N_CAMPAIGNS):
            if c not in touched:
                assert z[c] == 0.0


def test_logistic_shap_requires_negative_journeys():
    m = LogisticSHAP()
    with pytest.raises(ValueError):
        m.fit([_journey([(0, 1.0)])], n_campaigns=N_CAMPAIGNS, seed=0, negative_journeys=None)


def test_darnn_smoke_fit_and_credit():
    rng = np.random.default_rng(1)
    pos, neg = _synthetic_pos_neg(rng, n=40)
    m = DARNN(hidden_size=4, epochs=2)
    m.fit(pos, n_campaigns=N_CAMPAIGNS, seed=0, negative_journeys=neg)
    for j in pos[:10]:
        z = m.credit(j)
        assert z.shape == (N_CAMPAIGNS,)
        assert z.sum() == pytest.approx(1.0, abs=1e-4)
        assert (z >= -1e-6).all()


def test_darnn_requires_negative_journeys():
    m = DARNN()
    with pytest.raises(ValueError):
        m.fit([_journey([(0, 1.0)])], n_campaigns=N_CAMPAIGNS, seed=0, negative_journeys=None)
