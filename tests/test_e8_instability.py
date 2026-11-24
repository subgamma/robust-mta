import os

import pytest

from anytime_attribution.experiments._primary_journeys import CACHE_PATH
from anytime_attribution.experiments import e8_instability as e8

pytestmark = pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="primary journey cache (built from the raw Criteo file) not available")


def test_e8_reversal_rates_in_unit_interval():
    result = e8.run(bootstrap_b=20)
    assert 0.0 <= result["sampling"]["reversal_rate"] <= 1.0
    for v in result["model_choice_primary"]["reversal_rate_by_model"].values():
        assert 0.0 <= v <= 1.0
    for v in result["secondary"]["model_choice_reversal_rate_by_model"].values():
        assert 0.0 <= v <= 1.0
    for v in result["secondary"]["darnn_seed_reversal_rate"].values():
        assert 0.0 <= v <= 1.0


def test_e8_secondary_effective_n_matches_established_804_20089():
    result = e8.run(bootstrap_b=5)
    sec = result["secondary"]
    assert sec["n_journeys"] == 20089
    assert sec["n_multi_campaign"] == 804
