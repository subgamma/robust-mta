import os

import pytest

from anytime_attribution.experiments._primary_journeys import CACHE_PATH
from anytime_attribution.experiments import e6_robustness_radius as e6

pytestmark = pytest.mark.skipif(not os.path.exists(CACHE_PATH), reason="primary journey cache (built from the raw Criteo file) not available")


def _small_run():
    journeys, top_20 = e6.build_and_cache_primary_journeys()
    sec_journeys, top_10 = e6.build_and_cache_secondary_journeys()
    import anytime_attribution.experiments._primary_journeys as pj
    orig_p, orig_s = pj.build_and_cache_primary_journeys, pj.build_and_cache_secondary_journeys
    pj.build_and_cache_primary_journeys = lambda *a, **k: (journeys[:4000], top_20)
    pj.build_and_cache_secondary_journeys = lambda *a, **k: (sec_journeys[:3000], top_10)
    e6.build_and_cache_primary_journeys = pj.build_and_cache_primary_journeys
    e6.build_and_cache_secondary_journeys = pj.build_and_cache_secondary_journeys
    try:
        return e6.run(last_touch_subsample_n=500)
    finally:
        pj.build_and_cache_primary_journeys = orig_p
        pj.build_and_cache_secondary_journeys = orig_s
        e6.build_and_cache_primary_journeys = orig_p
        e6.build_and_cache_secondary_journeys = orig_s


def test_e6_gamma_star_le_gamma_hat_domains_and_fragility_ranking_sorted():
    result = _small_run()
    for model in ("linear", "time_decay"):
        g = result["guaranteed"][model]
        assert g["n_certified"] <= g["n_pairs"]
        ranking = g["fragility_ranking"]
        mins = [r["min_gamma_star"] for r in ranking if r["min_gamma_star"] is not None]
        assert mins == sorted(mins)


def test_e6_secondary_multi_campaign_note_matches_804_20089_pattern():
    result = _small_run()
    sec = result["empirical_secondary"]
    assert sec["n_multi_campaign"] <= sec["n_journeys_secondary"]
    assert "campaign" in sec["note"]
