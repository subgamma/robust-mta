import os

import pytest

from anytime_attribution.experiments._primary_journeys import CACHE_PATH
from anytime_attribution.experiments import e7_interventions as e7

pytestmark = pytest.mark.skipif(
    not os.path.exists(CACHE_PATH) or not os.path.exists(e7.E6_RESULTS_PATH),
    reason="primary journey cache (built from the raw Criteo file) or e6 results not available",
)


def _small_run():
    journeys, top_20 = e7.build_and_cache_primary_journeys()
    import anytime_attribution.experiments._primary_journeys as pj
    orig = pj.build_and_cache_primary_journeys
    # The cumulative-share CS is wider at small N than a grid CS would be: 4000 journeys leaves R0 empty, 10000 gives
    # a non-degenerate reference graph.
    pj.build_and_cache_primary_journeys = lambda *a, **k: (journeys[:10000], top_20)
    e7.build_and_cache_primary_journeys = pj.build_and_cache_primary_journeys
    try:
        return e7.run(n_reps=1, n_reps_targeted=2)
    finally:
        pj.build_and_cache_primary_journeys = orig
        e7.build_and_cache_primary_journeys = orig


def test_e7_g1_uniform_loss_never_wrongly_certifies():
    """G1 is uniform across campaigns -- the true ratio is unchanged, so
    the robust rule (well inside the model) should show wrong_rate 0."""
    result = _small_run()
    for rho, g in result["G1"].items():
        assert g["robust_wrong_rate_mean"] == 0.0, f"G1 rho={rho} wrongly certified"


def test_e7_drr_and_wrong_rate_in_unit_interval():
    result = _small_run()
    for group in ("G1", "G2", "G3"):
        for key, g in result[group].items():
            for metric in g:
                if metric.startswith(("naive_wrong", "robust_wrong", "drr")) and isinstance(g[metric], float):
                    assert 0.0 <= g[metric] <= 1.0 or g[metric] != g[metric]  # allow nan if R0 empty


def test_e7_g3_labelled_out_of_model():
    result = _small_run()
    for rho, g in result["G3"].items():
        assert g["out_of_model"] is True


def test_e7_g2_cross_check_present_and_consistent():
    """Cross-check counters are internally consistent.

        Margin-and-wrong is a subset of n_pair_rep_combinations, and r=1.25 never shows a possible sign reversal: every
        R_0 pair is certified at Gamma=1.25, so Gamma*_jk >= 1.25, and Gamma*_jk < rho_ref_jk, so rho_ref_jk > 1.25
        and r=1.25 > rho_ref_jk is impossible.
    """
    result = _small_run()
    for r, g in result["G2"].items():
        cc = g["cross_check"]
        assert cc["n_margin_reversal_possible"] <= cc["n_pair_rep_combinations"]
        assert cc["n_margin_reversal_possible_and_naively_wrong"] <= cc["n_margin_reversal_possible"]
        if float(r) == 1.25:
            assert cc["n_sign_reversal_possible"] == 0


def test_e7_g2_targeted_uses_e6_least_robust_pair():
    result = _small_run()
    gt = result["G2_targeted"]
    assert gt["stronger"] != gt["weaker"]
    lo, hi = gt["settings"]["r_low"], gt["settings"]["r_high"]
    assert lo["r"] < gt["rho_ref_jk"] < hi["r"]
    assert lo["predicted_sign_reversal"] is False
    assert hi["predicted_sign_reversal"] is True
    for s in (lo, hi):
        assert 0.0 <= s["naive_wrong_rate"] <= 1.0
        assert 0.0 <= s["robust_wrong_rate"] <= 1.0
