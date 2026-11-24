import os

import pytest

from anytime_attribution.experiments.e4_criteo_intervention import (
    CREDITS_PATH, J_IDX, K_IDX, MISSPECIFIED_CONFIGS, MISSPECIFIED_R, R_GRID,
    _load_reference_journeys, _reference_shares, run,
)

pytestmark = pytest.mark.skipif(not os.path.exists(CREDITS_PATH), reason="credits not materialized (run scripts/materialize_criteo_credits.py)")


def test_reference_shares_and_three_regime_design():
    campaign_idx, n_touches, conv_time = _load_reference_journeys(CREDITS_PATH)
    a_ref = _reference_shares(campaign_idx, n_touches, 20)
    rho = a_ref[J_IDX] / a_ref[K_IDX]
    assert rho > 1
    # module docstring's derivation: r < sqrt(rho) resumes, r > rho reverses
    # R_GRID = (1.1, 1.25, 1.5, 2.0, 4.0); rho_ref**2 is injected dynamically inside run.
    resumes_flags = [r <= rho / r for r in R_GRID]
    assert resumes_flags == [True, True, False, False, False]
    # stream order is chronological
    assert list(conv_time) == sorted(conv_time)


def test_e4_criteo_robust_never_wrong_small_run():
    result = run(n_reps=3, n_checkpoints=6)
    for res in result["results"].values():
        for L, rate in res["robust_wrong_rate"].items():
            assert rate == 0.0

    for res in result["results"].values():
        assert isinstance(res["resumes"], bool)
    assert result["config_hash"]


def test_e4_criteo_gamma_used_and_misspecified_rows():
    """Well-specified Gamma_used=r, the renamed ceiling field, and two misspecified demonstration configs."""
    result = run(n_reps=3, n_checkpoints=6)
    for r_key, res in result["results"].items():
        assert res["gamma_used"] == float(r_key)
        assert "population_ceiling_rho_over_r" in res
        assert "gamma_star_theory" not in res

    misspec_rows = [r for r, res in result["results"].items() if "misspecified" in res]
    assert misspec_rows == [MISSPECIFIED_R]
    misspec = result["results"][MISSPECIFIED_R]["misspecified"]
    assert len(misspec) == len(MISSPECIFIED_CONFIGS)
    for g, label in MISSPECIFIED_CONFIGS:
        m = misspec[str(g)]
        assert m["gamma_assumed"] == g
        assert m["r_true"] == MISSPECIFIED_R
        assert m["label"] == label


def test_e4_criteo_naive_is_production_cs_at_gamma1_not_legacy_wsr():
    """The headline naive_* fields come from the cumulative-share CS robust rule at Gamma=1, not from the fixed-mean WSR
    result, which sits under the labelled `legacy_naive_wsr` key.
    """
    result = run(n_reps=3, n_checkpoints=6)
    for res in result["results"].values():
        assert "naive_correct_rate" in res and "naive_unresolved_rate" in res
        assert "production CS" in res["naive_label"]
        assert "legacy_naive_wsr" in res
        assert "LEGACY" in res["legacy_naive_wsr"]["label"]
        diag = res["legacy_naive_wsr"]["constant_share_diagnostic"]
        assert diag["campaign_j_status"] in ("interval", "empty", "indeterminate")
        assert diag["campaign_k_status"] in ("interval", "empty", "indeterminate")


def test_e4_criteo_rho_ref_squared_stress_point_and_gate():
    """rho_ref**2 is injected into the grid dynamically and flagged; the gate (no well-specified row ever wrong-
    certifies) is checked and surfaced in the output.
    """
    result = run(n_reps=3, n_checkpoints=6)
    assert "rho_ref_squared" in result
    assert any(res["is_rho_ref_squared_stress_point"] for res in result["results"].values())
    assert result["robust_gate_violations"] == []
    for res in result["results"].values():
        assert "expected_post_thinning_gap" in res
        assert "expected_gap_below_neg_delta" in res
    assert "smallest_prespecified_grid_point_with_naive_reversal_ge_50pct" in result
