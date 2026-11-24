import json

from anytime_attribution.experiments.e4_confident_reversal import (
    run, A_TRUE, J, KK, R_GRID, MISSPECIFIED_R_TRUE, MISSPECIFIED_GAMMA_USED_GRID,
)

_FAST_MISSPEC = dict(misspec_n_looks=2000, misspec_n_reps=3, misspec_checkpoints=(500, 2000))


def test_e4_true_ordering_and_regimes():
    assert A_TRUE[J] > A_TRUE[KK]
    # module docstring's three-regime design: r=1.25 resumes, r=1.5/2.0 don't
    ratio = A_TRUE[J] / A_TRUE[KK]
    resumes_flags = [r <= ratio / r for r in R_GRID]
    assert resumes_flags == [True, False, False]


def test_e4_robust_never_wrong_small_run():
    """robust never certifies the wrong order, regardless of sample size, even in a short run that resolves nothing."""
    result = run(n_reps=5, n_looks=2000, **_FAST_MISSPEC)
    for r_key, res in result["results"].items():
        for L, rate in res["robust_wrong_rate"].items():
            assert rate == 0.0, f"robust wrongly certified at r={r_key}, look={L}"


def test_e4_resumes_flag_is_native_bool_and_survives_json_roundtrip():
    """Regression guard: `resumes` must be a native Python bool before
    dumping (numpy bool_ falls through json.dump's default=str, and the
    resulting string "False" round-trips truthy in Python)."""
    result = run(n_reps=3, n_looks=500, **_FAST_MISSPEC)
    for res in result["results"].values():
        assert isinstance(res["resumes"], bool)

    dumped = json.loads(json.dumps(result, default=str))
    for res in dumped["results"].values():
        assert res["resumes"] in (True, False)  # not the strings "True"/"False"


def test_e4_gamma_used_is_well_specified_and_ceiling_renamed():
    """Each row is well-specified (Gamma_used == r) and the population ceiling uses its own name."""
    result = run(n_reps=3, n_looks=500, **_FAST_MISSPEC)
    for r_key, res in result["results"].items():
        assert res["gamma_used"] == float(r_key)
        assert "population_ceiling_rho_over_r" in res
        assert "gamma_star_theory" not in res


def test_e4_naive_is_production_cs_at_gamma1_not_legacy_wsr():
    """The headline naive_* fields come from the cumulative-share CS robust rule at Gamma=1 (as in
    e4_criteo_intervention.py), not from the fixed-mean WSR result, which sits under the labelled `legacy_naive_wsr`
    key.
    """
    result = run(n_reps=3, n_looks=500, **_FAST_MISSPEC)
    assert result["cs_method"] == "howard_eb_ratio"
    assert "pairwise_multiplicity_convention" in result
    for res in result["results"].values():
        assert "naive_correct_rate" in res and "naive_unresolved_rate" in res
        assert "production CS" in res["naive_label"]
        assert "legacy_naive_wsr" in res
        assert "LEGACY" in res["legacy_naive_wsr"]["label"]


def test_e4_misspecified_demo_r_true_3_population_precomputed_and_labelled():
    """Misspecification demo: r_true=3.0, Gamma_used in {1.1,1.5,2.0,3.0}, with population-level quantities reported
    before simulation.
    """
    result = run(n_reps=3, n_looks=500, **_FAST_MISSPEC)
    ms = result["misspecified_r_true_3"]
    pop = ms["population"]
    assert pop["r_true"] == MISSPECIFIED_R_TRUE == 3.0
    assert "true_full_observability_gap" in pop
    assert "population_post_thinning_observed_gap" in pop
    # r_true=3 is a genuine population reversal: a_j_true < a_k_true
    assert pop["a_j_true"] < pop["a_k_true"]
    assert pop["population_post_thinning_observed_gap"] < 0

    by_gamma = ms["by_gamma_used"]
    assert set(by_gamma.keys()) == {str(g) for g in MISSPECIFIED_GAMMA_USED_GRID}
    for g in MISSPECIFIED_GAMMA_USED_GRID:
        m = by_gamma[str(g)]
        assert m["gamma_assumed"] == g
        assert m["r_true"] == MISSPECIFIED_R_TRUE
        assert m["well_specified"] == (g >= MISSPECIFIED_R_TRUE)
        pg = pop["per_gamma_used"][g]
        assert "population_robust_lower_gap_k_over_j" in pg

    # the well-specified row (Gamma_used=3.0 >= r_true=3.0) must never wrong-certify
    well_specified_row = by_gamma["3.0"]
    for rate in well_specified_row["robust_wrong_rate"].values():
        assert rate == 0.0
