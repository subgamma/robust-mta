from anytime_attribution.experiments.e2_optional_stopping import run, A_TRUE, J, KK


def test_e2_true_delta_is_zero():
    """E2 needs a genuine null (true tie) to test optional-stopping bias."""
    assert A_TRUE[J] == A_TRUE[KK]


def test_e2_cs_methods_never_falsely_stop_in_small_run():
    result = run(n_reps=30, n_looks=300)
    assert result["methods"]["naive_cs"]["wrong_rate_overall"] == 0.0
    assert result["methods"]["robust_cs"]["wrong_rate_overall"] == 0.0


def test_e2_result_has_all_methods():
    result = run(n_reps=10, n_looks=100)
    expected = {"bootstrap", "fixed_eb", "csranks_style", "naive_cs", "robust_cs"}
    assert set(result["methods"].keys()) == expected
    for m in expected:
        rate = result["methods"][m]["wrong_rate_overall"]
        assert 0.0 <= rate <= 1.0
