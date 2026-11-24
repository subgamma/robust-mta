from anytime_attribution.experiments.e1_coverage import run, METHODS


def test_e1_robust_cs_meets_acceptance_bar():
    """acceptance: robust CS >= 0.95 at every look count,
    for BOTH marginal and simultaneous tables. Small n_reps for test speed;
    the full run (200 reps) is in results/e1/."""
    result = run(n_reps=10, reported_looks=(1, 10), finest_l=20)
    for L, cov in result["marginal_table"]["robust_cs"].items():
        assert cov >= 0.95, f"robust_cs marginal failed at look={L}: {cov}"
    for L, cov in result["simultaneous_table"]["robust_cs"].items():
        assert cov >= 0.95, f"robust_cs simultaneous failed at look={L}: {cov}"


def test_e1_cs_methods_identical_across_L_by_anytime_validity():
    """naive_cs/robust_cs are checked continuously (1..N_FULL); the number
    of discrete looks recorded cannot change the verdict at N_FULL."""
    result = run(n_reps=5, reported_looks=(1, 10), finest_l=20)
    for method in ["naive_cs", "robust_cs"]:
        marg_vals = set(result["marginal_table"][method].values())
        simul_vals = set(result["simultaneous_table"][method].values())
        assert len(marg_vals) == 1, f"{method} marginal varies across L: {result['marginal_table'][method]}"
        assert len(simul_vals) == 1, f"{method} simultaneous varies across L: {result['simultaneous_table'][method]}"


def test_e1_tables_have_all_methods_and_looks():
    result = run(n_reps=5, reported_looks=(1, 10), finest_l=20)
    for table_key in ["marginal_table", "simultaneous_table"]:
        assert set(result[table_key].keys()) == set(METHODS)
        for m in METHODS:
            assert set(result[table_key][m].keys()) == {1, 10}
            for v in result[table_key][m].values():
                assert 0.0 <= v <= 1.0
    assert set(result["bonferroni_status"].keys()) == set(METHODS)
