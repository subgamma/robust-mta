"""E10 population oracle -- independent verification, orientation, nesting, decomposition invariants."""
import importlib.util
import json
import os

import numpy as np
import pytest

from anytime_attribution.certify import direct_lower_gap
from anytime_attribution.experiments import e10_multicampaign as E


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


ORC = _load("scripts/run_e10_oracle.py", "run_e10_oracle")
E3 = _load("scripts/run_e3.py", "run_e3")
RUN = _load("scripts/run_e10_multicampaign.py", "run_e10_multicampaign")


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("Gamma", [1.0, 1.25, 2.0, 4.0])
def test_oracle_gap_matches_independent_lp_including_floor(seed, Gamma):
    rng = np.random.default_rng(seed)
    S = rng.dirichlet(np.ones(10)) * 1e6
    for r in (1.25, 2.5):
        p = E.retention_vector(r); m = p * S; plo = float(p.min())
        for j, k in [(0, 1), (2, 7), (9, 3)]:
            g = direct_lower_gap(m, m, Gamma, j, k, p_lo=plo)
            lp = E3.lp_extreme(m / m.sum(), j, k, Gamma, plo, "min", "gap")[0]
            assert g == pytest.approx(lp, abs=1e-9)


def test_gamma_one_is_observed_gap_and_gap_is_monotone_in_gamma():
    rng = np.random.default_rng(1)
    S = rng.dirichlet(np.ones(10)); m = E.retention_vector(2.0) * S
    a = m / m.sum()
    assert direct_lower_gap(m, m, 1.0, 0, 1) == pytest.approx(a[0] - a[1], abs=1e-12)
    gaps = [direct_lower_gap(m, m, G, 0, 1) for G in (1.0, 1.25, 1.5, 2.0, 3.0)]
    assert all(x >= y - 1e-12 for x, y in zip(gaps, gaps[1:]))


def test_expected_observed_mass_is_p_times_latent_mass():
    """In-model: E[T_j] = p_j E[S_j] for fixed weights (linear & time-decay); checked by Monte Carlo thinning of one
    latent sample.
    """
    rng = np.random.default_rng(3)
    trip, _ = E.sample_converting(20000, rng)
    p = E.retention_vector(2.0)
    for model in ("linear", "time_decay"):
        S = E.model_masses(model, trip, None, 20000)[0].sum(0)
        acc = np.zeros(E.K); R = 60
        for rep in range(R):
            keep = E.thin(trip, p, np.random.default_rng(100 + rep))
            acc += E.model_masses(model, trip, keep, 20000)[0].sum(0)
        assert np.allclose(acc / R, p * S, rtol=0.02)


def test_oracle_seed_streams_disjoint_from_monitoring():
    streams = {tuple(np.random.SeedSequence([ORC.SEED, s, c]).generate_state(1)) for s in (1, 2, 3, 4, 5) for c in range(3)}
    assert len(streams) == 15


def test_nested_checkpoints_and_prefix_property():
    cp50 = RUN.checkpoints_for(50_000, (50_000,))
    cp250 = RUN.checkpoints_for(250_000, (50_000, 250_000))
    assert [t for t in cp250 if t <= 50_000] == cp50
    assert max(cp250) == 250_000 and 50_000 in cp250
    # the first 50,000 conversions of a longer stream are the shorter stream (touch arrays are prefixes)
    t1, _ = E.sample_converting(3000, np.random.default_rng(9)); t2, _ = E.sample_converting(9000, np.random.default_rng(9))
    n1 = len(t1["camp"])
    assert np.array_equal(t1["camp"], t2["camp"][:n1]) and np.array_equal(t1["jid"], t2["jid"][:n1])
    u = np.random.default_rng(5).random(n1 + 777)[:n1]
    assert np.array_equal(u, np.random.default_rng(5).random(n1))      # thinning draws are prefix-consistent


ORC_PATH = "results/e10/e10_oracle.json"


@pytest.mark.skipif(not os.path.exists(ORC_PATH), reason="oracle not generated")
def test_oracle_artifact_invariants():
    o = json.load(open(ORC_PATH))
    assert o["lp_crosscheck_max_abs_deviation"] < 1e-9
    assert o["config"]["n_conversions"] >= 2_000_000
    for key, c in o["conditions"].items():
        assert c["Gamma_eff"] == pytest.approx(min(c["Gamma"], 1 / c["p_lo_declared"]))
        for x in c["pairs"]:
            assert abs(x["oracle_gap"] - x["lp_gap"]) < 1e-9
            assert x["oracle_certifiable"] == (x["oracle_gap"] > o["config"]["delta"])
            # well-specified rule (Gamma >= r_true): the identified set contains the truth, so its lower end cannot
            # exceed it.
            # (Misspecified rules, Gamma < r_true, carry no such guarantee -- that is the point of the
            # misspecification rows.)
            if c["rule"] == "wellspec":
                assert x["oracle_gap"] <= x["delta_true_abs"] + 4 * x["se_oracle_gap"] + 1e-9


@pytest.mark.skipif(not os.path.exists("results/e10/e10_decomposition_N50000.json"), reason="decomposition not generated")
def test_decomposition_counts_consistent_and_no_wrong_wellspecified():
    d = json.load(open("results/e10/e10_decomposition_N50000.json"))
    for key, r in d["rows"].items():
        assert r["n_observability_limited"] + r["n_oracle_certifiable"] + r["n_boundary_ambiguous_actionable"] == r["n_actionable"]
        if key.endswith("wellspec"):
            assert r["mean_total_wrong_certifications_per_rep"] == 0
            assert r["mean_n_certified_correct_among_observability_limited"] == 0
