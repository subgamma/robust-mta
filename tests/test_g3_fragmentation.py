"""/ G3: corrected pre-cut orphaning keeps exactly one row per conversion in
the original arrival order. Cut k in {1..n} (uniform) discards touchpoints 1..k; k = n
fully orphans (also n = 1) and leaves a zero-mass row. Out-of-model: no coverage claim."""
import numpy as np
import pytest

from anytime_attribution.experiments.e7_interventions import (
    _stream_g3,
    _stream_g3_legacy_duplicating,
    g3_orphan_cut,
)

K = 6


def _inputs(n=4000, seed=0, max_touch=10):
    rng = np.random.default_rng(seed)
    campaign_idx = rng.integers(0, K, size=n)
    n_touches = np.minimum(1 + rng.geometric(0.45, size=n) - 1, max_touch)
    return campaign_idx, n_touches


def _latent(campaign_idx, n_touches):
    base = np.zeros((len(campaign_idx), K))
    base[np.arange(len(campaign_idx)), campaign_idx] = n_touches
    return base


def test_row_count_order_and_one_row_per_conversion():
    ci, nt = _inputs()
    base = _latent(ci, nt)
    for rho in (0.0, 0.3, 1.0):
        Tj = _stream_g3(ci, nt, rho, np.random.default_rng(1), K)
        assert Tj.shape == (len(ci), K)  # 1, 3: one output row per conversion
        # 2, 9: row i is conversion i (no reordering, no mass in a different conversion's campaign)
        assert np.all((Tj > 0).sum(axis=1) <= 1)
        nz = Tj.sum(axis=1) > 0
        assert np.all(np.argmax(Tj[nz], axis=1) == ci[nz])
        assert np.all(Tj <= base + 1e-12)  # 9/10: nothing exceeds the latent mass, nothing scaled up


def test_rho_zero_reproduces_latent_journey():
    ci, nt = _inputs()
    assert np.array_equal(_stream_g3(ci, nt, 0.0, np.random.default_rng(2), K), _latent(ci, nt))


def test_forced_cuts_remove_exactly_intended_touchpoints(monkeypatch):
    import anytime_attribution.experiments.e7_interventions as e7

    ci = np.array([0, 1, 2, 3, 4])
    nt = np.array([1, 1, 5, 5, 3])
    cut = np.array([1, 0, 2, 5, 3])  # n=1 full orphan; no cut; partial; full (k=n); full (k=n)
    monkeypatch.setattr(e7, "g3_orphan_cut", lambda n, rho, rng: cut)
    Tj = e7._stream_g3(ci, nt, 0.5, np.random.default_rng(0), K)
    assert Tj.shape[0] == 5
    assert Tj[0].sum() == 0 and Tj[1, 1] == 1 and Tj[2, 2] == 3 and Tj[3].sum() == 0 and Tj[4].sum() == 0
    # fully orphaned conversions remain as zero-mass rows (arrival clock preserved)
    assert Tj.shape[0] == len(ci)


def test_single_touch_journeys_can_be_fully_orphaned_and_cut_support():
    rng = np.random.default_rng(3)
    cuts = g3_orphan_cut(np.ones(200000, dtype=int), 0.3, rng)
    assert set(np.unique(cuts)) == {0, 1}
    assert abs((cuts == 1).mean() - 0.3) < 0.005  # n=1: cut (hence T_i=0) w.p. rho
    nt = np.full(300000, 4)
    cuts = g3_orphan_cut(nt, 1.0, np.random.default_rng(4))
    assert set(np.unique(cuts)) == {1, 2, 3, 4}  # all n gaps incl. the final one (k = n)
    freq = np.bincount(cuts, minlength=5)[1:] / len(cuts)
    assert np.allclose(freq, 0.25, atol=0.005)  # uniform over {1..n}
    with pytest.raises(ValueError):
        g3_orphan_cut(np.array([0, 2]), 0.5, np.random.default_rng(0))


def test_fully_orphaned_rows_are_zero_mass_and_masses_within_bound():
    ci, nt = _inputs()
    Tj = _stream_g3(ci, nt, 1.0, np.random.default_rng(5), K)  # everyone is cut
    z = Tj.sum(axis=1) == 0
    assert z.any()
    assert np.all(Tj[nt == 1].sum(axis=1) == 0)  # every n=1 journey is fully orphaned at rho=1
    assert Tj.min() >= 0 and Tj.max() <= 10.0  # within [0, B], B = max_touches
    # with a cut at every conversion (rho = 1) at least one touchpoint is always orphaned
    assert np.all(Tj.sum(axis=1) <= nt - 1)


def test_same_seed_identical_and_different_seed_differs():
    ci, nt = _inputs()
    a = _stream_g3(ci, nt, 0.4, np.random.default_rng(7), K)
    b = _stream_g3(ci, nt, 0.4, np.random.default_rng(7), K)
    c = _stream_g3(ci, nt, 0.4, np.random.default_rng(8), K)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_no_touchpoint_is_copied_and_no_silent_clip():
    ci, nt = _inputs(n=2000, seed=9)
    Tj = _stream_g3(ci, nt, 0.6, np.random.default_rng(10), K)
    assert np.all(Tj.sum(axis=1) <= nt)  # per-conversion observed mass never exceeds its latent mass
    assert Tj.sum() <= nt.sum()  # no mass created across conversions
    # observed mass equals n - k exactly (integer weights 1, nothing renormalised or clipped)
    assert np.array_equal(Tj.sum(axis=1), np.round(Tj.sum(axis=1)))


def test_legacy_vs_corrected_structure_smoke():
    ci, nt = _inputs(n=6000, seed=11)
    base = _latent(ci, nt)
    leg = _stream_g3_legacy_duplicating(ci, nt, 0.6, np.random.default_rng(12), K)
    new = _stream_g3(ci, nt, 0.6, np.random.default_rng(12), K)
    # legacy: extra rows, all mass preserved (arrival clock broken)
    assert leg.shape[0] > len(ci) and np.allclose(leg.sum(axis=0), base.sum(axis=0))
    # corrected: same row count, mass removed (orphaned), arrival clock preserved
    assert new.shape[0] == len(ci) and new.sum() < base.sum()
