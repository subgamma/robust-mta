import polars as pl
import pytest

from anytime_attribution.criteo import (
    RAW_COLUMNS,
    load_raw,
    converting_touches,
    journey_key,
    top_k_campaigns,
    top_k_by_exposure,
    build_journeys,
    build_secondary_journeys,
    build_secondary_negative_examples,
    config_hash,
)

DAY = 86400


def _write_synthetic_csv(path):
    """A tiny synthetic Criteo-shaped dataset: two users, one with a
    2-touch converting journey, one with a colliding conversion_id across
    users (regression guard for the (uid, conversion_id) grouping bug)."""
    rows = [
        # uid, campaign, conversion, conv_ts, conv_id, click, timestamp
        (0, 100, 1, 6 * DAY, 900, 1, 4 * DAY),  # conv_day=7 (warmup); tau=2 days
        (0, 200, 1, 6 * DAY, 900, 1, 5 * DAY),  # tau=1 day
        (1, 100, 1, 5 * DAY, 900, 1, 4 * DAY),  # same conversion_id, different uid
        (2, 300, 0, -1, -1, 0, 1 * DAY),  # non-converting, excluded from journeys
        (3, 400, 1, 40 * DAY, 901, 1, 39 * DAY),  # after monitored window
    ]
    header = "timestamp\tuid\tcampaign\tconversion\tconversion_timestamp\tconversion_id\tattribution\tclick\tclick_pos\tclick_nb\tcost\tcpo\ttime_since_last_click\t" + "\t".join(f"cat{i}" for i in range(1, 10))
    lines = [header]
    for uid, campaign, conversion, conv_ts, conv_id, click, ts in rows:
        lines.append(
            f"{ts}\t{uid}\t{campaign}\t{conversion}\t{conv_ts}\t{conv_id}\t0\t{click}\t-1\t-1\t0.0\t0.0\t-1\t" + "\t".join("0" for _ in range(9))
        )
    path.write_text("\n".join(lines))


def test_load_raw_verifies_schema(tmp_path):
    p = tmp_path / "data.tsv"
    _write_synthetic_csv(p)
    lf = load_raw(str(p))
    assert set(RAW_COLUMNS).issubset(set(lf.collect_schema().names()))


def test_load_raw_rejects_bad_schema(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("a\tb\n1\t2\n")
    with pytest.raises(ValueError):
        load_raw(str(p))


def test_journey_key_disambiguates_colliding_conversion_ids(tmp_path):
    """Regression guard: conversion_id 900 is shared by uid 0 and uid 1 in
    the synthetic fixture; they must be treated as two distinct journeys."""
    p = tmp_path / "data.tsv"
    _write_synthetic_csv(p)
    lf = load_raw(str(p))
    touches = converting_touches(lf)
    keyed = journey_key(touches)
    n_journeys = keyed.select(pl.col("journey_id").n_unique()).collect().item()
    assert n_journeys == 3  # (uid=0,cid=900), (uid=1,cid=900), (uid=3,cid=901)


def test_top_k_campaigns_ranks_by_distinct_journeys(tmp_path):
    p = tmp_path / "data.tsv"
    _write_synthetic_csv(p)
    lf = load_raw(str(p))
    touches = converting_touches(lf)
    warmup = touches.filter(pl.col("conv_day") <= 7)
    top = top_k_campaigns(warmup, k=2)
    assert set(top) == {100, 200}  # campaign 400's journey converts on day 40


def test_build_journeys_applies_cap_and_campaign_restriction(tmp_path):
    p = tmp_path / "data.tsv"
    _write_synthetic_csv(p)
    lf = load_raw(str(p))
    touches = converting_touches(lf)
    campaign_index = {100: 0, 200: 1}
    journeys = build_journeys(touches, campaign_index, lookback_days=3, max_touches=10)
    # (uid=0,cid=900): touches at day 8 (tau=2) and day 9 (tau=1) -> both within lookback=3
    # (uid=1,cid=900): touch at day 4, conv at day 5 (tau=1) -> within lookback
    # (uid=3,cid=901): campaign 400 not in index -> dropped entirely
    assert len(journeys) == 2
    n_touches = sorted(j.n_touches for j in journeys)
    assert n_touches == [1, 2]


def test_build_journeys_lookback_units_are_days_not_seconds(tmp_path):
    """Regression guard for the unit-mismatch bug: lookback_days=3 must
    compare against day-scaled tau, not raw-second differences."""
    p = tmp_path / "data.tsv"
    _write_synthetic_csv(p)
    lf = load_raw(str(p))
    touches = converting_touches(lf)
    campaign_index = {100: 0, 200: 1}
    journeys = build_journeys(touches, campaign_index, lookback_days=3, max_touches=10)
    assert sum(j.n_touches for j in journeys) == 3  # none dropped by a 3-day lookback here

    tight = build_journeys(touches, campaign_index, lookback_days=1.5, max_touches=10)
    # (uid=0,cid=900) touch at tau=2 days is now dropped by a 1.5-day lookback
    assert sum(j.n_touches for j in tight) == 2


def test_config_hash_deterministic_and_sensitive_to_content():
    cfg1 = {"a": 1, "b": [1, 2, 3]}
    cfg2 = {"b": [1, 2, 3], "a": 1}  # different key order, same content
    cfg3 = {"a": 1, "b": [1, 2, 4]}
    assert config_hash(cfg1) == config_hash(cfg2)
    assert config_hash(cfg1) != config_hash(cfg3)


# Secondary (literature-convention linkage) construction.
def _write_secondary_csv(path):
    """uid=10: an impression on campaign 500 (not clicked) and a click on
    campaign 500, then two clicks on campaign 600, converting on campaign
    600 -- 4 touches within lookback, 2 distinct campaigns, cross-campaign
    only via time proximity (the campaign-500 touches are not part of
    conversion_id 950). uid=11: only 2 touches -- must be excluded by the
    >3-touchpoint filter. uid=12: non-converting, 4 top-K touches -- a
    candidate negative example.
    """
    rows = [
        # uid, campaign, conversion, conv_ts, conv_id, click, timestamp
        (10, 500, 0, -1, -1, 0, 1 * DAY),        # impression only, not clicked
        (10, 500, 0, -1, -1, 1, 1.5 * DAY),      # click, non-converting row
        (10, 600, 0, -1, -1, 1, 3 * DAY),        # click, non-converting row
        (10, 600, 1, 4 * DAY, 950, 1, 3.5 * DAY),  # the conversion touch itself
        (11, 500, 0, -1, -1, 1, 1 * DAY),
        (11, 500, 1, 2 * DAY, 951, 1, 1.5 * DAY),  # only 2 touches -> excluded
        (12, 500, 0, -1, -1, 1, 1 * DAY),
        (12, 500, 0, -1, -1, 1, 1.2 * DAY),
        (12, 600, 0, -1, -1, 1, 1.4 * DAY),
        (12, 600, 0, -1, -1, 1, 1.6 * DAY),
    ]
    header = "timestamp\tuid\tcampaign\tconversion\tconversion_timestamp\tconversion_id\tattribution\tclick\tclick_pos\tclick_nb\tcost\tcpo\ttime_since_last_click\t" + "\t".join(f"cat{i}" for i in range(1, 10))
    lines = [header]
    for uid, campaign, conversion, conv_ts, conv_id, click, ts in rows:
        lines.append(
            f"{ts}\t{uid}\t{campaign}\t{conversion}\t{conv_ts}\t{conv_id}\t0\t{click}\t-1\t-1\t0.0\t0.0\t-1\t" + "\t".join("0" for _ in range(9))
        )
    path.write_text("\n".join(lines))


def test_top_k_by_exposure_ranks_by_impressions_not_conversions(tmp_path):
    p = tmp_path / "data.tsv"
    _write_secondary_csv(p)
    lf = load_raw(str(p))
    warmup = lf.filter((pl.col("timestamp") // DAY + 1).is_between(1, 7))
    # campaign 500: 5 impressions (uid10 x2, uid11 x1, uid12 x2); campaign 600: 4 (uid10 x2, uid12 x2)
    top = top_k_by_exposure(warmup, k=2)
    assert set(top) == {500, 600}
    top1 = top_k_by_exposure(warmup, k=1)
    assert top1 == [500]


def test_build_secondary_journeys_links_across_campaigns_by_time(tmp_path):
    p = tmp_path / "data.tsv"
    _write_secondary_csv(p)
    lf = load_raw(str(p))
    campaign_index = {500: 0, 600: 1}
    journeys = build_secondary_journeys(
        lf, campaign_index, day_lo=1, day_hi=7, lookback_days=7, max_touches=10, min_touches=3
    )
    # only uid=10's journey survives the >3-touchpoint filter (uid=11 has 2 touches)
    assert len(journeys) == 1
    j = journeys[0]
    assert j.n_touches == 4
    touched_campaigns = {t.campaign for t in j.touches}
    assert touched_campaigns == {0, 1}  # cross-campaign linkage: campaign 500 AND 600


def test_build_secondary_journeys_respects_min_touches_filter(tmp_path):
    p = tmp_path / "data.tsv"
    _write_secondary_csv(p)
    lf = load_raw(str(p))
    campaign_index = {500: 0, 600: 1}
    # min_touches=4 excludes uid=10's exactly-4-touch journey too (not > 4)
    journeys = build_secondary_journeys(
        lf, campaign_index, day_lo=1, day_hi=7, lookback_days=7, max_touches=10, min_touches=4
    )
    assert len(journeys) == 0


def test_build_secondary_negative_examples_uses_non_converting_users(tmp_path):
    p = tmp_path / "data.tsv"
    _write_secondary_csv(p)
    lf = load_raw(str(p))
    campaign_index = {500: 0, 600: 1}
    neg = build_secondary_negative_examples(
        lf, campaign_index, day_lo=1, day_hi=7, lookback_days=7, max_touches=10,
        min_touches=3, n_sample=10, seed=0,
    )
    # only uid=12 is non-converting with >3 top-K touches
    assert len(neg) == 1
    assert neg[0].n_touches == 4


def _tied_tsv(path, order):
    """Journeys with identical conversion timestamps; `order` permutes the uids' row order in the file."""
    header = "timestamp\tuid\tcampaign\tconversion\tconversion_timestamp\tconversion_id\tattribution\tclick\tclick_pos\tclick_nb\tcost\tcpo\ttime_since_last_click\t" + "\t".join(f"cat{i}" for i in range(1, 10))
    lines = [header]
    for uid in order:
        camp = 100 + (uid % 3)
        lines.append(f"{DAY}\t{uid}\t{camp}\t1\t{3 * DAY}\t{uid}\t0\t1\t-1\t-1\t0.0\t0.0\t-1\t" + "\t".join("0" for _ in range(9)))
    path.write_text("\n".join(lines))


def test_journey_order_follows_row_ordinal_among_tied_conversion_times(tmp_path):
    """All journeys convert at the same second; the output order must be the file's row order, every time."""
    from anytime_attribution.criteo import ROW_ORD

    order = [7, 2, 9, 4, 1, 8, 3, 6, 5, 0]
    p = tmp_path / "t.tsv"
    _tied_tsv(p, order)
    lf = load_raw(str(p))
    assert lf.select(ROW_ORD).collect()[ROW_ORD].to_list() == list(range(len(order)))
    index = {100: 0, 101: 1, 102: 2}
    runs = [[j.touches[0].campaign for j in build_journeys(converting_touches(lf), index, 7.0, 10)] for _ in range(3)]
    expected = [100 + (uid % 3) for uid in order]
    expected = [index[c] for c in expected]
    assert runs[0] == runs[1] == runs[2] == expected
