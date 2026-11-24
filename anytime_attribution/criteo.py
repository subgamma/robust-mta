"""Criteo Attribution Modeling for Bidding dataset: ingestion, journey reconstruction, splits and the cap rule.

The cap (Assumption A4) and the top-K campaign set are fixed from days 1-7 only,
before the monitored stream (days 8-30) is read. The config hash is logged with
every derived artifact.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import polars as pl
import yaml

from .journeys import Journey, Touch, apply_cap as apply_cap_to_journey

DAY_SECONDS = 86400

RAW_COLUMNS = [
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp",
    "conversion_id", "attribution", "click", "click_pos", "click_nb",
    "cost", "cpo", "time_since_last_click",
    "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9",
]


@dataclass(frozen=True)
class FilterStep:
    name: str
    value: int


ROW_ORD = "row_ord"
# Deterministic order (see REPRODUCIBILITY.md): Polars does not guarantee group_by/unique order and many journeys
# share a conversion timestamp, so journeys sort by (conversion timestamp, smallest row ordinal), touches by
# (timestamp, row ordinal), and sampled populations are sorted before sampling.
ORDER_RULE = "conv_time_then_row_ordinal_v1"


def load_raw(path: str) -> pl.LazyFrame:
    """Lazy-load the raw tsv.gz and verify the expected schema."""
    lf = pl.scan_csv(path, separator="\t")
    cols = lf.collect_schema().names()
    missing = set(RAW_COLUMNS) - set(cols)
    if missing:
        raise ValueError(f"unexpected Criteo schema: missing columns {missing}")
    # Row ordinal in the released file: the only tie-breaking key (see ORDER_RULE).
    return lf.with_row_index(ROW_ORD)


def converting_touches(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Touch rows on converting journeys (conversion=1; every such row also has click=1), with a 1-indexed conversion day."""
    return lf.filter(pl.col("conversion") == 1).with_columns(
        (pl.col("conversion_timestamp") // DAY_SECONDS + 1).alias("conv_day")
    )


def journey_key(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Journeys are keyed by (uid, conversion_id): conversion_id alone is not unique across users (2,910 of 435,810 ids
    are shared).
    """
    return lf.with_columns(
        pl.struct(["uid", "conversion_id"]).alias("journey_id")
    )


def top_k_campaigns(warmup_touches: pl.LazyFrame, k: int) -> list[int]:
    """Top-k campaigns by distinct converting-journey count, from days 1-7 touches only."""
    lf = journey_key(warmup_touches)
    ranked = (
        lf.group_by("campaign")
        .agg(pl.col("journey_id").n_unique().alias("n_journeys"))
        .sort("n_journeys", descending=True)
        .collect()
    )
    return ranked["campaign"].head(k).to_list()


def journey_length_and_tau_stats(warmup_touches: pl.LazyFrame) -> dict:
    """Quantiles used to fix the A4 cap (lookback_days, max_touches) from warm-up data only."""
    lf = journey_key(warmup_touches)
    per_journey = lf.group_by("journey_id").agg(pl.len().alias("n_touches")).collect()
    tau = lf.with_columns(
        ((pl.col("conversion_timestamp") - pl.col("timestamp")) / DAY_SECONDS).alias("tau_days")
    ).select("tau_days").collect()
    qs = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    return {
        "n_touches_quantiles": {q: float(per_journey["n_touches"].quantile(q)) for q in qs},
        "tau_days_quantiles": {q: float(tau["tau_days"].quantile(q)) for q in qs},
        "n_warmup_journeys": per_journey.shape[0],
    }


def build_journeys(
    touches: pl.LazyFrame,
    campaign_index: dict[int, int],
    lookback_days: float,
    max_touches: int,
) -> list[Journey]:
    """Group touches into `Journey` objects (linear weight), restrict to `campaign_index` and apply the A4 cap.

    Journeys with no touch left after the campaign restriction are dropped.
    """
    lf = journey_key(touches).filter(pl.col("campaign").is_in(list(campaign_index)))
    grouped = (
        lf.group_by("journey_id")
        .agg([
            pl.col("conversion_timestamp").first().alias("conversion_timestamp"),
            pl.col(ROW_ORD).min().alias(ROW_ORD),
            pl.struct(["campaign", "timestamp", ROW_ORD]).alias("touch_list"),
        ])
        .sort(["conversion_timestamp", ROW_ORD])
        .collect()
    )
    out = []
    for row in grouped.iter_rows(named=True):
        # Timestamps in days so lookback_days and the time-decay half-life share units.
        conv_time = float(row["conversion_timestamp"]) / DAY_SECONDS
        raw_touches = [
            (campaign_index[t["campaign"]], float(t["timestamp"]) / DAY_SECONDS, t[ROW_ORD])
            for t in row["touch_list"]
        ]
        touches_sorted = [(c, ts) for c, ts, _ in sorted(raw_touches, key=lambda x: (x[1], x[2]))]
        touch_objs = tuple(
            Touch(campaign=c, timestamp=ts, weight=1.0) for c, ts in touches_sorted
        )
        journey = Journey(touches=touch_objs, conversion_time=conv_time)
        capped = apply_cap_to_journey(journey, lookback_days=lookback_days, max_touches=max_touches)
        if capped.n_touches > 0:
            out.append(capped)
    return out


def build_negative_examples(
    lf: pl.LazyFrame,
    campaign_index: dict[int, int],
    day_lo: int,
    day_hi: int,
    max_touches: int,
    n_sample: int,
    seed: int = 0,
) -> list[Journey]:
    """Non-converting users' click histories as pseudo-journeys, for training the logistic and DARNN models only.

    Each is one user's clicks on the top-K campaigns within [day_lo, day_hi], the
    most recent `max_touches` kept, with `conversion_time` set to the last click as a
    placeholder anchor so `campaign_masses` can compute count features. `n_sample`
    users are sampled uniformly from those active in the window.
    """
    day_col = (pl.col("timestamp") // DAY_SECONDS + 1)
    pool = (
        lf.filter(
            (pl.col("conversion") == 0)
            & (pl.col("click") == 1)
            & (pl.col("campaign").is_in(list(campaign_index)))
            & day_col.is_between(day_lo, day_hi)
        )
    )
    uids = pool.select(pl.col("uid").unique().sort()).collect()["uid"]  # sorted: explicit population order
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, len(uids))
    sampled_uids = rng.choice(uids.to_numpy(), size=n_sample, replace=False).tolist()

    grouped = (
        pool.filter(pl.col("uid").is_in(sampled_uids))
        .group_by("uid")
        .agg(pl.struct(["campaign", "timestamp", ROW_ORD]).alias("touch_list"))
        .sort("uid")
        .collect()
    )
    out = []
    for row in grouped.iter_rows(named=True):
        raw_touches = [
            (campaign_index[t["campaign"]], float(t["timestamp"]) / DAY_SECONDS, t[ROW_ORD])
            for t in row["touch_list"]
        ]
        touches_sorted = [(c, ts) for c, ts, _ in sorted(raw_touches, key=lambda x: (x[1], x[2]))][-max_touches:]
        touch_objs = tuple(Touch(campaign=c, timestamp=ts, weight=1.0) for c, ts in touches_sorted)
        conv_time = touches_sorted[-1][1]
        out.append(Journey(touches=touch_objs, conversion_time=conv_time))
    return out


# Secondary (literature-convention linkage) construction, as in CAMTA/CausalMTA: top-K most exposed campaigns by
# impression count (warm-up), all of a user's top-K touches in the lookback window regardless of conversion_id,
# max_touches cap and a minimum-touch filter applied to both positive and negative examples. Artifacts built here must
# be labelled "literature-convention linkage".


def top_k_by_exposure(warmup_lf: pl.LazyFrame, k: int) -> list[int]:
    """Top-k campaigns by raw impression count in warm-up (secondary construction); `top_k_campaigns` ranks by
    conversions.
    """
    ranked = (
        warmup_lf.group_by("campaign")
        .agg(pl.len().alias("n_impressions"))
        .sort("n_impressions", descending=True)
        .collect()
    )
    return ranked["campaign"].head(k).to_list()


def _windowed_campaign_journeys(
    lf: pl.LazyFrame,
    campaign_index: dict[int, int],
    conv_events: pl.DataFrame,
    lookback_days: float,
    max_touches: int,
    min_touches: int,
) -> list[Journey]:
    """Shared by `build_secondary_journeys` and `build_secondary_negative_examples`.

    For each anchor event (uid, journey_id, anchor_timestamp) in raw seconds, gathers
    the uid's touches on top-K campaigns within [anchor - lookback, anchor], applies
    the max-touch cap and keeps journeys with more than `min_touches` touches.
    """
    touch_pool = (
        lf.filter(pl.col("campaign").is_in(list(campaign_index)))
        .select(["uid", "campaign", "timestamp", ROW_ORD])
        .collect()
    )
    lookback_seconds = lookback_days * DAY_SECONDS
    joined = conv_events.join(touch_pool, on="uid", how="inner")
    windowed = joined.filter(
        (pl.col("timestamp") <= pl.col("anchor_timestamp"))
        & (pl.col("timestamp") >= pl.col("anchor_timestamp") - lookback_seconds)
    )
    grouped = windowed.group_by("journey_id").agg([
        pl.col("anchor_timestamp").first().alias("anchor_timestamp"),
        pl.col("event_ord").first().alias("event_ord"),
        pl.struct(["campaign", "timestamp", ROW_ORD]).alias("touch_list"),
    ]).sort(["anchor_timestamp", "event_ord"])
    out = []
    for row in grouped.iter_rows(named=True):
        conv_time = float(row["anchor_timestamp"]) / DAY_SECONDS
        raw_touches = [
            (campaign_index[t["campaign"]], float(t["timestamp"]) / DAY_SECONDS, t[ROW_ORD])
            for t in row["touch_list"]
        ]
        touches_sorted = [(c, ts) for c, ts, _ in sorted(raw_touches, key=lambda x: (x[1], x[2]))][-max_touches:]
        if len(touches_sorted) <= min_touches:
            continue
        touch_objs = tuple(Touch(campaign=c, timestamp=ts, weight=1.0) for c, ts in touches_sorted)
        out.append(Journey(touches=touch_objs, conversion_time=conv_time))
    return out


def build_secondary_journeys(
    lf: pl.LazyFrame,
    campaign_index: dict[int, int],
    day_lo: int,
    day_hi: int,
    lookback_days: float,
    max_touches: int,
    min_touches: int = 3,
) -> list[Journey]:
    """Literature-convention (CAMTA/CausalMTA-style) converting journeys.

    Conversions whose own campaign is in the top-K exposure set, linked to all of that
    user's top-K touches in the lookback window (by time proximity, not conversion_id).
    See the secondary-construction note above for the choices made.
    """
    conv_events = (
        lf.filter(
            (pl.col("conversion") == 1)
            & (pl.col("campaign").is_in(list(campaign_index)))
            & ((pl.col("conversion_timestamp") // DAY_SECONDS + 1).is_between(day_lo, day_hi))
        )
        .with_columns(pl.struct(["uid", "conversion_id"]).alias("journey_id"))
        .group_by(["uid", "journey_id", pl.col("conversion_timestamp").alias("anchor_timestamp")])
        .agg(pl.col(ROW_ORD).min().alias("event_ord"))
        .sort(["anchor_timestamp", "event_ord"])
        .collect()
    )
    return _windowed_campaign_journeys(lf, campaign_index, conv_events, lookback_days, max_touches, min_touches)


def build_secondary_negative_examples(
    lf: pl.LazyFrame,
    campaign_index: dict[int, int],
    day_lo: int,
    day_hi: int,
    lookback_days: float,
    max_touches: int,
    min_touches: int,
    n_sample: int,
    seed: int = 0,
) -> list[Journey]:
    """Literature-convention non-converting pseudo-journeys.

    For a random sample of non-converting users active on a top-K campaign, anchors at
    their last such impression and gathers top-K touches in the trailing window, with
    the same cap and min-touch filter as the positive construction.
    """
    day_col = pl.col("timestamp") // DAY_SECONDS + 1
    active = (
        lf.filter(
            (pl.col("conversion") == 0)
            & (pl.col("campaign").is_in(list(campaign_index)))
            & day_col.is_between(day_lo, day_hi)
        )
        .select(["uid", "timestamp", ROW_ORD])
        .collect()
    )
    uids = active["uid"].unique().sort()  # sorted: explicit population order
    rng = np.random.default_rng(seed)
    n_sample = min(n_sample, len(uids))
    sampled_uids = rng.choice(uids.to_numpy(), size=n_sample, replace=False).tolist()

    anchors = (
        active.filter(pl.col("uid").is_in(sampled_uids))
        .group_by("uid")
        .agg([pl.col("timestamp").max().alias("anchor_timestamp"), pl.col(ROW_ORD).min().alias("event_ord")])
        .with_columns(pl.col("uid").alias("journey_id"))
    )
    return _windowed_campaign_journeys(lf, campaign_index, anchors, lookback_days, max_touches, min_touches)


def config_hash(config: dict) -> str:
    """Stable content hash of a config dict."""
    blob = yaml.dump(config, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
