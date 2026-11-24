"""Primary monitored-window journeys with per-touch timestamps.

E5-E8 need masses under any weight scheme (linear, time-decay). The one-hot credit
vectors in `results/credits/linear/0.parquet` are not enough, because A_j = E[S_j]/E[S]
depends on the per-journey weighted mass. Rebuilding from the 16M-row raw file is
expensive, so the journeys are built once and cached in
`results/primary_journeys_cache.pkl` (git-ignored). The primary construction yields
109,221 journeys: 274,337 monitored converting-touch rows after the top-20 campaign
restriction and the A4 cap.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import polars as pl

from ..criteo import ORDER_RULE, build_journeys, build_secondary_journeys, converting_touches, load_config, load_raw
from ..journeys import Journey, Touch

CACHE_PATH = "results/primary_journeys_cache.pkl"
SECONDARY_CACHE_PATH = "results/secondary_journeys_cache.pkl"
CONFIG_PATH = "config/criteo.yaml"


def build_and_cache_primary_journeys(
    config_path: str = CONFIG_PATH,
    cache_path: str = CACHE_PATH,
    force: bool = False,
) -> tuple[list[Journey], list[int]]:
    """Returns (journeys, top_20), journeys sorted by conversion_time with linear weights.

    Time-decay masses are derived on demand by `time_decay_mass`. Cached after the first build.
    """
    cache = Path(cache_path)
    if cache.exists() and not force:
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if data.get("order_rule") == ORDER_RULE:
            return data["journeys"], data["top_20"]
        # Rebuild a cache made under a different journey-ordering rule (see criteo.ORDER_RULE).

    cfg = load_config(config_path)
    top_20 = cfg["campaigns"]["top_20"]
    lookback_days = cfg["cap"]["lookback_days"]
    max_touches = cfg["cap"]["max_touches"]
    campaign_index = {c: i for i, c in enumerate(top_20)}

    lf = load_raw(cfg["dataset"]["path"])
    touches = converting_touches(lf).filter(pl.col("conv_day").is_between(8, 30))
    journeys = build_journeys(touches, campaign_index, lookback_days=lookback_days, max_touches=max_touches)
    journeys = sorted(journeys, key=lambda j: j.conversion_time)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"journeys": journeys, "top_20": top_20, "order_rule": ORDER_RULE}, f)
    return journeys, top_20


def build_and_cache_secondary_journeys(
    config_path: str = CONFIG_PATH,
    cache_path: str = SECONDARY_CACHE_PATH,
    force: bool = False,
) -> tuple[list[Journey], list[int]]:
    """Secondary (literature-convention linkage) monitored-window (days 8-30) journeys: top-10 campaigns by exposure,
    min_touches=3.

    Returns (journeys sorted by conversion_time, top_10_by_exposure).
    """
    cache = Path(cache_path)
    if cache.exists() and not force:
        with open(cache, "rb") as f:
            data = pickle.load(f)
        if data.get("order_rule") == ORDER_RULE:
            return data["journeys"], data["top_10"]

    cfg = load_config(config_path)
    top_10 = cfg["secondary"]["top_10_by_exposure"]
    lookback_days = cfg["cap"]["lookback_days"]
    max_touches = cfg["cap"]["max_touches"]
    min_touches = cfg["secondary"]["min_touches"]
    campaign_index = {c: i for i, c in enumerate(top_10)}

    lf = load_raw(cfg["dataset"]["path"])
    journeys = build_secondary_journeys(
        lf, campaign_index, day_lo=8, day_hi=30,
        lookback_days=lookback_days, max_touches=max_touches, min_touches=min_touches,
    )
    journeys = sorted(journeys, key=lambda j: j.conversion_time)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump({"journeys": journeys, "top_10": top_10, "order_rule": ORDER_RULE}, f)
    return journeys, top_10


def time_decay_journey(j: Journey, h: float = 7.0) -> Journey:
    """Same touches with weight 2**(-tau/h)."""
    touches = tuple(
        Touch(campaign=t.campaign, timestamp=t.timestamp, weight=2.0 ** (-(j.conversion_time - t.timestamp) / h))
        for t in j.touches
    )
    return Journey(touches=touches, conversion_time=j.conversion_time)


def last_touch_journey(j: Journey) -> Journey:
    last_idx = max(range(len(j.touches)), key=lambda i: j.touches[i].timestamp)
    touches = tuple(
        Touch(campaign=t.campaign, timestamp=t.timestamp, weight=1.0 if i == last_idx else 0.0)
        for i, t in enumerate(j.touches)
    )
    return Journey(touches=touches, conversion_time=j.conversion_time)
