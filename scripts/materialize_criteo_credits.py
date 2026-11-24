"""Materialize per-journey attribution credits from the raw Criteo file.

Writes the eight parquet files read by the Criteo experiments:

    <out-root>/credits/{linear,time_decay,last_touch,logistic_shap}/0.parquet            (primary, K = 20)
    <out-root>/credits_secondary/{linear,time_decay,last_touch,logistic_shap}/0.parquet (secondary, K = 10)

(`scripts/train_darnn_gpu.py` writes the five DARNN files of the secondary construction.)

Schema: journey_idx, conversion_time_days, n_touches, credit_0 .. credit_{K-1}. Learned models (logistic + SHAP)
are fitted once on the warm-up window (days 1-7) and frozen; credits are computed on the monitored window
(days 8-30). Output is deterministic given the raw file and config/criteo.yaml (seed 0); see ORDER_RULE in
anytime_attribution/criteo.py.

Usage (from the repository root):
    python scripts/materialize_criteo_credits.py
    python scripts/materialize_criteo_credits.py --data data/criteo_attribution_dataset.tsv.gz --out-root results

The raw file is the public Criteo Attribution Modeling for Bidding dataset (see DATA_LICENSE.md).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, ".")

import polars as pl

from anytime_attribution.criteo import (
    build_journeys,
    build_negative_examples,
    build_secondary_journeys,
    build_secondary_negative_examples,
    converting_touches,
    load_config,
    load_raw,
)
from anytime_attribution.models.last_touch import LastTouch
from anytime_attribution.models.linear import Linear
from anytime_attribution.models.logistic_shap import LogisticSHAP
from anytime_attribution.models.time_decay import TimeDecay

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", default=None, help="raw Criteo file (default: dataset.path from the config)")
parser.add_argument("--config", default="config/criteo.yaml")
parser.add_argument("--out-root", default="results", help="directory that receives credits/ and credits_secondary/")
parser.add_argument("--primary-negatives", type=int, default=100_000, help="non-converting users sampled to fit the primary logistic model")
parser.add_argument("--only", choices=["primary", "secondary", "both"], default="both")
args = parser.parse_args()

cfg = load_config(args.config)
data_path = args.data or cfg["dataset"]["path"]
if not os.path.exists(data_path):
    sys.exit(f"Raw Criteo file not found: {data_path}. Run `python scripts/download_criteo.py` first.")
seed = cfg["seed"]
lookback = cfg["cap"]["lookback_days"]
max_touches = cfg["cap"]["max_touches"]
w_lo, w_hi = cfg["days"]["warmup"]
m_lo, m_hi = cfg["days"]["monitored"]


def write_credits(journeys, models, n_campaigns, root):
    """One parquet per fitted model in `models`; journeys are written in the order of `criteo.ORDER_RULE`, identical in
    every file.
    """
    for name, model in models.items():
        rows = []
        for idx, j in enumerate(journeys):
            z = model.credit(j)
            row = {"journey_idx": idx, "conversion_time_days": j.conversion_time, "n_touches": j.n_touches}
            for c in range(n_campaigns):
                row[f"credit_{c}"] = float(z[c])
            rows.append(row)
        path = f"{root}/{name}/0.parquet"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pl.DataFrame(rows).write_parquet(path)
        print(f"  wrote {path} ({len(rows)} rows)", flush=True)


t0 = time.time()
lf = load_raw(data_path)

if args.only in ("primary", "both"):
    campaign_index = {c: i for i, c in enumerate(cfg["campaigns"]["top_20"])}
    K = len(campaign_index)
    touches = converting_touches(lf)
    warm = build_journeys(touches.filter(pl.col("conv_day").is_between(w_lo, w_hi)), campaign_index, lookback, max_touches)
    mon = build_journeys(touches.filter(pl.col("conv_day").is_between(m_lo, m_hi)), campaign_index, lookback, max_touches)
    neg = build_negative_examples(lf, campaign_index, w_lo, w_hi, max_touches, n_sample=args.primary_negatives, seed=seed)
    print(f"primary: {len(warm)} warm-up, {len(mon)} monitored journeys, {len(neg)} negatives ({time.time()-t0:.0f}s)", flush=True)
    models = {"linear": Linear(), "time_decay": TimeDecay(h=7.0), "last_touch": LastTouch(), "logistic_shap": LogisticSHAP()}
    for m in models.values():
        m.fit(warm, K, seed=seed, negative_journeys=neg)
    write_credits(mon, models, K, f"{args.out_root}/credits")

if args.only in ("secondary", "both"):
    sec = cfg["secondary"]
    campaign_index = {c: i for i, c in enumerate(sec["top_10_by_exposure"])}
    K = len(campaign_index)
    warm = build_secondary_journeys(lf, campaign_index, w_lo, w_hi, lookback, max_touches, sec["min_touches"])
    mon = build_secondary_journeys(lf, campaign_index, m_lo, m_hi, lookback, max_touches, sec["min_touches"])
    neg = build_secondary_negative_examples(lf, campaign_index, w_lo, w_hi, lookback, max_touches, sec["min_touches"],
                                            n_sample=sec["negative_n_sample"], seed=seed)
    print(f"secondary: {len(warm)} warm-up, {len(mon)} monitored journeys, {len(neg)} negatives ({time.time()-t0:.0f}s)", flush=True)
    models = {"linear": Linear(), "time_decay": TimeDecay(h=7.0), "last_touch": LastTouch(), "logistic_shap": LogisticSHAP()}
    for m in models.values():
        m.fit(warm, K, seed=seed, negative_journeys=neg)
    write_credits(mon, models, K, f"{args.out_root}/credits_secondary")

print(f"done in {time.time()-t0:.0f}s")
