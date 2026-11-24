"""Read-only check of the Criteo campaign selection and the day 7/8 boundary (primary K=20, secondary K=10).

Recomputed from the raw file with the repository's own code; writes results/diagnostics/criteo_selection_audit.json.
"""
import sys, json, pickle
sys.path.insert(0, ".")
import numpy as np, polars as pl
from scipy.stats import spearmanr
from anytime_attribution.criteo import (ROW_ORD, load_raw, converting_touches, top_k_campaigns, top_k_by_exposure, build_journeys,
                                         build_secondary_journeys, load_config, DAY_SECONDS)
from anytime_attribution.experiments._primary_journeys import build_and_cache_primary_journeys
from anytime_attribution.experiments.e5_certified_fraction import _model_masses

cfg = load_config("config/criteo.yaml")
top20, top10 = cfg["campaigns"]["top_20"], cfg["secondary"]["top_10_by_exposure"]
lf = load_raw(load_config("config/criteo.yaml")["dataset"]["path"])
raw = lf.select(["timestamp", "uid", "campaign", "conversion", "conversion_timestamp", "conversion_id", ROW_ORD]).collect()
out = {}

# 1. day boundary, using the repo's own expression on synthetic boundary rows
b = pl.LazyFrame({"conversion": [1, 1, 1, 1], "conversion_timestamp": [604799, 604800, 2591999, 2592000], "timestamp": [0] * 4,
                  "uid": [1] * 4, "campaign": [1] * 4, "conversion_id": [1, 2, 3, 4]})
out["boundary_conv_day_of_ts_604799,604800,2591999,2592000"] = converting_touches(b).collect()["conv_day"].to_list()
ct = converting_touches(raw.lazy()).collect()
d7 = ct.filter(pl.col("conv_day") == 7)["conversion_timestamp"]; d8 = ct.filter(pl.col("conv_day") == 8)["conversion_timestamp"]
out["raw_day7_conversion_ts_min_max"] = [int(d7.min()), int(d7.max())]; out["raw_day8_conversion_ts_min_max"] = [int(d8.min()), int(d8.max())]
warm = ct.filter(pl.col("conv_day").is_between(1, 7)); mon = ct.filter(pl.col("conv_day").is_between(8, 30))
out["warmup_max_touch_ts"] = int(warm["timestamp"].max()); out["warmup_conv_ts_max"] = int(warm["conversion_timestamp"].max())
out["monitored_conv_ts_min"] = int(mon["conversion_timestamp"].min())
out["warmup_events"] = warm.select("uid", "conversion_id").unique().height; out["monitored_events"] = mon.select("uid", "conversion_id").unique().height

# 2. primary selection: recompute from warm-up conversions only and compare with config
re20 = top_k_campaigns(warm.lazy(), 20)
out["primary_top20_recomputed_equals_config_exact_order"] = re20 == top20
out["primary_top20_recomputed_equals_config_as_set"] = set(re20) == set(top20)
re20_all = top_k_campaigns(mon.lazy(), 20)   # what monitored-window selection would have picked (sensitivity only)
out["primary_overlap_warmup_vs_monitored_top20"] = len(set(top20) & set(re20_all))
cnt = lambda df: df.with_columns(pl.struct("uid", "conversion_id").alias("j")).group_by("campaign").agg(pl.col("j").n_unique().alias("n"))
cw, cm = cnt(warm).filter(pl.col("campaign").is_in(top20)), cnt(mon).filter(pl.col("campaign").is_in(top20))
jj = cw.join(cm, on="campaign", suffix="_mon")
rho = spearmanr(jj["n"].to_numpy(), jj["n_mon"].to_numpy())
out["primary_spearman_warm_vs_monitored_counts_within_top20"] = [float(rho.statistic), float(rho.pvalue)]
mon_ev = mon.with_columns(pl.struct("uid", "conversion_id").alias("j")).group_by("j").agg(pl.col("campaign").n_unique().alias("nc"), pl.col("campaign").first().alias("c"))
out["monitored_events_total"] = mon_ev.height
out["monitored_events_on_top20"] = int(mon_ev["c"].is_in(top20).sum())
out["monitored_events_multi_campaign"] = int((mon_ev["nc"] > 1).sum())

# 3. T_i == sum_j T_ij on the processed primary stream
journeys, t20 = build_and_cache_primary_journeys()
Tj, B = _model_masses(journeys, "linear", 20)
T = Tj.sum(axis=1)
out["primary_stream_n"] = len(journeys); out["primary_stream_B"] = B
out["primary_stream_max_T"] = float(T.max()); out["primary_stream_min_T"] = float(T.min())
out["primary_stream_rows_with_nonzero_masses_on_more_than_one_campaign"] = int(((Tj > 0).sum(axis=1) > 1).sum())
out["primary_stream_equals_monitored_events_on_top20_after_cap"] = [len(journeys), out["monitored_events_on_top20"]]
conv_times = np.array([j.conversion_time for j in journeys])
out["primary_stream_chronological"] = bool(np.all(np.diff(conv_times) >= 0))
out["primary_stream_conv_time_days_min_max"] = [float(conv_times.min()), float(conv_times.max())]
# raw check: every conversion=1 touch of a processed journey is on a top-20 campaign
proc = mon_ev.filter(pl.col("c").is_in(top20))["j"]
mon_all = mon.with_columns(pl.struct("uid", "conversion_id").alias("j"))
sub = mon_all.filter(pl.col("j").is_in(proc.implode()) )
out["raw_touch_rows_of_processed_journeys_off_top20"] = int((~sub["campaign"].is_in(top20)).sum())

# 4. user overlap and shared-touch checks between selection and monitoring samples
wu = set(warm["uid"].unique().to_list()); mu = mon["uid"].unique()
out["monitored_users"] = mu.len(); out["monitored_users_also_in_warmup_conversions"] = int(mu.is_in(list(wu)).sum())
key = lambda df: df.select("uid", "campaign", "timestamp").unique()
sh = key(warm).join(key(mon), on=["uid", "campaign", "timestamp"], how="inner")
out["touch_rows_shared_between_warmup_and_monitored_conversions"] = sh.height

# 5. secondary: which definition of 'warm-up exposure' reproduces config top-10?
imp = raw.select("campaign", "timestamp", "conversion", "conversion_timestamp")
by_imp_day = imp.filter((pl.col("timestamp") // DAY_SECONDS + 1) <= 7)
out["secondary_top10_by_impression_timestamp_day_le7_equals_config"] = [top_k_by_exposure(by_imp_day.lazy(), 10) == top10, set(top_k_by_exposure(by_imp_day.lazy(), 10)) == set(top10)]
conv_part = imp.filter(pl.col("conversion") == 0)
out["secondary_top10_all_rows_equals_config_as_set(any_day)"] = set(top_k_by_exposure(imp.lazy(), 10)) == set(top10)
mon_imp = imp.filter((pl.col("timestamp") // DAY_SECONDS + 1) > 7)
out["secondary_top10_by_impression_days_gt7_overlap_with_config"] = len(set(top_k_by_exposure(mon_imp.lazy(), 10)) & set(top10))
cfg_k = cfg["secondary"]["min_touches"]
out["secondary_config_min_touches"] = cfg_k
sec = build_secondary_journeys(lf, {c: i for i, c in enumerate(top10)}, 8, 30, cfg["cap"]["lookback_days"], cfg["cap"]["max_touches"], cfg_k)
out["secondary_n_journeys_min_touches_cfg(>cfg)"] = len(sec)
out["secondary_n_journeys_min_touches_cfg-1(>=cfg)"] = len(build_secondary_journeys(lf, {c: i for i, c in enumerate(top10)}, 8, 30, cfg["cap"]["lookback_days"], cfg["cap"]["max_touches"], cfg_k - 1))
out["secondary_min_touch_count_observed"] = min(j.n_touches for j in sec)
out["secondary_max_touch_count_observed"] = max(j.n_touches for j in sec)
Ts = np.array([j.n_touches for j in sec]); out["secondary_T_le_B"] = bool(Ts.max() <= 10)
out["secondary_all_touches_on_top10"] = all(0 <= t.campaign < 10 for j in sec for t in j.touches)
out["secondary_earliest_touch_day_of_monitored_journeys"] = float(min(t.timestamp for j in sec for t in j.touches))
import os
os.makedirs("results/diagnostics", exist_ok=True)
json.dump(out, open("results/diagnostics/criteo_selection_audit.json", "w"), indent=1, default=str)
for k, v in out.items(): print(k, v)
