"""Train the DARNN model on the secondary Criteo construction (five seeds) and materialize its credits for days 8-30.

DARNN appears only on the secondary construction (literature-convention linkage, K = 10), because every journey
of the primary construction touches a single campaign. The model is small (GRU, hidden size 16), trains in
seconds per seed on a CPU, and a GPU is optional.

Output (read by experiments/e8_instability.py):
    results/credits_secondary/darnn/<seed>.parquet for seed in 0..4
with schema journey_idx, conversion_time_days, n_touches, credit_0 .. credit_9, in the journey order of
scripts/materialize_criteo_credits.py (criteo.ORDER_RULE).

Usage (from the repository root, after `python scripts/download_criteo.py`):
    python scripts/train_darnn_gpu.py                 # --device auto: CUDA, else Apple MPS, else CPU
    python scripts/train_darnn_gpu.py --device cuda

Results are not guaranteed bit-identical across devices and torch builds; E8 reports seed-level statistics.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, ".")

import polars as pl

from anytime_attribution.criteo import (
    build_secondary_journeys,
    build_secondary_negative_examples,
    load_config,
    load_raw,
)
from anytime_attribution.models.darnn import DARNN


def resolve_device(name: str) -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="compute device; 'auto' picks CUDA, then Apple MPS, then CPU")
parser.add_argument("--epochs", type=int, default=5, help="training epochs (default: the model's own default)")
parser.add_argument("--hidden-size", type=int, default=16)
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
parser.add_argument("--data", default=None, help="raw Criteo file (default: dataset.path from the config)")
parser.add_argument("--config", default="config/criteo.yaml")
parser.add_argument("--out-dir", default="results/credits_secondary/darnn")
args = parser.parse_args()

cfg = load_config(args.config)
data_path = args.data or cfg["dataset"]["path"]
if not os.path.exists(data_path):
    sys.exit(f"Raw Criteo file not found: {data_path}. Run `python scripts/download_criteo.py` first.")
device = resolve_device(args.device)
sec = cfg["secondary"]
campaign_index = {c: i for i, c in enumerate(sec["top_10_by_exposure"])}
n_campaigns = len(campaign_index)
lookback = cfg["cap"]["lookback_days"]
max_touches = cfg["cap"]["max_touches"]
w_lo, w_hi = cfg["days"]["warmup"]
m_lo, m_hi = cfg["days"]["monitored"]
print(f"device: {device}", flush=True)

t0 = time.time()
lf = load_raw(data_path)
warm = build_secondary_journeys(lf, campaign_index, w_lo, w_hi, lookback, max_touches, sec["min_touches"])
neg = build_secondary_negative_examples(lf, campaign_index, w_lo, w_hi, lookback, max_touches, sec["min_touches"],
                                        n_sample=sec["negative_n_sample"], seed=cfg["seed"])
mon = build_secondary_journeys(lf, campaign_index, m_lo, m_hi, lookback, max_touches, sec["min_touches"])
print(f"{len(warm)} warm-up, {len(neg)} negative, {len(mon)} monitored journeys ({time.time()-t0:.0f}s)", flush=True)

os.makedirs(args.out_dir, exist_ok=True)
for seed in args.seeds:
    ts = time.time()
    model = DARNN(hidden_size=args.hidden_size, epochs=args.epochs, device=device)
    model.fit(warm, n_campaigns, seed=seed, negative_journeys=neg)
    rows = []
    for idx, j in enumerate(mon):
        z = model.credit(j)
        row = {"journey_idx": idx, "conversion_time_days": j.conversion_time, "n_touches": j.n_touches}
        for c in range(n_campaigns):
            row[f"credit_{c}"] = float(z[c])
        rows.append(row)
    path = f"{args.out_dir}/{seed}.parquet"
    pl.DataFrame(rows).write_parquet(path)
    print(f"seed {seed}: wrote {path} ({len(rows)} rows, {time.time()-ts:.1f}s)", flush=True)
print(f"done in {time.time()-t0:.0f}s")
