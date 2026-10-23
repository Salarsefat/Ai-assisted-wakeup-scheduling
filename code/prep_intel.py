"""Pre-process the Intel Berkeley Research Lab trace into a 1-minute temperature matrix.

Usage: python3 -I prep_intel.py <path/to/data.txt> <out_dir> [earliest_start] [tag]

Steps (all reported in the paper):
  1. keep motes 1..54, temperature in [5, 45] C and voltage >= 2.3 V
     (dying motes report nonsensical temperatures such as 122 C);
  2. average readings per mote over 1-minute bins;
  3. pick the contiguous window of W days (step 1 day) and the motes that
     maximise coverage; keep motes with >= MIN_COV of minutes observed;
  4. fill gaps by linear interpolation in time and save the observation mask
     (metrics are computed on observed minutes only);
  5. save the matrix (T x N), the mote ids and a JSON summary.
"""
import json
import sys

import numpy as np
import pandas as pd

W_DAYS = 10
MIN_COV = 0.80

src, out = sys.argv[1], sys.argv[2]
EARLIEST = sys.argv[3] if len(sys.argv) > 3 else None   # e.g. 2004-03-10 for a held-out window
TAG = sys.argv[4] if len(sys.argv) > 4 else ""
cols = ["date", "time", "epoch", "mote", "temp", "hum", "light", "volt"]
df = pd.read_csv(src, sep=r"\s+", header=None, names=cols, engine="c",
                 on_bad_lines="skip", dtype=str)
n_raw = len(df)
for c in ["epoch", "mote", "temp", "hum", "light", "volt"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["date", "time", "mote", "temp", "volt"])
df["ts"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce",
                          format="mixed")
df = df.dropna(subset=["ts"])
df = df[(df.mote >= 1) & (df.mote <= 54)]
n_valid_fmt = len(df)
df = df[(df.temp >= 5) & (df.temp <= 45) & (df.volt >= 2.3)]
n_clean = len(df)
df["minute"] = df.ts.dt.floor("min")
per_min = df.groupby(["minute", "mote"]).temp.mean().unstack("mote")
full_idx = pd.date_range(per_min.index.min(), per_min.index.max(), freq="min")
per_min = per_min.reindex(full_idx)

# choose the W-day window (daily step) maximising number of motes with coverage >= MIN_COV
best = None
start = per_min.index.min().normalize() + pd.Timedelta(days=1)
if EARLIEST:
    start = max(start, pd.Timestamp(EARLIEST))
end_limit = per_min.index.max()
while start + pd.Timedelta(days=W_DAYS) <= end_limit:
    win = per_min.loc[start:start + pd.Timedelta(days=W_DAYS) - pd.Timedelta(minutes=1)]
    cov = win.notna().mean()
    good = cov[cov >= MIN_COV]
    score = (len(good), float(good.mean()) if len(good) else 0.0)
    if best is None or score > best[0]:
        best = (score, start, list(good.index))
    start += pd.Timedelta(days=1)

(score, start, motes) = best
win = per_min.loc[start:start + pd.Timedelta(days=W_DAYS) - pd.Timedelta(minutes=1), motes]
observed_frac = float(win.notna().values.mean())
filled = win.interpolate(method="linear", limit_direction="both", axis=0)
mat = filled.values.astype(np.float64)
np.save(f"{out}/intel_temp_1min{TAG}.npy", mat)
np.save(f"{out}/intel_mask_1min{TAG}.npy", win.notna().values)   # True = observed minute
summary = {
    "rows_in_file": int(n_raw),
    "observed_minutes_frac": float(win.notna().values.mean()),
    "rows_parsed_mote_1_54": int(n_valid_fmt),
    "rows_after_range_and_voltage_filter": int(n_clean),
    "window_start": str(start),
    "window_days": W_DAYS,
    "slots": int(mat.shape[0]),
    "motes_kept": [int(m) for m in motes],
    "n_motes": len(motes),
    "min_coverage_threshold": MIN_COV,
    "fraction_minutes_observed_before_interp": observed_frac,
    "temp_min": float(np.nanmin(mat)), "temp_max": float(np.nanmax(mat)),
    "mean_abs_1min_increment": float(np.mean(np.abs(np.diff(mat, axis=0)))),
}
with open(f"{out}/intel_summary{TAG}.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
