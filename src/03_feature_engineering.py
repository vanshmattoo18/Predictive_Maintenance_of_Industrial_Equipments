"""Aggregate the raw 1-second sensor stream into 5-minute windows of summary
statistics (mean/std/max for analog sensors, mean for digital signals, plus
duty-cycle features). This turns a huge, highly autocorrelated time series into
a much smaller, more model-friendly table of windows, each labeled failure/normal.
"""

import pandas as pd

PROCESSED_PATH = "data/processed/metropt3_labeled.csv"
OUTPUT_PATH = "data/processed/metropt3_features.csv"

# 5-minute windows balance two concerns: too fine (e.g. raw 1s rows) leaves
# almost no signal per row and massively inflates the dataset; too coarse
# would blur out short failure events. 5min keeps failure windows (which last
# hours) well represented while cutting row count by ~300x.
WINDOW = "5min"

ANALOG_COLS = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL_COLS = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]

df = pd.read_csv(PROCESSED_PATH, index_col="timestamp", parse_dates=True)

grouper = df.resample(WINDOW)

agg = {}

# Mean captures the typical operating level in a window; std captures how
# noisy/unstable the sensor was — instability is often a leading indicator of
# failure even when the mean looks normal.
for col in ANALOG_COLS:
    agg[f"{col}_mean"] = grouper[col].mean()
    agg[f"{col}_std"] = grouper[col].std()

agg["Motor_current_max"] = grouper["Motor_current"].max()

for col in DIGITAL_COLS:
    agg[f"{col}_mean"] = grouper[col].mean()

# COMP=0 means air intake present -> compressor running/loaded (per PDF: COMP active/1 = no air intake = off/offloaded)
# duty_cycle and switch_count summarize how hard/often the compressor cycled
# within a window — mechanical stress indicators that a plain mean/std of the
# raw digital signal wouldn't capture.
agg["duty_cycle"] = grouper["COMP"].apply(lambda s: (s == 0).mean())
agg["switch_count"] = grouper["COMP"].apply(lambda s: (s.diff().fillna(0) != 0).sum())

# A window is labeled a failure if ANY second inside it was a failure — we'd
# rather flag a window that's only partially failure-affected than silently
# average the signal away and miss it.
agg["failure"] = grouper["failure"].max()

features = pd.DataFrame(agg)

print("=== SHAPE ===")
print(features.shape)

print()
print("=== COLUMN NAMES ===")
print(list(features.columns))

print()
print("=== CLASS BALANCE ===")
n_failure = int(features["failure"].sum())
n_total = len(features)
print(f"failure=1: {n_failure} ({n_failure / n_total * 100:.4f}%)")
print(f"failure=0: {n_total - n_failure} ({(n_total - n_failure) / n_total * 100:.4f}%)")

print()
print("=== FIRST 3 ROWS (transposed) ===")
print(features.head(3).T.to_string())

print()
print("=== NaN COUNT PER COLUMN (before dropping) ===")
print(features.isnull().sum())

# NaNs here come from *_std columns on windows that happened to contain only
# a single raw sample (std of one value is undefined). These windows are rare
# and dropping them is simpler and safer than imputing a std we don't have
# real evidence for.
rows_before = len(features)
features = features.dropna()
rows_after = len(features)
rows_dropped = rows_before - rows_after

print()
print("=== ROWS DROPPED (any NaN) ===")
print(rows_dropped)

print()
print("=== NEW SHAPE ===")
print(features.shape)

print()
print("=== NEW CLASS BALANCE ===")
n_failure = int(features["failure"].sum())
n_total = len(features)
print(f"failure=1: {n_failure} ({n_failure / n_total * 100:.4f}%)")
print(f"failure=0: {n_total - n_failure} ({(n_total - n_failure) / n_total * 100:.4f}%)")

print()
print("=== NaN COUNT PER COLUMN (after dropping) ===")
print(features.isnull().sum())
print(f"Total NaNs remaining: {int(features.isnull().sum().sum())}")

features.to_csv(OUTPUT_PATH)
print()
print(f"Saved features dataframe to {OUTPUT_PATH}")
