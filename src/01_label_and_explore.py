"""Load the raw MetroPT-3 air compressor sensor log, attach binary failure
labels using the known maintenance-report failure windows, and explore the
resulting class balance. Produces data/processed/metropt3_labeled.csv, which
every downstream script builds on.
"""

import pandas as pd

RAW_PATH = "data/raw/MetroPT3(AirCompressor).csv"
PROCESSED_PATH = "data/processed/metropt3_labeled.csv"

# These windows come from the maintenance reports accompanying the MetroPT-3
# dataset, not from the sensor data itself — the raw CSV has no failure
# column, so ground truth has to be injected manually from known incident dates.
FAILURE_WINDOWS = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]

df = pd.read_csv(RAW_PATH)

# "Unnamed: 0" is a leftover row-index column from how the CSV was exported
df = df.drop(columns=["Unnamed: 0"])

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# Default every row to normal, then flip rows inside a known failure window
# to 1. This produces the binary target used by every model in this project.
df["failure"] = 0
for start, end in FAILURE_WINDOWS:
    df.loc[start:end, "failure"] = 1

total_rows = len(df)
failure_rows = int(df["failure"].sum())
normal_rows = total_rows - failure_rows

print("=== TOTAL ROWS ===")
print(total_rows)

print()
# This class balance check is the reason every model script downstream uses
# class_weight='balanced' (or an equivalent threshold-tuning workaround):
# failures are a tiny fraction of rows, so an unweighted classifier could get
# ~98% accuracy by always predicting "normal" and never catch a real failure.
print("=== FAILURE ROWS ===")
print(f"{failure_rows} ({failure_rows / total_rows * 100:.4f}%)")

print()
print("=== NORMAL ROWS ===")
print(f"{normal_rows} ({normal_rows / total_rows * 100:.4f}%)")

print()
print("=== DATE RANGE ===")
print(f"{df.index.min()} to {df.index.max()}")

print()
print("=== ROWS PER MONTH ===")
print(df.index.to_period("M").value_counts().sort_index())

df.to_csv(PROCESSED_PATH)
print()
print(f"Saved labeled dataframe to {PROCESSED_PATH}")
