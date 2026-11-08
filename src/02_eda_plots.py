"""Exploratory data analysis plots for the labeled MetroPT-3 dataset: a full
7-month sensor timeline, a zoomed view of one failure event, and boxplots
comparing sensor distributions between normal and failure windows. Purely
visual — no modeling here, just building intuition for which sensors look
informative before feature engineering and model training.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

PROCESSED_PATH = "data/processed/metropt3_labeled.csv"
FIGURES_DIR = "reports/figures"

FAILURE_WINDOWS = [
    ("2020-04-18 00:00", "2020-04-18 23:59"),
    ("2020-05-29 23:30", "2020-05-30 06:00"),
    ("2020-06-05 10:00", "2020-06-07 14:30"),
    ("2020-07-15 14:30", "2020-07-15 19:00"),
]

NORMAL_COLOR = "#2C6E9E"
FAILURE_COLOR = "#C0392B"
SHADE_COLOR = "#F5B7B1"

df = pd.read_csv(PROCESSED_PATH, index_col="timestamp", parse_dates=True)


def shade_failures(ax):
    for start, end in FAILURE_WINDOWS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color=SHADE_COLOR, alpha=0.6, zorder=0)


# ---------------------------------------------------------------------------
# Plot 1 — Full timeline overview (daily means)
# ---------------------------------------------------------------------------
# Daily means (rather than raw 1s data) are used here purely for
# visualization scale — plotting ~7 months of 1-second readings would be an
# unreadable wall of noise, so we smooth to see the long-run trend and how it
# relates to the shaded failure windows.
daily = df[["Motor_current", "Oil_temperature", "TP2"]].resample("D").mean()
daily_failure = df["failure"].resample("D").max()

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

axes[0].plot(daily.index, daily["Motor_current"], color=NORMAL_COLOR, linewidth=1.5)
axes[0].set_ylabel("Motor current (A)")
axes[0].set_title("Motor current — daily mean")

axes[1].plot(daily.index, daily["Oil_temperature"], color=NORMAL_COLOR, linewidth=1.5)
axes[1].set_ylabel("Oil temp (°C)")
axes[1].set_title("Oil temperature — daily mean")

axes[2].plot(daily.index, daily["TP2"], color=NORMAL_COLOR, linewidth=1.5)
axes[2].set_ylabel("TP2 (bar)")
axes[2].set_title("TP2 (compressor pressure) — daily mean")

axes[3].plot(daily_failure.index, daily_failure.values, color=FAILURE_COLOR, linewidth=1.5, drawstyle="steps-post")
axes[3].set_ylabel("Failure label")
axes[3].set_title("Failure label (daily max)")
axes[3].set_ylim(-0.1, 1.1)
axes[3].set_xlabel("Date")

for ax in axes:
    shade_failures(ax)

fig.suptitle("MetroPT-3: 7 months of compressor operation", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIGURES_DIR}/timeline_overview.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Plot 2 — Zoomed into Failure 1 (raw data)
# ---------------------------------------------------------------------------
# Unlike Plot 1, this uses the raw unaveraged signal over a narrow window so
# we can see exactly how sensors behave in the minutes/hours around a single
# real failure event, rather than a smoothed-out trend.
zoom = df.loc["2020-04-15":"2020-04-21"]

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

axes[0].plot(zoom.index, zoom["Motor_current"], color=NORMAL_COLOR, linewidth=0.8)
axes[0].set_ylabel("Motor current (A)")
axes[0].set_title("Motor current — raw")

axes[1].plot(zoom.index, zoom["Oil_temperature"], color=NORMAL_COLOR, linewidth=0.8)
axes[1].set_ylabel("Oil temp (°C)")
axes[1].set_title("Oil temperature — raw")

axes[2].plot(zoom.index, zoom["TP2"], color=NORMAL_COLOR, linewidth=0.8)
axes[2].set_ylabel("TP2 (bar)")
axes[2].set_title("TP2 (compressor pressure) — raw")

axes[3].plot(zoom.index, zoom["failure"], color=FAILURE_COLOR, linewidth=1.2, drawstyle="steps-post")
axes[3].set_ylabel("Failure label")
axes[3].set_title("Failure label")
axes[3].set_ylim(-0.1, 1.1)
axes[3].set_xlabel("Date")

for ax in axes:
    shade_failures(ax)

fig.suptitle("Failure 1 (18 April 2020) — zoomed view", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(f"{FIGURES_DIR}/failure1_zoom.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# Plot 3 — Class comparison boxplots
# ---------------------------------------------------------------------------
# Directly compares each analog sensor's distribution between normal and
# failure rows. A sensor with clearly separated boxes is a strong candidate
# predictive feature; heavy overlap suggests that sensor alone won't help a
# classifier much.
analog_cols = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]

plot_df = df[analog_cols + ["failure"]].copy()
plot_df["failure"] = plot_df["failure"].map({0: "Normal", 1: "Failure"})
melted = plot_df.melt(id_vars="failure", var_name="sensor", value_name="value")

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, col in enumerate(analog_cols):
    ax = axes[i]
    sub = melted[melted["sensor"] == col]
    sns.boxplot(
        data=sub,
        x="failure",
        y="value",
        hue="failure",
        order=["Normal", "Failure"],
        hue_order=["Normal", "Failure"],
        palette={"Normal": NORMAL_COLOR, "Failure": FAILURE_COLOR},
        legend=False,
        showfliers=False,
        ax=ax,
    )
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("")

axes[-1].axis("off")

fig.suptitle("Sensor distributions: Normal vs Failure", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(f"{FIGURES_DIR}/class_comparison_boxplots.png", dpi=150)
plt.show()

print("Saved:")
print(f"  {FIGURES_DIR}/timeline_overview.png")
print(f"  {FIGURES_DIR}/failure1_zoom.png")
print(f"  {FIGURES_DIR}/class_comparison_boxplots.png")
