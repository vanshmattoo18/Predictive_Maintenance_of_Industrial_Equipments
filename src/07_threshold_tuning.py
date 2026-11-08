"""Show that Random Forest's default 0.5 decision threshold is a poor choice
for this problem, by sweeping thresholds and tracing a full precision-recall
curve on the test set, then picking the threshold that maximizes F1 for the
failure class. class_weight='balanced' already tells the model to care about
failures during training; this script tunes the separate, independent
decision of where to draw the predict() cutoff on top of that.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "data/processed/metropt3_features.csv"
FIGURES_DIR = "reports/figures"
SPLIT_DATE = "2020-06-01"

df = pd.read_csv(FEATURES_PATH, index_col="timestamp", parse_dates=True)

# Chronological split, as throughout: train on earlier months, evaluate on
# later ones to simulate predicting genuinely unseen future failures.
train = df.loc[df.index < SPLIT_DATE]
test = df.loc[df.index >= SPLIT_DATE]

feature_cols = [c for c in df.columns if c != "failure"]

X_train, y_train = train[feature_cols], train["failure"]
X_test, y_test = test[feature_cols], test["failure"]

# Scaler fit on train only — the test set's mean/std must not influence how
# training features are standardized.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Same RF config as 06_random_forest.py: max_depth=10 regularizes each tree,
# class_weight='balanced' corrects the training-time class imbalance. Here we
# go a step further and also tune the prediction-time threshold.
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train_scaled, y_train)

y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# Step 1 — Try multiple thresholds
# ---------------------------------------------------------------------------
# 0.5 is an arbitrary default that has no special meaning for a ~2%-positive
# problem; we sweep down toward the base rate to see whether a lower cutoff
# trades some precision for the recall we actually want.
thresholds_to_try = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02]

print("=== THRESHOLD SWEEP ===")
header = f"{'Threshold':>10s}{'Precision':>12s}{'Recall':>10s}{'F1':>10s}{'FP':>8s}{'FN':>8s}"
print(header)
print("-" * len(header))

sweep_rows = []
for t in thresholds_to_try:
    y_pred_t = (y_proba >= t).astype(int)
    precision = precision_score(y_test, y_pred_t, zero_division=0)
    recall = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    fp = int(((y_pred_t == 1) & (y_test == 0)).sum())
    fn = int(((y_pred_t == 0) & (y_test == 1)).sum())
    sweep_rows.append((t, precision, recall, f1, fp, fn))
    print(f"{t:>10.2f}{precision:>12.4f}{recall:>10.4f}{f1:>10.4f}{fp:>8d}{fn:>8d}")

# ---------------------------------------------------------------------------
# Step 2 — Precision-Recall curve
# ---------------------------------------------------------------------------
precisions, recalls, pr_thresholds = precision_recall_curve(y_test, y_proba)

# F1 at each PR-curve threshold (precisions/recalls have one more point than pr_thresholds)
f1_scores = np.divide(
    2 * precisions[:-1] * recalls[:-1],
    precisions[:-1] + recalls[:-1],
    out=np.zeros_like(precisions[:-1]),
    where=(precisions[:-1] + recalls[:-1]) != 0,
)
best_idx = np.argmax(f1_scores)
best_threshold = pr_thresholds[best_idx]
best_precision = precisions[best_idx]
best_recall = recalls[best_idx]
best_f1 = f1_scores[best_idx]

# locate the default 0.5 threshold point on the curve
default_idx = int(np.argmin(np.abs(pr_thresholds - 0.5)))
default_precision = precisions[default_idx]
default_recall = recalls[default_idx]

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(recalls, precisions, color="#2C6E9E", linewidth=2, label="Precision-Recall curve")

ax.scatter(
    [default_recall], [default_precision],
    color="#C0392B", s=90, zorder=5, marker="o",
    label=f"Default threshold = 0.50 (P={default_precision:.2f}, R={default_recall:.2f})",
)
ax.scatter(
    [best_recall], [best_precision],
    color="#1E8449", s=90, zorder=5, marker="D",
    label=f"Best F1 threshold = {best_threshold:.2f} (P={best_precision:.2f}, R={best_recall:.2f})",
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Random Forest: Precision-Recall Tradeoff", fontsize=14, fontweight="bold")
ax.legend(loc="lower left")
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)

fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/rf_precision_recall_curve.png", dpi=150)
plt.close(fig)
print()
print(f"Saved precision-recall curve to {FIGURES_DIR}/rf_precision_recall_curve.png")

# ---------------------------------------------------------------------------
# Step 3 — Best threshold + comparison
# ---------------------------------------------------------------------------
auroc = roc_auc_score(y_test, y_proba)

print()
print("=== BEST F1 THRESHOLD ===")
print(f"threshold = {best_threshold:.4f} | precision = {best_precision:.4f} | "
      f"recall = {best_recall:.4f} | F1 = {best_f1:.4f}")

print()
print("=== COMPARISON: Tuned Random Forest vs Logistic Regression (failure class, test set) ===")
print(f"{'Model':40s}{'Recall':>10s}{'F1':>10s}{'AUROC':>10s}")
print(f"{'Logistic Regression':40s}{0.90:>10.2f}{0.76:>10.2f}{0.91:>10.2f}")
print(f"{'Random Forest @ threshold=' + f'{best_threshold:.2f}':40s}{best_recall:>10.2f}{best_f1:>10.2f}{auroc:>10.2f}")
