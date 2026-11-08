import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

"""Demonstrates regularization strength and the bias-variance tradeoff using
Logistic Regression's C parameter (C = 1/lambda). Sweeps C across several
orders of magnitude and plots train vs test F1 to show the classic
underfitting-to-overfitting curve, then reports which C actually generalizes
best on held-out data.
"""

sns.set_theme(style="whitegrid")

FEATURES_PATH = "data/processed/metropt3_features.csv"
FIGURES_DIR = "reports/figures"
SPLIT_DATE = "2020-06-01"

TRAIN_COLOR = "#2C6E9E"
TEST_COLOR = "#C0392B"

# Spans several orders of magnitude on both sides of the sklearn default
# (C=1.0) so the sweep covers clearly-underfit, near-default, and clearly
# overfit regimes.
C_VALUES = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

df = pd.read_csv(FEATURES_PATH, index_col="timestamp", parse_dates=True)

# Chronological split, consistent with every other script in this project.
train = df.loc[df.index < SPLIT_DATE]
test = df.loc[df.index >= SPLIT_DATE]

feature_cols = [c for c in df.columns if c != "failure"]

X_train, y_train = train[feature_cols], train["failure"]
X_test, y_test = test[feature_cols], test["failure"]

# Scaler fit on train only, to avoid leaking test-set statistics into the
# features the model is regularized and trained on.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Step 1 — Sweep C values
# ---------------------------------------------------------------------------
# In sklearn's LogisticRegression, C is the INVERSE of regularization strength:
# small C = strong regularization (simpler model, may underfit), large C = weak
# regularization (complex model, may overfit). This is equivalent to lambda in
# ridge/lasso regression from the course (C = 1/lambda).

print("######################################################")
print("# STEP 1 — REGULARIZATION (C) SWEEP")
print("######################################################")
print()

rows = []
for C in C_VALUES:
    # class_weight='balanced' is held fixed across the whole sweep so that C
    # is the only thing varying — isolating the regularization effect from
    # the separate imbalance-correction effect.
    model = LogisticRegression(C=C, class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]

    rows.append(
        {
            "C": C,
            "Train F1": f1_score(y_train, y_pred_train, pos_label=1),
            "Test F1": f1_score(y_test, y_pred_test, pos_label=1),
            "Train Recall": recall_score(y_train, y_pred_train, pos_label=1),
            "Test Recall": recall_score(y_test, y_pred_test, pos_label=1),
            "Test AUROC": roc_auc_score(y_test, y_proba_test),
        }
    )

results = pd.DataFrame(rows)

display = results.copy()
for col in ["Train F1", "Test F1", "Train Recall", "Test Recall", "Test AUROC"]:
    display[col] = display[col].map(lambda v: f"{v:.3f}")
print(display.to_string(index=False))

# ---------------------------------------------------------------------------
# Step 2 — Regularization curve plot
# ---------------------------------------------------------------------------
best_idx = results["Test F1"].idxmax()
best_C = results.loc[best_idx, "C"]
best_test_f1 = results.loc[best_idx, "Test F1"]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(results["C"], results["Train F1"], color=TRAIN_COLOR, marker="o", linewidth=2, label="Train F1")
ax.plot(results["C"], results["Test F1"], color=TEST_COLOR, marker="o", linewidth=2, label="Test F1")
ax.set_xscale("log")
ax.set_xlabel("C (log scale)")
ax.set_ylabel("F1 score (failure class)")
ax.set_title("Logistic Regression: Effect of Regularization Strength (C = 1/λ)", fontsize=13, fontweight="bold")

ax.axvline(best_C, color="gray", linestyle="--", linewidth=1.5, label=f"Best test C = {best_C}")

ax.text(
    results["C"].min(),
    1.0,
    "Strong regularization\n(underfitting)",
    ha="left",
    va="top",
    fontsize=9,
    color="dimgray",
)
ax.text(
    results["C"].max(),
    1.0,
    "Weak regularization\n(overfitting)",
    ha="right",
    va="top",
    fontsize=9,
    color="dimgray",
)
ax.set_ylim(top=1.0)

ax.legend(loc="center right")
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/lr_regularization_curve.png", dpi=150)
plt.close(fig)

print()
print(f"Saved: {FIGURES_DIR}/lr_regularization_curve.png")

# ---------------------------------------------------------------------------
# Step 3 — Summary
# ---------------------------------------------------------------------------
print()
print("######################################################")
print("# STEP 3 — SUMMARY")
print("######################################################")
print()
print(f"Best C value: {best_C}")
print(f"Best test F1: {best_test_f1:.3f}")

default_f1 = results.loc[results["C"] == 1.0, "Test F1"].iloc[0]
if best_C == 1.0:
    print("The best C matches our default (C=1.0) — tuning did not find a better setting.")
else:
    print(
        f"Tuning helped: C={best_C} achieves test F1={best_test_f1:.3f}, compared to "
        f"{default_f1:.3f} at the default C=1.0."
    )

print()
print(
    "This demonstrates the bias-variance tradeoff via regularization: too much regularization "
    "(small C) underfits by constraining the model too aggressively; too little (large C) "
    "overfits by allowing the model to fit noise. The optimal C balances model complexity "
    "against generalization."
)
