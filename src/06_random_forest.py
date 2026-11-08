"""Random Forest baseline: an ensemble of depth-limited, class-balanced
decision trees, evaluated at the default 0.5 decision threshold on the same
chronological train/test split used throughout this project. Also reports
feature importances, since unlike a single shallow tree a forest doesn't give
a human-readable decision path.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "data/processed/metropt3_features.csv"
SPLIT_DATE = "2020-06-01"

df = pd.read_csv(FEATURES_PATH, index_col="timestamp", parse_dates=True)

# Chronological split: train on earlier months, evaluate on later ones so the
# test set genuinely represents unseen future data rather than interpolated
# neighbors of training rows (which a random split would allow).
train = df.loc[df.index < SPLIT_DATE]
test = df.loc[df.index >= SPLIT_DATE]

feature_cols = [c for c in df.columns if c != "failure"]

X_train, y_train = train[feature_cols], train["failure"]
X_test, y_test = test[feature_cols], test["failure"]

# Scaler fit on train only, to avoid leaking test-set statistics into how
# training features are standardized.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Train Random Forest
# ---------------------------------------------------------------------------
# n_estimators=200 trees for a stable averaged prediction; max_depth=10 caps
# each individual tree's complexity (the forest's own regularization knob,
# same role as max_depth in 05_decision_tree.py) so trees don't each memorize
# training noise even though bagging already reduces variance somewhat;
# class_weight='balanced' reweights splits so the rare failure class isn't
# effectively ignored during training.
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# 1. Classification report
# ---------------------------------------------------------------------------
# We look at precision/recall/F1 for the failure class, not accuracy: with
# ~98% of test windows being normal, accuracy stays high even if the model
# never detects a failure, so it hides exactly the behavior we care about.
print("=== CLASSIFICATION REPORT (TEST SET) ===")
report = classification_report(y_test, y_pred, target_names=["Normal", "Failure"])
print(report)

# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("=== CONFUSION MATRIX (TEST SET) ===")
print(f"TN (correctly predicted normal):   {tn}")
print(f"FP (false alarms):                 {fp}")
print(f"FN (missed failures):              {fn}")
print(f"TP (correctly detected failures):  {tp}")

# ---------------------------------------------------------------------------
# 3. AUROC
# ---------------------------------------------------------------------------
auroc = roc_auc_score(y_test, y_proba)
print()
print("=== AUROC ===")
print(f"{auroc:.4f}")

# ---------------------------------------------------------------------------
# 4. Plain-English summary
# ---------------------------------------------------------------------------
actual_failures = tp + fn
print()
print("=== SUMMARY ===")
print(
    f"Out of {actual_failures} actual failure windows in the test set, the model caught "
    f"{tp} and missed {fn}. It raised {fp} false alarms."
)

# ---------------------------------------------------------------------------
# 5. Top 10 feature importances
# ---------------------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print()
print("=== TOP 10 MOST IMPORTANT FEATURES ===")
for rank, (feat, score) in enumerate(importances.head(10).items(), start=1):
    print(f"{rank:2d}. {feat:30s} {score:.4f}")

# ---------------------------------------------------------------------------
# Comparison to previous models
# ---------------------------------------------------------------------------
rf_recall = report_dict = classification_report(y_test, y_pred, target_names=["Normal", "Failure"], output_dict=True)
rf_recall_val = report_dict["Failure"]["recall"]
rf_f1_val = report_dict["Failure"]["f1-score"]

print()
print("=== MODEL COMPARISON (failure class, test set) ===")
print(f"{'Model':35s}{'Recall':>10s}{'F1':>10s}{'AUROC':>10s}")
print(f"{'Logistic Regression':35s}{0.90:>10.2f}{0.76:>10.2f}{0.91:>10.2f}")
print(f"{'Decision Tree (best, depth=3)':35s}{'-':>10s}{0.32:>10.2f}{'-':>10s}")
print(f"{'Random Forest (n=200, depth=10)':35s}{rf_recall_val:>10.2f}{rf_f1_val:>10.2f}{auroc:>10.2f}")
