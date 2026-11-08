"""Baseline model for failure prediction: Logistic Regression with class
balancing, trained on a chronological train/test split of the MetroPT-3
feature table. Establishes the reference numbers (recall, F1, AUROC) that
every later, more complex model in this project is compared against.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "data/processed/metropt3_features.csv"
SPLIT_DATE = "2020-06-01"

df = pd.read_csv(FEATURES_PATH, index_col="timestamp", parse_dates=True)

# A chronological split (train = everything before June, test = everything
# after) instead of a random split, because this is operational time-series
# data: a random split would let the model train on windows that are minutes
# away from a test window (near-duplicate sensor readings), leaking
# information and making the evaluation unrealistically optimistic. A
# chronological split instead mimics the real deployment scenario — predict
# future failures having only seen past data.
train = df.loc[df.index < SPLIT_DATE]
test = df.loc[df.index >= SPLIT_DATE]

print("=== CHRONOLOGICAL SPLIT ===")
print(f"Train: {len(train)} windows, {train['failure'].sum():.0f} failures "
      f"({train['failure'].mean() * 100:.4f}%), range {train.index.min()} to {train.index.max()}")
print(f"Test:  {len(test)} windows, {test['failure'].sum():.0f} failures "
      f"({test['failure'].mean() * 100:.4f}%), range {test.index.min()} to {test.index.max()}")

feature_cols = [c for c in df.columns if c != "failure"]

X_train, y_train = train[feature_cols], train["failure"]
X_test, y_test = test[feature_cols], test["failure"]

# ---------------------------------------------------------------------------
# Step 1 — Scale features
# ---------------------------------------------------------------------------
print()
print("=== SCALING ===")
print(
    "We fit the StandardScaler ONLY on the training data (not on train+test combined) "
    "because fitting on the full dataset would let statistics (mean/std) from the test "
    "set — i.e. the future — leak into the scaling applied to the training set. That "
    "would give the model indirect knowledge of the test distribution before it should "
    "ever see it, producing an overly optimistic evaluation. The scaler's learned mean "
    "and std are then simply applied (transform only, no refit) to the test set."
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Step 2 — Train Logistic Regression
# ---------------------------------------------------------------------------
print()
print("=== MODEL: LOGISTIC REGRESSION ===")
print(
    "class_weight='balanced' automatically reweights the loss function inversely "
    "proportional to class frequency, so errors on the rare 'failure' class (~2% of "
    "windows) are penalized much more heavily than errors on the common 'normal' class. "
    "Without this, a model could achieve ~98% accuracy by always predicting 'normal' and "
    "never learn to detect failures at all. Balancing forces the model to actually try to "
    "separate the two classes instead of defaulting to the majority."
)

model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------------
# Step 3 — Evaluate on test set
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# We report precision/recall/F1/AUROC for the failure class instead of
# accuracy: with ~98% of windows being "normal", a model that predicts
# "normal" for everything scores ~98% accuracy while catching zero failures.
# Recall (did we catch the failures?) and precision (how many alarms were
# real?) are the metrics that actually matter for a maintenance system, and
# AUROC summarizes ranking quality independent of any one threshold.
print()
print("=== CLASSIFICATION REPORT (TEST SET) ===")
print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("=== CONFUSION MATRIX (TEST SET) ===")
print(f"TN (correctly predicted normal):   {tn}")
print(f"FP (false alarms):                 {fp}")
print(f"FN (missed failures):              {fn}")
print(f"TP (correctly detected failures):  {tp}")

auroc = roc_auc_score(y_test, y_proba)
print()
print("=== AUROC ===")
print(f"{auroc:.4f}")

# ---------------------------------------------------------------------------
# Step 4 — Plain-English summary
# ---------------------------------------------------------------------------
actual_failures = tp + fn
print()
print("=== SUMMARY ===")
print(
    f"Out of {actual_failures} actual failure windows in the test set, the model caught "
    f"{tp} and missed {fn}. It raised {fp} false alarms."
)
