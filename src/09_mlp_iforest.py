"""Two contrasting approaches to the imbalance problem: Part A trains a
supervised MLP (which has no class_weight option, so imbalance is instead
handled by tuning the decision threshold) and Part B trains an unsupervised
Isolation Forest that never sees failure labels at all, learning only what
"normal" compressor behavior looks like and flagging deviations from it.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "data/processed/metropt3_features.csv"
SPLIT_DATE = "2020-06-01"

df = pd.read_csv(FEATURES_PATH, index_col="timestamp", parse_dates=True)

# Chronological split, as in every other script: evaluating on later months
# only, having trained on earlier ones, simulates predicting real future
# failures rather than interpolating within the training period.
train = df.loc[df.index < SPLIT_DATE]
test = df.loc[df.index >= SPLIT_DATE]

feature_cols = [c for c in df.columns if c != "failure"]

X_train, y_train = train[feature_cols], train["failure"]
X_test, y_test = test[feature_cols], test["failure"]

# Scaler fit on train only, to avoid leaking test-set statistics; this is
# especially important here since both MLP (gradient-based) and Isolation
# Forest (distance/split-based) are sensitive to feature scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def report(name, y_test, y_pred, y_proba):
    print(f"=== {name}: CLASSIFICATION REPORT (TEST SET) ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"=== {name}: CONFUSION MATRIX (TEST SET) ===")
    print(f"TN (correctly predicted normal):   {tn}")
    print(f"FP (false alarms):                 {fp}")
    print(f"FN (missed failures):              {fn}")
    print(f"TP (correctly detected failures):  {tp}")

    auroc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    print()
    print(f"=== {name}: AUROC ===")
    print(f"{auroc:.4f}" if auroc is not None else "N/A (no probability scores available)")

    actual_failures = tp + fn
    print()
    print(f"=== {name}: SUMMARY ===")
    print(
        f"Out of {actual_failures} actual failure windows in the test set, the model caught "
        f"{tp} and missed {fn}. It raised {fp} false alarms."
    )

    report_dict = classification_report(y_test, y_pred, target_names=["Normal", "Failure"], output_dict=True)
    return {
        "recall": report_dict["Failure"]["recall"],
        "precision": report_dict["Failure"]["precision"],
        "f1": report_dict["Failure"]["f1-score"],
        "auroc": auroc,
    }


# ---------------------------------------------------------------------------
# Part A — MLP (Multilayer Perceptron)
# ---------------------------------------------------------------------------
print("######################################################")
print("# PART A — MLP (Multilayer Perceptron)")
print("######################################################")
print()
print(
    "NOTE: MLPClassifier does not support class_weight. To handle the ~2% failure rate, "
    "we take predict_proba on the TRAINING set, sweep thresholds, and pick the one that "
    "maximizes F1 for the failure class. That threshold is then applied to test set "
    "probabilities instead of the default 0.5."
)
print()

# hidden_layer_sizes=(64, 32): two hidden layers give enough capacity to
# learn nonlinear sensor interactions without being so large that it trivially
# memorizes ~16k training rows. early_stopping=True holds out
# validation_fraction=0.15 of the training set and stops once validation loss
# stalls — this is itself a form of regularization against overfitting, used
# here because MLPClassifier has no class_weight or max_depth-style knob.
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
)
mlp_model.fit(X_train_scaled, y_train)

train_proba = mlp_model.predict_proba(X_train_scaled)[:, 1]
thresholds = np.linspace(0.01, 0.99, 99)
f1_scores = [f1_score(y_train, (train_proba >= t).astype(int)) for t in thresholds]
best_threshold = thresholds[int(np.argmax(f1_scores))]
best_train_f1 = max(f1_scores)

print(f"Best threshold (max F1 on training set): {best_threshold:.2f} (train F1 = {best_train_f1:.3f})")
print()

test_proba = mlp_model.predict_proba(X_test_scaled)[:, 1]
y_pred_mlp = (test_proba >= best_threshold).astype(int)

mlp_results = report("MLP", y_test, y_pred_mlp, test_proba)

# ---------------------------------------------------------------------------
# Part B — Isolation Forest (unsupervised anomaly detection)
# ---------------------------------------------------------------------------
print()
print("######################################################")
print("# PART B — Isolation Forest (unsupervised anomaly detection)")
print("######################################################")
print()

# Training ONLY on normal rows is the entire point of this method: Isolation
# Forest learns a model of what "normal" looks like (points that need few
# random splits to isolate are typical/dense; points needing many splits are
# anomalous), without ever being told which rows were failures.
# contamination=0.02 tells it roughly what fraction of future data to expect
# as anomalous, matching our known ~2% failure rate, and sets its default
# decision threshold accordingly.
X_train_normal = X_train_scaled[y_train.values == 0]

iso_model = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
iso_model.fit(X_train_normal)

raw_pred = iso_model.predict(X_test_scaled)  # -1 = anomaly, 1 = normal
y_pred_iso = (raw_pred == -1).astype(int)

# Use the negated anomaly score as a continuous score for AUROC: lower score_samples -> more anomalous
anomaly_score = -iso_model.score_samples(X_test_scaled)

iso_results = report("Isolation Forest", y_test, y_pred_iso, anomaly_score)

print()
print(
    "NOTE: Isolation Forest is an unsupervised method — it was trained without any failure "
    "labels, only learning what healthy compressor behavior looks like. This makes it "
    "applicable in scenarios where labeled failures don't exist, addressing Gap G1 from the "
    "CIA-1 literature review."
)

# ---------------------------------------------------------------------------
# Part C — Comparison table
# ---------------------------------------------------------------------------
print()
print("######################################################")
print("# PART C — MODEL COMPARISON (failure class, test set)")
print("######################################################")
print()
header = f"{'Model':40s}{'Recall':>10s}{'F1':>10s}{'AUROC':>10s}"
print(header)
print("-" * len(header))
print(f"{'Logistic Regression':40s}{0.90:>10.2f}{0.76:>10.2f}{0.91:>10.2f}")
print(f"{'Decision Tree (best depth=3)':40s}{'-':>10s}{0.32:>10.2f}{'-':>10s}")
print(f"{'Random Forest (default 0.5)':40s}{0.14:>10.2f}{0.25:>10.2f}{0.87:>10.2f}")
print(f"{'Random Forest (tuned threshold=0.17)':40s}{0.34:>10.2f}{0.47:>10.2f}{'-':>10s}")
print(f"{'SVM (RBF)':40s}{0.38:>10.2f}{0.47:>10.2f}{0.92:>10.2f}")
print(f"{'K-NN (k=5)':40s}{0.01:>10.2f}{0.03:>10.2f}{0.53:>10.2f}")
print(
    f"{'MLP (tuned threshold=' + f'{best_threshold:.2f})':40s}"
    f"{mlp_results['recall']:>10.2f}{mlp_results['f1']:>10.2f}{mlp_results['auroc']:>10.2f}"
)
print(
    f"{'Isolation Forest (unsupervised)':40s}"
    f"{iso_results['recall']:>10.2f}{iso_results['f1']:>10.2f}{iso_results['auroc']:>10.2f}"
)
