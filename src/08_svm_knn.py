"""Two more classifiers on the same chronological split: an RBF-kernel SVM
(which, unlike Logistic Regression/Random Forest, does support class_weight)
and K-Nearest Neighbors (which does not, and is included partly to show what
happens to a majority-vote classifier on a heavily imbalanced problem).
"""

import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FEATURES_PATH = "data/processed/metropt3_features.csv"
SPLIT_DATE = "2020-06-01"

df = pd.read_csv(FEATURES_PATH, index_col="timestamp", parse_dates=True)

# Chronological split, consistent with every other script: train on earlier
# months, test on later ones, to avoid leaking near-duplicate neighboring
# timestamps across the train/test boundary the way a random split would.
train = df.loc[df.index < SPLIT_DATE]
test = df.loc[df.index >= SPLIT_DATE]

feature_cols = [c for c in df.columns if c != "failure"]

X_train, y_train = train[feature_cols], train["failure"]
X_test, y_test = test[feature_cols], test["failure"]

# Scaler fit on train only. This matters even more for SVM/K-NN than for
# tree-based models: both are distance/margin-based, so unscaled features
# with larger numeric ranges would dominate the distance calculation
# regardless of actual predictive relevance.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def evaluate(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"=== {name}: CLASSIFICATION REPORT (TEST SET) ===")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"=== {name}: CONFUSION MATRIX (TEST SET) ===")
    print(f"TN (correctly predicted normal):   {tn}")
    print(f"FP (false alarms):                 {fp}")
    print(f"FN (missed failures):              {fn}")
    print(f"TP (correctly detected failures):  {tp}")

    auroc = roc_auc_score(y_test, y_proba)
    print()
    print(f"=== {name}: AUROC ===")
    print(f"{auroc:.4f}")

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
# Part A — SVM
# ---------------------------------------------------------------------------
print("######################################################")
print("# PART A — SVM (RBF kernel)")
print("######################################################")
print()

# class_weight='balanced' reweights the margin penalty so misclassifying a
# failure costs much more than misclassifying a normal point, same purpose as
# in Logistic Regression/Random Forest. probability=True additionally fits a
# calibration step so predict_proba is available (needed for AUROC below).
svm_model = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)

start = time.time()
svm_results = evaluate("SVM", svm_model, X_train_scaled, y_train, X_test_scaled, y_test)
elapsed = time.time() - start

print()
print(f"SVM training+evaluation took {elapsed:.1f} seconds")
if elapsed > 120:
    print("WARNING: SVM took longer than 2 minutes.")

# ---------------------------------------------------------------------------
# Part B — K-NN
# ---------------------------------------------------------------------------
print()
print("######################################################")
print("# PART B — K-Nearest Neighbors")
print("######################################################")
print()
print(
    "NOTE: KNeighborsClassifier does not support class_weight. K-NN classifies a point "
    "by majority vote among its nearest neighbors, with no mechanism to upweight the rare "
    "'failure' class during that vote. On a dataset where failures are only ~2% of windows, "
    "this means K-NN is structurally biased toward predicting the majority ('normal') class, "
    "especially for k=5 where a failure point needs 3+ of its 5 nearest neighbors to also be "
    "failures to be classified as one — a high bar when failures are rare in feature space."
)
print()

knn_model = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
knn_results = evaluate("K-NN", knn_model, X_train_scaled, y_train, X_test_scaled, y_test)

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
print(f"{'SVM (RBF)':40s}{svm_results['recall']:>10.2f}{svm_results['f1']:>10.2f}{svm_results['auroc']:>10.2f}")
print(f"{'K-NN (k=5)':40s}{knn_results['recall']:>10.2f}{knn_results['f1']:>10.2f}{knn_results['auroc']:>10.2f}")
