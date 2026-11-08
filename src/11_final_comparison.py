import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

"""Master comparison script for CIA-2: retrains every model from scripts
04-10 under one roof, with one consistent evaluation routine, so every row in
the final table is computed the exact same way (same split, same scaler, same
metric definitions). This is the script whose CSV/plot output goes directly
into the report. The "Train Recall" column exists specifically to expose
overfitting: a model whose train recall is much higher than its test recall
has memorized training patterns that don't generalize.
"""

sns.set_theme(style="whitegrid")

FEATURES_PATH = "data/processed/metropt3_features.csv"
FIGURES_DIR = "reports/figures"
REPORTS_DIR = "reports"
SPLIT_DATE = "2020-06-01"
MODEL_COLOR = "#2C6E9E"

df = pd.read_csv(FEATURES_PATH, index_col="timestamp", parse_dates=True)

# Chronological split for every model in this comparison: train on earlier
# months, test on later ones. Using the SAME split for all models is what
# makes the comparison table fair — any random split would let different
# runs get different (not directly comparable) train/test partitions.
train = df.loc[df.index < SPLIT_DATE]
test = df.loc[df.index >= SPLIT_DATE]

feature_cols = [c for c in df.columns if c != "failure"]

X_train, y_train = train[feature_cols], train["failure"]
X_test, y_test = test[feature_cols], test["failure"]

# One scaler, fit on train only, shared by every model below — consistent
# preprocessing is as important to a fair comparison as the consistent split.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rows = []


def best_f1_threshold(y_true, y_proba):
    # Sweeps candidate thresholds and picks the one maximizing F1 on
    # whichever set is passed in (always TRAIN data for every call site
    # below) — this is the standalone version of the threshold-tuning
    # strategy from 07_threshold_tuning.py and 09_mlp_iforest.py, needed here
    # for any model lacking a class_weight knob (MLP) or where we additionally
    # want a tuned-threshold variant alongside the default-threshold one
    # (RF, SVM).
    thresholds = np.linspace(0.01, 0.99, 99)
    scores = [f1_score(y_true, (y_proba >= t).astype(int)) for t in thresholds]
    return thresholds[int(np.argmax(scores))]


def add_row(name, y_train_true, y_train_pred, y_test_true, y_test_pred, y_test_score):
    # Precision/recall/F1 on the failure class specifically (not accuracy):
    # accuracy is dominated by the ~98% normal class and stays high even for
    # a model that never predicts a failure. AUROC is computed from
    # continuous scores (proba or anomaly score) and summarizes ranking
    # quality independent of any single threshold choice.
    precision = precision_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, pos_label=1, zero_division=0)
    auroc = roc_auc_score(y_test_true, y_test_score)
    tn, fp, fn, tp = confusion_matrix(y_test_true, y_test_pred).ravel()
    # Train Recall, evaluated with the SAME prediction rule (threshold or
    # .predict()) used on test, is the overfitting diagnostic for this table:
    # a large train-recall vs test-recall gap means the model (or its tuned
    # threshold) fit training-set quirks that don't hold up on unseen data.
    train_recall = recall_score(y_train_true, y_train_pred, pos_label=1, zero_division=0)

    rows.append(
        {
            "Model": name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AUROC": auroc,
            "FP": fp,
            "FN": fn,
            "Train Recall": train_recall,
        }
    )


# ---------------------------------------------------------------------------
# 1. Logistic Regression
# ---------------------------------------------------------------------------
# class_weight='balanced' is the standard imbalance fix used across this
# project wherever the estimator supports it. Default threshold (.predict())
# is used here — no separate tuning step, serving as the reference baseline.
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
add_row(
    "Logistic Regression",
    y_train,
    lr.predict(X_train_scaled),
    y_test,
    lr.predict(X_test_scaled),
    lr.predict_proba(X_test_scaled)[:, 1],
)

# ---------------------------------------------------------------------------
# 2. Decision Tree
# ---------------------------------------------------------------------------
# max_depth=3 is the best depth found by the validation curve in
# 05_decision_tree.py — max_depth is this model's regularization parameter,
# directly analogous to lambda in ridge/lasso: smaller max_depth = stronger
# regularization = simpler, less overfit tree.
dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
dt.fit(X_train_scaled, y_train)
add_row(
    "Decision Tree (depth=3)",
    y_train,
    dt.predict(X_train_scaled),
    y_test,
    dt.predict(X_test_scaled),
    dt.predict_proba(X_test_scaled)[:, 1],
)

# ---------------------------------------------------------------------------
# 3. Random Forest — default 0.5 and tuned threshold
# ---------------------------------------------------------------------------
# Same config as 06_random_forest.py (max_depth=10 regularizes each tree,
# class_weight='balanced' corrects training-time imbalance). Reported at both
# the default 0.5 cutoff and a train-tuned threshold, so the comparison table
# shows directly whether tuning the threshold on train data generalizes.
rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
rf.fit(X_train_scaled, y_train)

rf_train_proba = rf.predict_proba(X_train_scaled)[:, 1]
rf_test_proba = rf.predict_proba(X_test_scaled)[:, 1]

add_row(
    "Random Forest (default 0.5)",
    y_train,
    (rf_train_proba >= 0.5).astype(int),
    y_test,
    (rf_test_proba >= 0.5).astype(int),
    rf_test_proba,
)

rf_threshold = best_f1_threshold(y_train, rf_train_proba)
add_row(
    f"Random Forest (tuned {rf_threshold:.2f})",
    y_train,
    (rf_train_proba >= rf_threshold).astype(int),
    y_test,
    (rf_test_proba >= rf_threshold).astype(int),
    rf_test_proba,
)

# ---------------------------------------------------------------------------
# 4. SVM RBF — default 0.5 and tuned threshold
# ---------------------------------------------------------------------------
svm = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

svm_train_proba = svm.predict_proba(X_train_scaled)[:, 1]
svm_test_proba = svm.predict_proba(X_test_scaled)[:, 1]

# NOTE: SVC.predict() uses the decision boundary, not predict_proba >= 0.5 —
# predict_proba comes from a separate Platt-scaling calibration that can
# disagree with the decision function. Use .predict() for the "default" row.
add_row(
    "SVM RBF (default 0.5)",
    y_train,
    svm.predict(X_train_scaled),
    y_test,
    svm.predict(X_test_scaled),
    svm_test_proba,
)

svm_threshold = best_f1_threshold(y_train, svm_train_proba)
add_row(
    f"SVM RBF (tuned {svm_threshold:.2f})",
    y_train,
    (svm_train_proba >= svm_threshold).astype(int),
    y_test,
    (svm_test_proba >= svm_threshold).astype(int),
    svm_test_proba,
)

# ---------------------------------------------------------------------------
# 5. K-NN
# ---------------------------------------------------------------------------
# K-NN has no class_weight option — with k=5 a failure point needs 3+ of its
# 5 nearest neighbors to also be failures to be classified as one, a high bar
# when failures are only ~2% of the feature space. Included as-is (no
# tuning) to illustrate how badly a plain majority-vote model handles
# imbalance without correction.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
add_row(
    "K-NN (k=5)",
    y_train,
    knn.predict(X_train_scaled),
    y_test,
    knn.predict(X_test_scaled),
    knn.predict_proba(X_test_scaled)[:, 1],
)

# ---------------------------------------------------------------------------
# 6. MLP — tuned threshold
# ---------------------------------------------------------------------------
# No class_weight support here either; imbalance is handled purely by
# threshold tuning below. early_stopping=True is MLP's own regularization
# mechanism (stop before the network fully memorizes train data).
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    max_iter=500,
    early_stopping=True,
    random_state=42,
)
mlp.fit(X_train_scaled, y_train)

mlp_train_proba = mlp.predict_proba(X_train_scaled)[:, 1]
mlp_test_proba = mlp.predict_proba(X_test_scaled)[:, 1]

mlp_threshold = best_f1_threshold(y_train, mlp_train_proba)
add_row(
    f"MLP (tuned {mlp_threshold:.2f})",
    y_train,
    (mlp_train_proba >= mlp_threshold).astype(int),
    y_test,
    (mlp_test_proba >= mlp_threshold).astype(int),
    mlp_test_proba,
)

# ---------------------------------------------------------------------------
# 7. Isolation Forest — unsupervised, trained on normal train rows only
# ---------------------------------------------------------------------------
# The only model in this table that never sees y_train during fit(). It's
# included to compare a purely unsupervised anomaly detector against
# supervised classifiers — useful in scenarios where labeled failures aren't
# available (see 09_mlp_iforest.py for the full discussion).
X_train_normal = X_train_scaled[y_train.values == 0]

iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
iso.fit(X_train_normal)

iso_train_pred = (iso.predict(X_train_scaled) == -1).astype(int)
iso_test_pred = (iso.predict(X_test_scaled) == -1).astype(int)
iso_test_score = -iso.decision_function(X_test_scaled)  # higher = more anomalous

add_row(
    "Isolation Forest (unsupervised)",
    y_train,
    iso_train_pred,
    y_test,
    iso_test_pred,
    iso_test_score,
)

# ---------------------------------------------------------------------------
# 8. Logistic Regression on PCA (10 components, 95% variance)
# ---------------------------------------------------------------------------
# 10 components is the number found in 10_pca_analysis.py to retain 95% of
# variance in the 25-feature set. Included to directly test whether
# dimensionality reduction helps or hurts the strongest baseline model.
pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

lr_pca = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_pca.fit(X_train_pca, y_train)

add_row(
    "Logistic Regression (PCA, 10 comps)",
    y_train,
    lr_pca.predict(X_train_pca),
    y_test,
    lr_pca.predict(X_test_pca),
    lr_pca.predict_proba(X_test_pca)[:, 1],
)

# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------
results = pd.DataFrame(rows)
results = results[["Model", "Precision", "Recall", "F1", "AUROC", "FP", "FN", "Train Recall"]]

print("######################################################")
print("# FINAL MODEL COMPARISON (failure class, test set)")
print("######################################################")
print()

display = results.copy()
for col in ["Precision", "Recall", "F1", "AUROC", "Train Recall"]:
    display[col] = display[col].map(lambda v: f"{v:.3f}")
print(display.to_string(index=False))

results.to_csv(f"{REPORTS_DIR}/model_comparison.csv", index=False)
print()
print(f"Saved: {REPORTS_DIR}/model_comparison.csv")

# ---------------------------------------------------------------------------
# Best model summary
# ---------------------------------------------------------------------------
best_f1 = results.loc[results["F1"].idxmax(), "Model"]
best_auroc = results.loc[results["AUROC"].idxmax(), "Model"]
best_recall = results.loc[results["Recall"].idxmax(), "Model"]

print()
print(f"Best model by F1: {best_f1}")
print(f"Best model by AUROC: {best_auroc}")
print(f"Best model by Recall: {best_recall}")

# ---------------------------------------------------------------------------
# F1 bar chart, sorted descending
# ---------------------------------------------------------------------------
sorted_results = results.sort_values("F1", ascending=False)

fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(sorted_results["Model"][::-1], sorted_results["F1"][::-1], color=MODEL_COLOR)
ax.set_xlabel("F1 score (failure class)")
ax.set_title("Model Comparison: F1 Score (sorted descending)", fontsize=14, fontweight="bold")
ax.set_xlim(0, 1)
for i, v in enumerate(sorted_results["F1"][::-1]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/model_comparison_f1.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIGURES_DIR}/model_comparison_f1.png")
