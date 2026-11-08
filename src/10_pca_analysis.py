import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

"""Principal Component Analysis on the MetroPT-3 features: how many
components are needed to retain most of the variance, what the data looks
like projected onto the first two components, and whether training Logistic
Regression on the reduced feature space helps or hurts versus the full
25-feature baseline.
"""

sns.set_theme(style="whitegrid")

FEATURES_PATH = "data/processed/metropt3_features.csv"
FIGURES_DIR = "reports/figures"
SPLIT_DATE = "2020-06-01"

NORMAL_COLOR = "#2C6E9E"
FAILURE_COLOR = "#C0392B"

df = pd.read_csv(FEATURES_PATH, index_col="timestamp", parse_dates=True)

# Chronological split, as throughout the project.
train = df.loc[df.index < SPLIT_DATE]
test = df.loc[df.index >= SPLIT_DATE]

feature_cols = [c for c in df.columns if c != "failure"]

X_train, y_train = train[feature_cols], train["failure"]
X_test, y_test = test[feature_cols], test["failure"]

# PCA is explicitly a variance/distance-based method, so standardizing first
# is essential — without it, high-variance-scale sensors (e.g. pressure in
# bar) would dominate the principal components over low-variance ones (e.g.
# normalized duty cycle) purely due to units, not actual signal importance.
# Fit on train only, same leakage-avoidance reasoning as every other script.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Part A — PCA explained variance
# ---------------------------------------------------------------------------
print("######################################################")
print("# PART A — PCA EXPLAINED VARIANCE")
print("######################################################")
print()

# Fit PCA on scaled TRAIN data only — same reasoning as the scaler: the
# principal component directions are a statistic of the data, and fitting
# them on test data too would leak future information into the transform
# applied to training features.
pca_full = PCA(random_state=42)
pca_full.fit(X_train_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("Explained variance ratio per component:")
for i, ratio in enumerate(explained, start=1):
    print(f"  PC{i:<3d} {ratio:.4f}   (cumulative: {cumulative[i - 1]:.4f})")

n_90 = int(np.argmax(cumulative >= 0.90)) + 1
n_95 = int(np.argmax(cumulative >= 0.95)) + 1

print()
print(f"Components needed for 90% variance: {n_90}")
print(f"Components needed for 95% variance: {n_95}")

n_show = min(15, len(explained))
fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(range(1, n_show + 1), explained[:n_show], color=NORMAL_COLOR, alpha=0.8, label="Per-component variance")
ax.plot(range(1, n_show + 1), cumulative[:n_show], color=FAILURE_COLOR, marker="o", linewidth=2, label="Cumulative variance")
ax.set_xlabel("Principal component")
ax.set_ylabel("Explained variance ratio")
ax.set_xticks(range(1, n_show + 1))
ax.set_ylim(0, 1.05)
ax.axhline(0.90, color="gray", linestyle="--", linewidth=1)
ax.axhline(0.95, color="gray", linestyle="--", linewidth=1)
ax.text(n_show, 0.90, " 90%", va="bottom", ha="right", fontsize=9, color="gray")
ax.text(n_show, 0.95, " 95%", va="bottom", ha="right", fontsize=9, color="gray")

ax.legend(loc="center right")
ax.set_title("PCA: Explained Variance by Component", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/pca_explained_variance.png", dpi=150)
plt.close(fig)
print()
print(f"Saved: {FIGURES_DIR}/pca_explained_variance.png")

# ---------------------------------------------------------------------------
# Part B — 2D scatter plot
# ---------------------------------------------------------------------------
print()
print("######################################################")
print("# PART B — PCA 2D PROJECTION")
print("######################################################")
print()

pca_2d = PCA(n_components=2, random_state=42)
pca_2d.fit(X_train_scaled)
test_2d = pca_2d.transform(X_test_scaled)

normal_mask = y_test.values == 0
failure_mask = y_test.values == 1

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(
    test_2d[normal_mask, 0],
    test_2d[normal_mask, 1],
    color=NORMAL_COLOR,
    s=8,
    alpha=0.3,
    label="Normal",
)
ax.scatter(
    test_2d[failure_mask, 0],
    test_2d[failure_mask, 1],
    color=FAILURE_COLOR,
    s=30,
    alpha=0.8,
    label="Failure",
)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("PCA: Compressor Data Projected to 2 Dimensions", fontsize=14, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGURES_DIR}/pca_2d_scatter.png", dpi=150)
plt.close(fig)
print(f"Saved: {FIGURES_DIR}/pca_2d_scatter.png")
print(f"PC1 + PC2 explain {pca_2d.explained_variance_ratio_.sum():.1%} of total variance.")

# ---------------------------------------------------------------------------
# Part C — Logistic Regression on PCA-reduced features
# ---------------------------------------------------------------------------
print()
print("######################################################")
print("# PART C — LOGISTIC REGRESSION ON PCA-REDUCED FEATURES")
print("######################################################")
print()

# 95% variance retained is a conventional cutoff for "keep almost all the
# information, drop the components that are mostly noise/redundancy." Using
# n_95 computed above (rather than a hardcoded number) keeps this tied to
# what the data actually needs.
pca_95 = PCA(n_components=n_95, random_state=42)
X_train_pca = pca_95.fit_transform(X_train_scaled)
X_test_pca = pca_95.transform(X_test_scaled)

print(f"Using {n_95} components (95% variance retained).")
print()

lr_pca = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr_pca.fit(X_train_pca, y_train)

y_pred_pca = lr_pca.predict(X_test_pca)
y_proba_pca = lr_pca.predict_proba(X_test_pca)[:, 1]

print("=== Logistic Regression (PCA): CLASSIFICATION REPORT (TEST SET) ===")
print(classification_report(y_test, y_pred_pca, target_names=["Normal", "Failure"]))

report_dict = classification_report(y_test, y_pred_pca, target_names=["Normal", "Failure"], output_dict=True)
recall_pca = report_dict["Failure"]["recall"]
f1_pca = report_dict["Failure"]["f1-score"]
auroc_pca = roc_auc_score(y_test, y_proba_pca)

print(f"=== Logistic Regression (PCA): AUROC ===")
print(f"{auroc_pca:.4f}")

print()
print("######################################################")
print("# COMPARISON: FULL FEATURES vs. PCA-REDUCED")
print("######################################################")
print()
header = f"{'Model':40s}{'Recall':>10s}{'F1':>10s}{'AUROC':>10s}"
print(header)
print("-" * len(header))
print(f"{'Logistic Regression (full features)':40s}{0.90:>10.2f}{0.76:>10.2f}{0.91:>10.2f}")
print(f"{'Logistic Regression (PCA, ' + f'{n_95} comps)':40s}{recall_pca:>10.2f}{f1_pca:>10.2f}{auroc_pca:>10.2f}")

print()
print("=== CONCLUSION ===")
delta_f1 = f1_pca - 0.76
delta_recall = recall_pca - 0.90
delta_auroc = auroc_pca - 0.91
if abs(delta_f1) < 0.02 and abs(delta_auroc) < 0.02:
    verdict = "made little to no difference"
elif delta_f1 > 0:
    verdict = "helped"
else:
    verdict = "hurt"
print(
    f"Reducing {len(feature_cols)} features down to {n_95} principal components "
    f"(95% variance retained) {verdict} performance: recall changed by {delta_recall:+.2f}, "
    f"F1 by {delta_f1:+.2f}, and AUROC by {delta_auroc:+.2f} relative to the full-feature "
    f"Logistic Regression baseline. PCA compresses the feature space but is agnostic to the "
    f"failure label, so it is not guaranteed to preserve the directions that best separate "
    f"failures from normal operation."
)
