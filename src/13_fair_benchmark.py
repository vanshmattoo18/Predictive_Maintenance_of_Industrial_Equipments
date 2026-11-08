import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""Fair benchmark for MetroPT-3 failure prediction with Logistic Regression.

A: chronological train/val/test split, C chosen on validation only.
B: event-level evaluation (detection, lead time, false alarms/week) on test.
C: leakage demonstration with a random 80/20 split.
D: comparison against published numbers (saved to CSV).
"""

FEATURES_PATH = "data/processed/metropt3_features.csv"
OUT_CSV = "reports/literature_comparison.csv"
C_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True).sort_index()
FEATURES = [c for c in df.columns if c != "failure"]
X_all, y_all = df[FEATURES], df["failure"].astype(int)


def make_model(C):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=1000)


def metrics(y, pred, score):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0),
                auroc=roc_auc_score(y, score), fp=fp, fn=fn, tp=tp)


# ============================ PART A ====================================
print("=" * 70, "\nPART A: chronological split\n", "=" * 70, sep="")
# Splits are defined purely by time, so no future sample is ever used to
# train a model that is then scored on the past.
train = df[df.index < "2020-05-15"]
val = df[(df.index >= "2020-05-15") & (df.index < "2020-06-01")]
test = df[df.index >= "2020-06-01"]
for name, part in [("Train", train), ("Validation", val), ("Test", test)]:
    print(f"{name:11s} n={len(part):6d}  failures={int(part.failure.sum()):4d}  "
          f"({part.index.min()} -> {part.index.max()})")

# Scaler fit on TRAIN ONLY: validation statistics must not leak into the
# preprocessing used for hyperparameter selection.
sc = StandardScaler().fit(train[FEATURES])
Xtr, Xva = sc.transform(train[FEATURES]), sc.transform(val[FEATURES])

# C is selected using the VALIDATION set only; the test set is not touched
# here, so test performance is not optimised through model selection.
print("\nC sweep (validation F1, failure class):")
val_f1 = {}
for C in C_GRID:
    m = make_model(C).fit(Xtr, train["failure"])
    val_f1[C] = f1_score(val["failure"], m.predict(Xva), zero_division=0)
    print(f"  C={C:<6} val F1={val_f1[C]:.4f}")
best_C = max(C_GRID, key=lambda c: val_f1[c])
print(f"\nC selected on validation = {best_C}")

# Final model: refit scaler AND model on train+validation (more recent data,
# still strictly before the test period), then evaluate on test exactly once.
trval = pd.concat([train, val])
sc_final = StandardScaler().fit(trval[FEATURES])
final = make_model(best_C).fit(sc_final.transform(trval[FEATURES]), trval["failure"])
Xte = sc_final.transform(test[FEATURES])
yte = test["failure"].astype(int)
test_pred = final.predict(Xte)
test_score = final.predict_proba(Xte)[:, 1]
A = metrics(yte, test_pred, test_score)
print("\nTEST (evaluated once, C=%s):" % best_C)
print(f"  Precision={A['precision']:.4f} Recall={A['recall']:.4f} "
      f"F1={A['f1']:.4f} AUROC={A['auroc']:.4f}  FP={A['fp']} FN={A['fn']} TP={A['tp']}")

# ============================ PART B ====================================
print("\n", "=" * 70, "\nPART B: event-level evaluation on test\n", "=" * 70, sep="")
alarms = pd.Series(test_pred, index=test.index).astype(bool)
events = [("Failure 3", "2020-06-05 10:00", "2020-06-07 14:30"),
          ("Failure 4", "2020-07-15 14:30", "2020-07-15 19:00")]
PRE = pd.Timedelta("24h")
for name, s, e in events:
    s, e = pd.Timestamp(s), pd.Timestamp(e)
    # Window opens 24h before failure start; only alarms inside it count, so
    # alarms raised before the window cannot be credited as detections.
    in_win = alarms[(alarms.index >= s - PRE) & (alarms.index <= e)]
    first = in_win[in_win].index.min() if in_win.any() else None
    if first is None:
        print(f"{name}: {s} -> {e}  NOT detected")
    else:
        lead = (s - first).total_seconds() / 3600
        print(f"{name}: {s} -> {e}  detected=True  first alarm={first}  "
              f"lead time={lead:+.2f} h ({'early' if lead > 0 else 'after start'})")

# False alarms only counted on "clean" test time: more than 24h from any
# failure (before or after), so near-failure alarms aren't penalised.
clean = pd.Series(True, index=test.index)
for _, s, e in events:
    clean &= ~((clean.index >= pd.Timestamp(s) - PRE) & (clean.index <= pd.Timestamp(e) + PRE))
clean_alarm = alarms & clean
step = pd.Timedelta("5min")
weeks = clean.sum() * step / pd.Timedelta("7D")
# Alarm episodes = contiguous runs of alarm samples (a run counts as one alarm).
runs = (clean_alarm & ~clean_alarm.shift(1, fill_value=False)).sum()
print(f"\nClean test time: {clean.sum()} windows = {weeks:.2f} weeks")
print(f"False-alarm windows: {int(clean_alarm.sum())}  -> {clean_alarm.sum() / weeks:.1f} per week")
print(f"False-alarm episodes (contiguous runs): {int(runs)}  -> {runs / weeks:.1f} per week")

# ============================ PART C ====================================
print("\n", "=" * 70, "\nPART C: leakage demonstration (random 80/20 split)\n", "=" * 70, sep="")
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
# Scaler still fit on the training portion only, so ANY inflation seen here
# comes from the split itself (temporal neighbours), not from the scaler.
sc_r = StandardScaler().fit(Xr_tr)
mr = make_model(best_C).fit(sc_r.transform(Xr_tr), yr_tr)
Xr_te_s = sc_r.transform(Xr_te)
C_ = metrics(yr_te, mr.predict(Xr_te_s), mr.predict_proba(Xr_te_s)[:, 1])
print(f"Random split: Precision={C_['precision']:.4f} Recall={C_['recall']:.4f} "
      f"F1={C_['f1']:.4f} AUROC={C_['auroc']:.4f}")
print(f"\nF1 comparison:  chronological test = {A['f1']:.4f}   random split = {C_['f1']:.4f}")
print("NOTE: random splitting lets adjacent 5-minute windows from the same failure\n"
      "appear in both train and test. Those windows are near-duplicates (heavy\n"
      "autocorrelation), so this split CAN inflate scores; chronological splitting forbids it.")
if C_["f1"] < A["f1"]:
    print("CAUTION: in this run random-split F1 is LOWER than chronological F1, so\n"
          "inflation is NOT visible in F1 here. Random-split test has ~2% base rate\n"
          "across all months (precision is diluted), while the chronological test\n"
          "set has different class balance; F1 is not comparable across them.")

# ============================ PART D ====================================
print("\n", "=" * 70, "\nPART D: literature comparison\n", "=" * 70, sep="")
note = "No - different labeling/windowing/splits"
rows = [
    ("IJACSA 2025 (Akshitha Mary et al.)", "LSTM+Autoencoder", "80/20", 0.98, 0.434, 0.60, note),
    ("IJACSA 2025 baseline", "Sparse Autoencoder", "80/20", 0.80, 0.40, 0.533, note),
    ("Najjar et al. 2023", "Random Forest", "not stated", np.nan, np.nan, 0.85, note + " (binary, F1 approx.)"),
    ("Ours", f"Logistic Regression (val-selected C={best_C})", "chronological",
     round(A["precision"], 4), round(A["recall"], 4), round(A["f1"], 4),
     "Strictest split; closest to deployment"),
    ("Ours", f"Logistic Regression (same C={best_C})", "random 80/20 (leaky)",
     round(C_["precision"], 4), round(C_["recall"], 4), round(C_["f1"], 4),
     "Only comparable to literature's 80/20 splits, and inflated by leakage"),
]
lit = pd.DataFrame(rows, columns=["Source", "Model", "Split", "Precision", "Recall", "F1", "Comparable?"])
lit.to_csv(OUT_CSV, index=False)
print(lit.to_string(index=False))
print(f"\nSaved to {OUT_CSV}")
