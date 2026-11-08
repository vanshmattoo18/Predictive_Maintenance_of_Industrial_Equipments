import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler

"""Does MetroPT-3 support EARLY WARNING (predicting before onset), not just
detection?  Part 1: precursor analysis (no model). Part 2: models that predict
'within H hours before onset'. Part 3: summary and conclusion.
"""

FEATURES_PATH = "data/processed/metropt3_features.csv"
FIG_PATH = "reports/figures/precursor_zscores.png"
HORIZONS = [2, 6, 24]
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1]
PRECURSOR_FEATS = ["Motor_current_mean", "Oil_temperature_mean", "TP2_mean",
                   "DV_pressure_mean", "duty_cycle", "switch_count"]
T = pd.Timestamp
FAILS = {"F1": (T("2020-04-18 00:00"), T("2020-04-18 23:59")),
         "F2": (T("2020-05-29 23:30"), T("2020-05-30 06:00")),
         "F3": (T("2020-06-05 10:00"), T("2020-06-07 14:30")),
         "F4": (T("2020-07-15 14:30"), T("2020-07-15 19:00"))}
H_ = pd.Timedelta("1h")

df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True).sort_index()
FEATURES = [c for c in df.columns if c != "failure"]
# Same chronological split as src/14.
train = df[df.index < "2020-05-15"]
val = df[(df.index >= "2020-05-15") & (df.index < "2020-06-01")]
test = df[df.index >= "2020-06-01"]


def in_failure(index):
    m = pd.Series(False, index=index)
    for s, e in FAILS.values():
        m |= (index >= s) & (index <= e)
    return m


# ======================= PART 1: precursor check ==========================
print("=" * 78, "\nPART 1: precursor check (z-scores vs normal; no model)\n", "=" * 78, sep="")
# 'Normal' = more than 7 days from any failure (before start or after end).
# Pure descriptive statistics, so no train/test separation is needed here;
# nothing from this part feeds any model or threshold.
normal_mask = pd.Series(True, index=df.index)
for s, e in FAILS.values():
    normal_mask &= ~((df.index >= s - pd.Timedelta("7D")) & (df.index <= e + pd.Timedelta("7D")))
normal = df.loc[normal_mask, PRECURSOR_FEATS]
mu, sd = normal.mean(), normal.std().replace(0, np.nan)
print(f"Normal windows: {len(normal)} (std of switch_count etc. taken from these)")
PERIODS = ["7d-24h", "24h-2h", "last 2h"]
zres = {}
for name, (s, e) in FAILS.items():
    cuts = {"7d-24h": (s - pd.Timedelta("7D"), s - pd.Timedelta("24h")),
            "24h-2h": (s - pd.Timedelta("24h"), s - 2 * H_),
            "last 2h": (s - 2 * H_, s)}
    tab = pd.DataFrame(index=PRECURSOR_FEATS, columns=["normal"] + PERIODS, dtype=float)
    tab["normal"] = 0.0
    for p, (a, b) in cuts.items():
        seg = df.loc[(df.index >= a) & (df.index < b), PRECURSOR_FEATS]
        tab[p] = ((seg.mean() - mu) / sd) if len(seg) else np.nan
    zres[name] = tab
    print(f"\n{name} ({s} -> {e}) z-scores vs normal:")
    print(tab.to_string(float_format=lambda v: f"{v:+.2f}"))
    moved = [f for f in PRECURSOR_FEATS if (tab.loc[f, PERIODS].abs() >= 2).any()]
    print("  Features with |z|>=2 before onset:", moved if moved else "none")

# Plot: one subplot per feature, x = 4 periods, one line per failure.
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
xs = ["normal"] + PERIODS
for ax, f in zip(axes.ravel(), PRECURSOR_FEATS):
    for name, tab in zres.items():
        ax.plot(xs, tab.loc[f, xs].values, marker="o", label=name)
    ax.axhline(0, color="gray", lw=0.7)
    ax.set_title(f)
    ax.set_ylabel("z vs normal")
axes[0, 0].legend()
fig.suptitle("Pre-onset feature z-scores (period relative to failure onset)")
fig.tight_layout()
fig.savefig(FIG_PATH, dpi=130)
print(f"\nSaved plot to {FIG_PATH}")

# ======================= PART 2: early-warning models =====================
print("\n", "=" * 78, "\nPART 2: early-warning models\n", "=" * 78, sep="")


def pre_label(index, H):
    """1 if the window lies within H hours BEFORE any failure onset."""
    y = pd.Series(0, index=index)
    for s, _ in FAILS.values():
        y[(index >= s - H * H_) & (index < s)] = 1
    return y


def prep(part, H):
    """Label and DROP windows inside actual failures. They are dropped because
    the question is pure pre-failure prediction: in-failure windows are
    trivially abnormal (detection), and labelling them 0 would punish the model
    for correctly flagging a failure, while labelling them 1 would mix up
    detection with early warning."""
    keep = ~in_failure(part.index)
    p = part[keep]
    return p[FEATURES], pre_label(p.index, H)


summary = []
for H in HORIZONS:
    print(f"\n--- Horizon H = {H} h ---")
    Xtr, ytr = prep(train, H)
    Xva, yva = prep(val, H)
    Xte, yte = prep(test, H)
    print(f"positives: train={ytr.sum()} val={yva.sum()} test={yte.sum()}")
    if min(ytr.sum(), yva.sum(), yte.sum()) == 0:
        print("A split has no positive windows for this horizon; skipping.")
        summary.append((H, np.nan, "skipped", "skipped"))
        continue
    # Scaler on train only, C chosen on VALIDATION only (by AUROC: threshold-
    # free, so it is not distorted by the arbitrary 0.5 cut-off or by
    # degenerate all-zero predictions). The test set never influences choices.
    sc = StandardScaler().fit(Xtr)
    best_C, best_auc = None, -1
    for C in C_GRID:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(sc.transform(Xtr), ytr)
        auc = roc_auc_score(yva, m.predict_proba(sc.transform(Xva))[:, 1])
        print(f"  C={C:<7} val AUROC={auc:.4f}")
        if auc > best_auc:
            best_C, best_auc = C, auc
    print(f"C selected on validation = {best_C}")
    # Refit scaler+model on train+validation (all data before test), test once.
    Xtv, ytv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
    sc2 = StandardScaler().fit(Xtv)
    final = LogisticRegression(C=best_C, class_weight="balanced", max_iter=1000).fit(sc2.transform(Xtv), ytv)
    prob = pd.Series(final.predict_proba(sc2.transform(Xte))[:, 1], index=Xte.index)
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(yte, prob)
    print(f"TEST pre-failure class: AUROC={auc:.4f} "
          f"P={precision_score(yte, pred, zero_division=0):.4f} "
          f"R={recall_score(yte, pred, zero_division=0):.4f} "
          f"F1={f1_score(yte, pred, zero_division=0):.4f}")
    warned = {}
    for name in ("F3", "F4"):
        s, _ = FAILS[name]
        w = pred[(pred.index >= s - H * H_) & (pred.index < s)]
        hit = w[w == 1]
        if len(hit):
            warned[name] = f"yes ({(s - hit.index.min()) / H_:.2f} h ahead)"
        else:
            warned[name] = "no"
        print(f"  {name}: alarm in {H}h before onset? {warned[name]}"
              f"  (max prob in window = {prob[(prob.index >= s - H * H_) & (prob.index < s)].max():.3f})")
    summary.append((H, auc, warned["F3"], warned["F4"]))

# ======================= PART 3: summary ==================================
print("\n", "=" * 78, "\nPART 3: summary\n", "=" * 78, sep="")
print(f"{'Horizon':>8s} {'Test AUROC':>11s}  {'F3 warned?':<22s}{'F4 warned?':<22s}")
for H, auc, a, b in summary:
    print(f"{str(H) + ' h':>8s} {auc:>11.4f}  {a:<22s}{b:<22s}")

aucs = [s[1] for s in summary if not np.isnan(s[1])]
both = [s for s in summary if s[2].startswith("yes") and s[3].startswith("yes")]
print("\nConclusion:")
if not aucs:
    print("  No horizon had positives in all splits, so early warning could not be tested.")
else:
    best = max(aucs)
    if best < 0.6:
        print(f"  Best test AUROC is {best:.3f}, near 0.5: NO predictive signal for early warning.")
    elif both:
        hs = [s[0] for s in both]
        print(f"  Both test failures were warned at horizon(s) {hs} h; best AUROC {best:.3f}. "
              "Still, only 2 test failures (and 1 per train/val) means this is weak evidence.")
    else:
        print(f"  Best test AUROC is {best:.3f}, but no horizon warned BOTH test failures at "
              "threshold 0.5: warning is not reliable.")
print("  Caveat: with one failure in train, one in validation and two in test, any "
      "horizon result rests on very few events and cannot establish reliability.")
