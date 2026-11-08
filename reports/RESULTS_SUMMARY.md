# CIA-2 Results Summary: MetroPT-3 Predictive Maintenance

Numbers come from `reports/*.csv` and from script outputs (scripts 04, 05, 10, 12, 13, 14, 15 were re-run to confirm). Nothing here is estimated.

**How to read the trust levels**

| Tag | Scripts | Meaning |
|---|---|---|
| **EXPLORATORY** | 04-12 | Single train/test split. Some choices were tuned on the test set (marked below), so these scores are optimistic. Use for comparing model families, not as final performance. |
| **LEAKAGE-FREE** | 13-15 | Chronological train/validation/test split. Every choice (C, threshold, persistence) was made on validation; test was evaluated once. |

---

## 1. Dataset and labeling (scripts 01-02)
- **Source:** MetroPT-3 air compressor, 1 s logs: 1,516,948 rows, 2020-02-01 to 2020-09-01.
- **Labels:** The raw CSV has no failure column. Labels come from four maintenance-report windows (29,971 failure seconds):

| Failure | Window | Duration |
|---|---|---|
| F1 | 2020-04-18 00:00 to 23:59 | about 24 h |
| F2 | 2020-05-29 23:30 to 2020-05-30 06:00 | 6.5 h |
| F3 | 2020-06-05 10:00 to 2020-06-07 14:30 | 52.5 h |
| F4 | 2020-07-15 14:30 to 19:00 | 4.5 h |

## 2. Feature engineering (script 03)
- **Windows:** 5-minute windows, labelled failure if any second inside was failure. Result: **50,772 windows, 996 failures (1.96%)**, 25 features plus label.
- **Analog sensors** (TP2, TP3, H1, DV_pressure, Reservoirs, Oil_temperature, Motor_current): mean and std.
- **Extras:** `Motor_current_max`, the mean of 8 digital signals, `duty_cycle` (fraction of the window with COMP=0) and `switch_count` (COMP transitions).
- **NaN rows:** Windows with NaN (single-sample std) were dropped.

## 3. Splits
| Scripts | Train | Validation | Test |
|---|---|---|---|
| 04-12 (**exploratory**) | before 2020-06-01: 28,640 windows, 367 failures (F1, F2) | none | from 2020-06-01: 22,132 windows, 629 failures (F3, F4) |
| 13-15 (**leakage-free**) | before 2020-05-15: 24,662 windows, 288 failures (F1) | 2020-05-15 to 05-31: 3,978 windows, 79 failures (F2) | from 2020-06-01: 22,132 windows, 629 failures (F3, F4) |

The scaler is fit on train only everywhere. In 13-15 the final model is refit on train+validation (scaler refit too) before the single test evaluation.

## 4. Seven-model comparison (script 11, **EXPLORATORY**, test set)
Source: `reports/model_comparison.csv`. Rows marked "(tuned t)" use a threshold picked to maximize F1 on the **training** set (script 11). That is why they collapse on test.

| Model | Precision | Recall | F1 | AUROC | FP | FN | Train recall |
|---|---|---|---|---|---|---|---|
| Logistic Regression (C=1) | 0.654 | 0.898 | **0.757** | 0.913 | 299 | 64 | 0.984 |
| Decision Tree (depth=3) | 0.674 | 0.207 | 0.316 | 0.614 | 63 | 499 | 0.995 |
| Random Forest (0.5) | 0.989 | 0.145 | 0.252 | 0.874 | 1 | 538 | 1.000 |
| Random Forest (tuned 0.98) | 0.000 | 0.000 | 0.000 | 0.874 | 0 | 629 | 0.992 |
| SVM RBF (0.5) | 0.619 | 0.382 | 0.472 | 0.924 | 148 | 389 | 0.997 |
| SVM RBF (tuned 0.58) | 0.143 | 0.002 | 0.003 | 0.924 | 6 | 628 | 0.989 |
| K-NN (k=5) | 0.600 | 0.014 | 0.028 | 0.529 | 6 | 620 | 0.984 |
| MLP (tuned 0.23) | 0.143 | 0.002 | 0.003 | 0.863 | 6 | 628 | 0.978 |
| Isolation Forest (unsupervised) | 0.035 | 0.057 | 0.043 | 0.949 | 991 | 593 | 0.000 |
| Logistic Regression (PCA, 10 comps) | 0.570 | 0.897 | 0.697 | 0.932 | 426 | 65 | 0.984 |

- **Counts:** The CSV has 10 rows, of which 7 are distinct model families.
- **Caveat:** Script 07 (RF threshold sweep and PR curve) tuned its threshold directly on test. Those numbers are not in this table.
- **Main pattern:** Train recall is high (0.98-1.00) for nearly every supervised model, but test recall is very low for everything except Logistic Regression.

## 5. Decision tree depth (script 05, **EXPLORATORY**; depth chosen on test F1)
- **Overfitting demo:**

| Tree | Train recall | Test recall | Train AUROC | Test AUROC |
|---|---|---|---|---|
| Unlimited depth (fitted depth 16) | 1.000 | 0.213 | 1.000 | 0.605 |
| max_depth = 5 | 1.000 | 0.045 | 1.000 | 0.519 |

- **Counter-intuitive result:** The shallower tree has the larger train/test recall gap.
- **Depth curve** (test F1): depth 2 = 0.314, **depth 3 = 0.316 (best)**, depth 4 = 0.153, depth 5 = 0.072, depth 7 = 0.300, depth 10 and deeper = 0.314. The curve is non-monotonic, and the best depth was picked by test F1.

## 6. Logistic Regression regularization sweep (script 12, **EXPLORATORY; C tuned on test**)
| C | Train F1 | Test F1 | Test recall | Test AUROC |
|---|---|---|---|---|
| 0.001 | 0.392 | 0.776 | 0.908 | 0.995 |
| **0.01** | 0.506 | **0.902** | 0.903 | 0.950 |
| 0.1 | 0.502 | 0.841 | 0.900 | 0.924 |
| 0.5 | 0.495 | 0.757 | 0.898 | 0.915 |
| 1 | 0.492 | 0.757 | 0.898 | 0.913 |
| 5 | 0.496 | 0.582 | 0.890 | 0.899 |
| 10 | 0.498 | 0.592 | 0.890 | 0.896 |
| 50 | 0.512 | 0.538 | 0.881 | 0.886 |
| 100 | 0.511 | 0.542 | 0.887 | 0.886 |

- **Optimistic best:** The best test F1 of 0.902 at C=0.01 was chosen using the test set. Do **not** quote it as a performance claim.
- **Honest version:** Script 13's validation-based selection gives test F1 0.776 (Section 8).

## 7. PCA (script 10, **EXPLORATORY**)
- **Variance:** PC1 and PC2 explain 58.4%. 7 components reach 90% of the variance and 10 components reach 95%.
- **Logistic Regression on 10 components:** recall 0.90, F1 0.70, AUROC 0.93. The full-feature model has recall 0.90, F1 0.76, AUROC 0.91. PCA slightly lowered F1 and slightly raised AUROC.

## 8. Fair benchmark (script 13, **LEAKAGE-FREE**)
- **C selection:** Chosen on validation, **C = 0.001** (validation F1 0.301). Validation F1 was 0.16-0.18 for every C ≥ 0.003.

| Evaluation | Precision | Recall | F1 | AUROC | FP | FN |
|---|---|---|---|---|---|---|
| Chronological test | 0.678 | 0.908 | **0.776** | 0.995 | 271 | 58 |
| Random 80/20 split (same C) | 0.430 | 0.985 | 0.599 | 0.991 | n/a | n/a |

- **Event level:**
  - F3 was detected, with the first alarm only 0.17 h before onset.
  - F4 was missed.
  - There were 254 false-alarm windows over 10.27 clean weeks (24.7 per week), or 170 episodes (**16.6 per week**).
- **Leakage demo caveat:** Random-split F1 came out lower than chronological F1 here, so inflation is **not** visible in F1. Recall was higher on the random split (0.985 vs 0.908), which matches the leakage direction. The two test sets also have different failure rates, so F1 isn't directly comparable.

## 9. Final alarm policy (script 14, **LEAKAGE-FREE**)
- **Selection on validation:**
  - C = 0.0003 (validation F1 0.346). C = 0.0001 scored 0.342.
  - Policy: **threshold 0.95, persistence k = 2**. Validation lead time was +0.17 h at 0.59 false-alarm episodes per week.
  - Several policies tied on false alarms, and the choice rests on a single validation failure (F2).

| Metric (test) | Script 13 (thr 0.5, k=1) | Script 14 |
|---|---|---|
| F1 | 0.776 | **0.802** |
| Precision / recall | 0.678 / 0.908 | 0.979 / 0.679 |
| AUROC | 0.995 | 0.997 |
| FP / FN | 271 / 58 | 9 / 202 |
| F3 (52.5 h) | detected, +0.17 h | detected, +0.08 h |
| F4 (4.5 h) | missed | **missed** |
| False-alarm episodes per week | 16.6 | **0.0** |

- **Why F4 is missed:** Its probability peaks at 0.695, below the 0.95 threshold. The feature `DV_pressure_mean` (largest coefficient) is about 34 SD from normal in F3 but about normal in F4. F4 is also short (55 windows).
- **Lead-time caveat:** F3 detection comes only about 5 minutes before onset, which is detection rather than warning.

## 10. Early-warning test (script 15, **LEAKAGE-FREE**)
**Precursor check** (z-scores vs normal; no model; "before onset" covers the 7-day, 24h-2h and last-2h periods):

| Failure | Features with \|z\| ≥ 2 before onset |
|---|---|
| F1 | `switch_count` (+7.3 then +10.5), `Oil_temperature_mean` (to −2.1) |
| F2 | `switch_count` (+4.7 in the 7d-24h period only) |
| F3 | none |
| F4 | `Oil_temperature_mean` (to +2.6), `TP2_mean` (to +2.2) |

**Models** ("within H hours before onset", in-failure windows dropped; C chosen on validation AUROC):

| Horizon | Test AUROC | Precision | Recall | F3 "warned" | F4 "warned" |
|---|---|---|---|---|---|
| 2 h | 0.465 | 0.002 | 0.50 | yes (1.67 h) | yes (2.00 h) |
| 6 h | 0.545 | 0.007 | 0.67 | yes (5.83 h) | yes (6.00 h) |
| 24 h | 0.535 | 0.025 | 0.73 | yes (24.00 h) | yes (17.08 h) |

**Conclusion: no usable early-warning signal.**
- **AUROC:** All test AUROCs are near 0.5, and validation AUROC was 0.40-0.54.
- **Why "warned: yes" is misleading:** Precision is 0.2-2.5%, so the model alarms constantly. Earliest alarms land at the very start of each window, which is chance, not a precursor.

## 11. Literature comparison (`reports/literature_comparison.csv`)
The literature rows use different labeling, windowing and splits, so they are **not directly comparable** to ours.

| Source | Model | Split | P | R | F1 |
|---|---|---|---|---|---|
| IJACSA 2025 (Akshitha Mary et al.) | LSTM + Autoencoder | 80/20 | 0.98 | 0.434 | 0.60 |
| IJACSA 2025 baseline | Sparse Autoencoder | 80/20 | 0.80 | 0.40 | 0.533 |
| Najjar et al. 2023 | Random Forest | not stated | n/a | n/a | about 0.85 |
| Ours, LR (val-selected C=0.001) | chronological | | 0.678 | 0.908 | 0.776 |
| Ours, LR (same C) | random 80/20 (leaky) | | 0.430 | 0.985 | 0.599 |

These rows are from script 13, not script 14's alarm policy (F1 0.802).

## 12. Figures (`reports/figures/`)
| File | Script | What it shows |
|---|---|---|
| `timeline_overview.png` | 02 | Full 7-month timeline of motor current, oil temperature and TP2, with failure windows shaded. |
| `failure1_zoom.png` | 02 | Zoomed view of failure 1 (sensors and the failure label). |
| `class_comparison_boxplots.png` | 02 | Sensor distributions for normal vs failure windows. |
| `dt_validation_curve.png` | 05 | Decision-tree train vs test F1 across max_depth. |
| `rf_precision_recall_curve.png` | 07 | Random Forest PR curve with the default and best-F1 thresholds (test-tuned). |
| `pca_explained_variance.png` | 10 | Per-component and cumulative explained variance. |
| `pca_2d_scatter.png` | 10 | Test windows projected onto PC1/PC2, normal vs failure. |
| `model_comparison_f1.png` | 11 | F1 comparison across all models. |
| `lr_regularization_curve.png` | 12 | Logistic Regression train vs test F1 across C. |
| `alarm_policy_test_timeline.png` | 14 | Test-period probability, 0.95 threshold, alarm episodes, and F3/F4 shaded. |
| `precursor_zscores.png` | 15 | Pre-onset z-scores of six features across four periods, one line per failure. |

Descriptions for `model_comparison_f1.png` and the 02/05/07 figures come from the script code, not from viewing the images.

---

## Bottom line
- **Detection:** A heavily regularized Logistic Regression detects in-progress failures at about F1 0.78-0.80 on a clean chronological test.
- **Failure 4:** It still misses the short failure F4.
- **Early warning:** The data does not support it.
- **Leakage-free scope:** Only scripts 13-15 are leakage-free, and they rest on four failure events.
