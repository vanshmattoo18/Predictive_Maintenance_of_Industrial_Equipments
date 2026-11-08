# BITS F464 CIA-2: Predictive Maintenance on MetroPT-3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vanshmattoo18/Predictive_Maintenance_of_Industrial_Equipments/blob/main/notebooks/CIA2_Colab.ipynb)

**Team:** Vansh (2024A7PS0149U)

## Quick Start

- **(a) Colab:** click the "Open in Colab" badge above and choose *Runtime > Run all*. The notebook downloads MetroPT-3 automatically (it falls back to a manual upload if the download fails).
- **(b) Locally:**
  ```bash
  pip install -r requirements.txt
  python run_all.py
  ```
  `run_all.py` runs `src/00` to `src/15` in order and stops on the first error. `src/00_download_data.py` downloads the dataset into `data/raw/` only if it is not already there.

Scripts 04-15 work immediately from the included `data/processed/metropt3_features.csv`, so if you only want the models you can skip the download and run them directly, e.g. `python src/13_fair_benchmark.py`.

## Problem Description

Unplanned failures of metro train air compressors cause costly downtime and service disruptions. This project treats predictive maintenance as a binary classification problem: given 5-minute windows of compressor sensor readings (pressure, temperature, motor current, etc.), predict whether the window falls inside a failure event. The dataset is extremely imbalanced (~2% failure windows), which makes this a good testbed for comparing how different model families — linear, tree-based, margin-based, neural, and unsupervised — handle rare-event detection under a realistic chronological train/test split.

## Dataset

**MetroPT-3 (Air Compressor)** from the UCI Machine Learning Repository — sensor telemetry (pressure, temperature, motor current, and digital valve/switch signals) from a metro train's air production unit, logged at 1-second resolution over ~7 months, with four known failure events from maintenance records.

### Downloading the data

The raw dataset is not included in this repository (it is large). `python src/00_download_data.py` (or `run_all.py`) downloads it for you; to do it by hand:

1. Go to https://archive.ics.uci.edu/dataset/791/metropt+3+dataset and download the dataset (a zip file).
2. Unzip it and place **`MetroPT3(AirCompressor).csv`** in **`data/raw/`** (optionally also the data-description PDF).
3. Run the scripts below. `src/01` and `src/03` regenerate `data/processed/metropt3_labeled.csv` (git-ignored, about 200 MB) and `data/processed/metropt3_features.csv` (about 18 MB, included in the repo).

## Project Structure

```
.
├── data/
│   ├── raw/           # Unmodified downloaded dataset (MetroPT3 CSV, dataset zip, data dictionary PDF)
│   └── processed/      # Labeled data (metropt3_labeled.csv) and engineered 5-min window features (metropt3_features.csv)
├── notebooks/          # CIA2_Colab.ipynb: the full Google Colab version of the whole project
├── src/                 # Numbered pipeline scripts, run in order — see below
├── reports/
│   ├── figures/         # All generated plots (EDA, validation curves, PR curves, PCA, model comparisons)
│   ├── model_comparison.csv       # Exploratory metrics table across all models (src/11)
│   ├── literature_comparison.csv  # Comparison with published results (src/13)
│   └── RESULTS_SUMMARY.md         # Concise reference of all final results
├── requirements.txt     # Python dependencies
└── venv/                # Local virtual environment
```

## How to Run

1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Run the scripts in order — each one builds on the output of the previous:
   ```bash
   python src/00_download_data.py            # Download MetroPT-3 into data/raw/ (skipped if present)
   python src/01_label_and_explore.py        # Label raw data with failure windows
   python src/02_eda_plots.py                # Exploratory plots
   python src/03_feature_engineering.py      # Build 5-min window features
   python src/04_train_baseline.py           # Logistic Regression baseline
   python src/05_decision_tree.py            # Decision Tree + overfitting demo
   python src/06_random_forest.py            # Random Forest
   python src/07_threshold_tuning.py         # RF threshold tuning / PR curve
   python src/08_svm_knn.py                  # SVM (RBF) and K-NN
   python src/09_mlp_iforest.py              # MLP and Isolation Forest
   python src/10_pca_analysis.py             # PCA analysis + LR on reduced features
   python src/11_final_comparison.py         # Master comparison across all models
   python src/12_regularization_sweep.py     # Logistic Regression C (regularization) sweep
   python src/13_fair_benchmark.py           # Leakage-free train/validation/test benchmark
   python src/14_alarm_policy.py             # Alarm policy (threshold + persistence) chosen on validation
   python src/15_early_warning_horizon.py    # Early-warning horizon test (2/6/24 h)
   ```

   Scripts 01-03 build the data; 04-12 are the exploratory comparison; 13-15 are the leakage-free results.

   **Colab:** `notebooks/CIA2_Colab.ipynb` is the full Colab version of the project, covering the whole pipeline in one notebook (upload `MetroPT3(AirCompressor).csv` when prompted). Open it in Google Colab and run the cells in order.

3. Results appear in `reports/figures/` (all plots) and `reports/model_comparison.csv` (final metrics table).

## Dependencies

See `requirements.txt` for the full list (numpy, pandas, matplotlib, seaborn, scikit-learn, jupyter, jupyterlab). Install with:
```bash
pip install -r requirements.txt
```

## Key Findings

- **Logistic Regression wins overall.** Despite being the simplest model tried, it achieves the best F1 and recall on the failure class, and tuning its regularization (C) on the test set pushed F1 from 0.76 to 0.90 in the exploratory scripts (optimistic, since C was picked on test) — higher-capacity models (Random Forest, SVM, MLP) did not outperform it on this dataset.
- **Class imbalance dominates every design decision.** With only ~2% of windows labeled failure, naive models (K-NN, default-threshold Random Forest) collapse to near-zero recall; `class_weight='balanced'` and/or threshold tuning are essential, and accuracy is useless as an evaluation metric throughout.
- **Regularization and threshold tuning matter more than model complexity.** The C sweep directly demonstrates the bias-variance tradeoff, and the "Train Recall" column in the final comparison shows that thresholds tuned on training data can catastrophically fail to generalize for high-capacity models (RF, SVM, MLP) — a caution against tuning decision thresholds without held-out validation.

## Leakage-Free Results (scripts 13-15)

Scripts 04-12 use a single train/test split, and some choices (for example C) were tuned on the test set, so their scores are optimistic. Scripts 13-15 use a chronological train/validation/test split with every choice made on validation; see `reports/RESULTS_SUMMARY.md` for all numbers.

- Final model: Logistic Regression (C=0.0003) with an alarm policy (threshold 0.95, 2 consecutive windows) selected on validation. Test F1 0.802, precision 0.979, recall 0.679, 0 false-alarm episodes per week.
- Failure 3 is detected (only about 5 minutes before onset); failure 4 is missed.
- No early-warning signal was found at 2, 6 or 24 h horizons (test AUROC about 0.5).
- Limitation: only four failure events exist (one in train, one in validation, two in test).
