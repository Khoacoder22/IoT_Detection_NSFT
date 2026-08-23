# Spectral NFST: Running and Evaluation Guide

This guide describes the implementation of **Spectral NFST for Scalable
Multiclass Novelty Detection in IoT Systems** from `AD_with_Khoa_MND.pdf` and
shows how to run it on every dataset variant included in this repository.

Windows users who only need to run experiments should start with the simpler
Vietnamese guide: [HUONG_DAN_CHAY_SPECTRAL_NFST.md](HUONG_DAN_CHAY_SPECTRAL_NFST.md).
It provides ready-made PowerShell scripts for a single run, resumable grids, and
result ranking, so users do not need to write `foreach` or `Import-Csv` code.

## 1. Available datasets

The repository contains **10 CSV dataset variants across 8 dataset families**.
ToN-IoT and IoTID20 each have two size variants.

| # | Variant | Rows | Classes | Features |
|---:|---|---:|---:|---:|
| 1 | BoT_IoT_1000 | 10,118 | 6 | 22 |
| 2 | CIC_IoT2023_1000 | 10,000 | 6 | 46 |
| 3 | ToN_IoT_1000 | 9,597 | 8 | 41 |
| 4 | ToN_IoT_2000 | 18,103 | 8 | 41 |
| 5 | UNSW_NB15_1000 | 7,174 | 8 | 44 |
| 6 | IoTID20_1000 | 3,000 | 3 | 79 |
| 7 | IoTID20_2000 | 10,000 | 5 | 79 |
| 8 | N_BaIoT_1000 | 9,000 | 9 | 115 |
| 9 | Edge_IIoTset_1000 | 15,000 | 15 | 42 |
| 10 | 5G_NIDD_1000 | 9,000 | 9 | 48 |

The suffix is the registered sampling limit used to build the repository file;
it is not always equal to the final row count because datasets have different
numbers of classes and some category mappings filter or merge labels.

## 2. Algorithm phases

Training has five phases:

1. **Kernel similarity mapping**: construct the training kernel matrix
   `K[i,j] = k(x_i, x_j)`. RBF is the recommended strictly positive-definite
   kernel.
2. **Targeted manifold decomposition**: build a complete affinity graph inside
   each original class, compute its maximum spanning tree, and remove the
   `Q_j - 1` weakest tree edges. This creates exactly `Q_j` components for class
   `j`.
3. **Global component relabeling**: assign one global component ID to every
   discovered class sub-manifold while retaining its original class label.
4. **Indicator matrix construction**: create normalized matrix `H`, with
   `H[i,k] = 1/sqrt(|C_k|)` when sample `i` belongs to component `k`.
5. **Null projection and centroids**: form an orthonormal basis `V` for the
   centered kernel range, solve the projected within-scatter null problem, keep
   `Q - 1` directions, and compute the projected centroid of every component.

Closed-set inference computes a test sample's kernel similarities, projects it,
finds its nearest component centroid, and returns that component's original
class. Novelty inference additionally uses the nearest-centroid distance as
anomaly score `A`; a sample is labeled `-1` when `A > tau`.

## 3. Preprocessing used by the runner

For every dataset variant, the existing CPAI pipeline performs:

1. dataset-specific label selection, leakage-column removal, and category remap;
2. stratified 80/20 train/test split using the selected seed;
3. scaling fitted only on the training partition;
4. per-class Local Outlier Factor removal from attack training classes;
5. optional polynomial feature expansion when `--poly` is greater than 1.

The recommended Spectral NFST configuration is:

- kernel: `rbf`;
- scaler: `QuantileTransformer`;
- polynomial features: disabled (`--poly -1`);
- components per original class: `2`;
- random seed: `42`.

## 4. Metrics written to the CSV

The classification runner writes `results/spectral_nfst/classification.csv`.
The backward-compatible `run_spectral_nfst.py` alias retains the old
`results/spectral_nfst/summary.csv` default. Each row contains:

| Column | Meaning |
|---|---|
| MCC | Multiclass Matthews correlation coefficient, range -1 to 1. Higher is better. |
| ACC | Repository-compatible aggregate one-vs-rest accuracy, reported as a percentage. This is not ordinary sample accuracy for multiclass data. |
| TPR Macro | Macro recall: recall calculated per class and averaged, in percent. |
| FPR | Aggregate one-vs-rest false-positive rate, in percent. Lower is better. |
| PPV Macro | Macro precision, in percent. |
| F1 Macro | Macro F1-score, in percent. |
| AUC | Binary ROC AUC only; multiclass runs store `-1`. |
| CFS Matrix | Multiclass confusion matrix. |
| Training time | Model fitting time in seconds. |
| Test time | Prediction time in seconds. |

`NDR` and `AUC_N` remain `-1` because the classification runner does not choose
a novelty threshold `tau`. Novelty evaluation requires a validation protocol
with held-out unknown classes; the paper provides the decision rule but does not
specify one universal threshold.

MCC and macro F1 are the most useful headline metrics for these imbalanced,
multiclass datasets. Also inspect per-class recall and the confusion matrix.

## 5. Installation

From the repository root:

```powershell
python -m pip install pandas numpy scipy scikit-learn pytest
```

## 6. Recommended command for all 10 variants

Spectral NFST uses dense kernel and decomposition matrices. A full run on
10,000-18,000 rows can require many gigabytes of memory and substantial time.
Start with 100 samples per class to validate the complete ten-variant pipeline:

```powershell
python code/run_spectral_nfst_classification.py --kernel rbf --scaler QuantileTransformer --poly -1 --components-per-class 2 --samples-per-class 100 --seed 42
```

Results are appended to:

```text
results/spectral_nfst/classification.csv
```

Use a different output file to avoid appending duplicate rows:

```powershell
python code/run_spectral_nfst_classification.py --components-per-class 2 --samples-per-class 100 --output results/spectral_nfst/run_q2_seed42.csv
```

## 7. Full-data command

Omit `--samples-per-class` to use every row in every registered variant:

```powershell
python code/run_spectral_nfst_classification.py --kernel rbf --scaler QuantileTransformer --poly -1 --components-per-class 2 --seed 42 --output results/spectral_nfst/full_q2_seed42.csv
```

Run variants sequentially, as the provided script does. Parallel full-data runs
can multiply the already large dense-matrix memory requirement.

## 8. Run one dataset variant

Example for ToN-IoT 2,000:

```powershell
python code/run_spectral_nfst_classification.py --dataset ToN_IoT --limit 2000 --kernel rbf --scaler QuantileTransformer --poly -1 --components-per-class 2 --samples-per-class 100 --seed 42 --output results/spectral_nfst/ToN_IoT_2000_q2.csv
```

For the full ToN-IoT 2,000 variant, omit `--samples-per-class 100`.
Only ToN-IoT and IoTID20 have a 2,000-row-per-class variant; use limit 1,000
for the other dataset families.

The classification-only CLI accepts these experiment inputs:

- `--dataset`: one dataset family; omit it to run all 10 variants.
- `--limit`: registered rows-per-class variant, normally 1000 or 2000.
- `--kernel`: `linear`, `poly`, `rbf`, `sigmoid`, `abel`, `laplacian`, or
  `sobolev`. Spectral NFST does not accept `none` even though it appears in the
  shared kernel choice list.
- `--scaler`: `QuantileTransformer`, `StandardScaler`, `MinMaxScaler`,
  `RobustScaler`, or `Normalizer`.
- `--poly`: `-1` keeps the normal feature flow; `0` additionally applies the
  existing independent-row preprocessing; `2` or `3` first creates polynomial
  interaction features and then applies that preprocessing.
- `--components-per-class`: target MaxST partitions `Q_j` for each class.
- `--samples-per-class`: optional class-balanced cap for quick tests. Omit it
  for the complete selected CSV.
- `--seed`: controls the deterministic sampling and train/test split.
- `--output`: CSV file to which the run is appended.

### 8.1 PowerShell parameter grid for one dataset

The recommended beginner command is the resumable wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File tools/spectral_nfst/run_grid_classification.ps1 -Dataset BoT_IoT -Limit 1000
```

The longer loop below is retained for users who want to understand or customize
the raw PowerShell orchestration.

This example evaluates 3 kernels x 3 scalers x 3 component counts on
BoT-IoT. It uses one fixed seed and appends the 27 configurations to one CSV:

```powershell
$kernels = @("rbf", "laplacian", "linear")
$scalers = @("QuantileTransformer", "StandardScaler", "RobustScaler")
$components = @(1, 2, 3)
$resultFile = "results/spectral_nfst/grid_BoT_IoT_1000.csv"

foreach ($kernel in $kernels) {
    foreach ($scaler in $scalers) {
        foreach ($q in $components) {
            python code/run_spectral_nfst_classification.py `
                --dataset BoT_IoT `
                --limit 1000 `
                --kernel $kernel `
                --scaler $scaler `
                --poly -1 `
                --components-per-class $q `
                --samples-per-class 100 `
                --seed 42 `
                --output $resultFile
        }
    }
}
```

Use the sample cap only for an initial search. Re-run the strongest few
configurations without `--samples-per-class` and with several seeds before
reporting a final result. Parameter selection should ultimately use a validation
split; repeatedly choosing the best configuration on the test set gives an
optimistically biased estimate.

## 9. Novelty detection from Python

The command-line evaluation above measures closed-set multiclass classification.
For Algorithm 3 novelty detection, fit the estimator and pass a threshold chosen
on validation data:

```python
from cpai import SpectralNFST

model = SpectralNFST(
    n_components=2,
    kernel="rbf",
    novelty_threshold=0.25,
    novelty_label=-1,
)
model.fit(X_train, y_train)

y_closed = model.predict(X_test)
y_novelty, anomaly_scores = model.predict_with_novelty(X_test)
```

Do not select `tau` on the final test set. A suitable protocol holds out one or
more classes as unknown, chooses `tau` using separate validation data, and then
reports novelty recall, false-positive rate, and AUROC on the untouched test set.

## 10. Separate multi-novelty command

List the cleaned class labels before selecting held-out unknown classes:

```powershell
python code/run_spectral_nfst_novelty.py --dataset BoT_IoT --limit 1000 --list-classes
```

Run multi-novelty with both `theft` and `scan` absent from training:

```powershell
python code/run_spectral_nfst_novelty.py --dataset BoT_IoT --limit 1000 --unknown-labels theft scan --threshold 0.25 --kernel rbf --scaler QuantileTransformer --poly -1 --components-per-class 2 --samples-per-class 100 --seed 42
```

This command computes open-set MCC/F1 plus NDR, novelty FPR, AUROC and AUPRC.
The threshold is mandatory and is never optimized on test labels. Select it in
a separate validation experiment before using the command for final reporting.
