# Spectral NFST: Running and Evaluation Guide

This guide describes the implementation of **Spectral NFST for Scalable
Multiclass Novelty Detection in IoT Systems** from `AD_with_Khoa_MND.pdf` and
shows how to run it on every dataset variant included in this repository.

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

The batch runner writes `results/spectral_nfst/summary.csv`. Each row contains:

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
python code/run_spectral_nfst.py --kernel rbf --scaler QuantileTransformer --poly -1 --components-per-class 2 --samples-per-class 100 --seed 42
```

Results are appended to:

```text
results/spectral_nfst/summary.csv
```

Use a different output file to avoid appending duplicate rows:

```powershell
python code/run_spectral_nfst.py --components-per-class 2 --samples-per-class 100 --output results/spectral_nfst/run_q2_seed42.csv
```

## 7. Full-data command

Omit `--samples-per-class` to use every row in every registered variant:

```powershell
python code/run_spectral_nfst.py --kernel rbf --scaler QuantileTransformer --poly -1 --components-per-class 2 --seed 42 --output results/spectral_nfst/full_q2_seed42.csv
```

Run variants sequentially, as the provided script does. Parallel full-data runs
can multiply the already large dense-matrix memory requirement.

## 8. Run one dataset variant

Example for ToN-IoT 2,000:

```powershell
python code/train_evaluate.py --dataset ToN_IoT --limit 2000 --model SpectralNFST --kernel rbf --scaler QuantileTransformer --poly -1 --components-per-class 2 --samples-per-class 100 --seed 42 --output results/spectral_nfst/ToN_IoT_2000_q2.csv
```

For the full ToN-IoT 2,000 variant, omit `--samples-per-class 100`.

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
