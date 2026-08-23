# CPAI — NMF for Anomaly Detection

Research artifacts and code for the paper *CPAI NMF for Anomaly Detection*, plus the Spectral NFST algorithm described in `AD_with_Khoa_MND.pdf`. Implemented projections include HHH, HHHv2, KNFST, and SpectralNFST.

## Layout

```
CPAI/
├── paper/                         Paper artifacts
│   ├── CPAI_NMF_for_anomaly_detection.pdf
│   ├── Revised_CPAI_NMF_for_anomaly_detection.pdf
│   ├── CPAI-Review_Round_1.docx
│   ├── figures/                   PNG figures used in the paper
│   └── tables/                    .tgn table sources
│
├── code/
│   ├── cpai/                      Python package (reusable modules)
│   │   ├── paths.py               PROJECT_ROOT / DATA_DIR / RESULTS_DIR
│   │   ├── datasets.py            load_dataset(name) — reads data/<ds>.csv, cleans, returns DataFrame
│   │   ├── kernels.py             compute_kernel() — linear/poly/rbf/sigmoid + abel/laplacian/sobolev
│   │   ├── gaussian_elim.py       gaussian_elimination, form_independent
│   │   ├── preprocessing.py       end-to-end preprocess() (scale → poly → split → LOF → form_independent)
│   │   ├── models.py              HHH, HHHv2, KNFST, SpectralNFST
│   │   ├── baselines.py           sklearn baselines (KNN, LDA, NuSVC, SGD, GauNB, NC, RNC)
│   │   └── metrics.py             calc_index() — MCC, ACC, TPR/FPR/PPV/F1 macro, AUC
│   │
│   ├── train_evaluate.py          Single-run CLI: one (dataset, model, kernel, scaler, poly)
│   ├── run_experiments.py         Parallel orchestrator — presets + multiprocessing Pool
│   ├── scene_b_runner.py          Scene-B grid (CPAI-OvA/OvR × kernels × scalers × seeds)
│   ├── baseline_runner.py         Baseline + KNFST multiseed grid
│   ├── ablation_ridge.py          Ridge-regularisation ablation
│   ├── ablation_gamma.py          RBF gamma ablation
│   ├── rerun_best.py              Re-runs the best config per dataset
│   ├── build_n_baiot.py           Builds data/N_BaIoT_1000.csv from the raw N-BaIoT archive
│   ├── summarize_scene_b.py       runs/*.json → results/scene_b/summary.csv
│   ├── summarize_baselines.py     runs/*.json → results/baselines_default/summary.csv
│   ├── make_figures.py            Paper figures (ROC, confusion matrices) → paper/figures/
│   ├── make_report.py             Regenerates REPORT.md tables
│   ├── benchmark_parallel.py      Wall-clock/throughput benchmark
│   ├── profile_memory.py          Peak-RSS profiling
│   ├── profile_memory_v2.py       Peak-RSS profiling (subprocess isolation)
│   └── legacy/                    Original notebooks preserved for reference
│       ├── 01_preprocess.ipynb
│       ├── 02_train_and_evaluate.ipynb
│       └── Data_preprocess-Copy1.ipynb
│
├── data/                          Input datasets (balanced IoT traffic CSVs)
│   ├── 5G_NIDD_1000.csv
│   ├── BoT_IoT_1000.csv
│   ├── CIC_IoT2023_1000.csv
│   ├── N_BaIoT_1000.csv           Built by code/build_n_baiot.py (raw archive is gitignored)
│   ├── ToN_IoT_1000.csv
│   ├── ToN_IoT_2000.csv
│   ├── UNSW_NB15_1000.csv
│   ├── edge_iiotset_1000.csv
│   ├── iotid20_1000.csv
│   └── iotid20_2000.csv
│
├── results/
│   ├── scene_b/                   Main CPAI grid — summary.csv committed, runs/ gitignored
│   ├── baselines_default/         Baselines + KNFST — summary.csv committed, runs/ gitignored
│   ├── ablation_ridge/            Ridge ablation — summary.csv committed, runs/ gitignored
│   ├── ablation_gamma/            Gamma ablation — summary.csv committed, runs/ gitignored
│   ├── rerun_best/                Best-config re-runs
│   ├── final/                     CSVs that feed TongHop_res_new.xlsx
│   │   ├── <dataset>/
│   │   │   ├── HHH.csv            HHH / HHHv2 runs (standard kernels only)
│   │   │   ├── extend.csv         Additional HHHv2 runs
│   │   │   └── Competitors.csv    Baseline models (GauNB, KNN, LDA, NC, SVM, RNC, SGD, KNFST)
│   │   └── TongHop_res_new.xlsx   Final summary workbook for the paper
│   │
│   └── backup/                    Preserved raw dumps — to be triaged after re-run
│       ├── per_run/               Exploratory per-run CSVs (mostly HHHv2_Projection_*, AddLDA_*)
│       ├── summary/               Superseded HHH.csv files + old TongHop_res.xlsx
│       └── dumps/                 Duplicate copies of extend / Competitors files
│
└── _archive/                      Zips kept as a safety net; safe to delete later
```

## Running experiments

Install deps (one-off):

```bash
python3 -m pip install --user pandas numpy scipy scikit-learn openpyxl
```

Single run (one row appended to an output CSV):

```bash
python3 code/train_evaluate.py \
    --dataset BoT_IoT --model HHHv2 --kernel rbf \
    --scaler QuantileTransformer --poly -1 \
    --output results/runs/my_run.csv
```

Spectral NFST with two MaxST sub-manifolds per known class:

```bash
python3 code/run_spectral_nfst_classification.py \
    --kernel rbf --components-per-class 2 \
    --scaler QuantileTransformer --poly -1
```

The library estimator accepts a scalar, a class-ordered sequence, or a mapping
of original class labels to the paper's target counts `Q_j`. Its `predict()`
method implements closed-set multiclass inference; `predict_with_novelty(X,
threshold=tau)` returns both thresholded labels and the paper's nearest-manifold
anomaly score.

See [SPECTRAL_NFST_README.md](SPECTRAL_NFST_README.md) for the five algorithm
phases, metric definitions, memory guidance, and commands for all ten registered
dataset variants.

For a beginner-friendly Windows workflow, including resumable parameter grids,
Excel-lock protection, and a one-command result viewer, see
[HUONG_DAN_CHAY_SPECTRAL_NFST.md](HUONG_DAN_CHAY_SPECTRAL_NFST.md).

Classification and multi-novelty testing now have separate terminal entry points:

```bash
# Closed-set classification on all ten registered variants
python3 code/run_spectral_nfst_classification.py --samples-per-class 100

# First inspect valid labels, then hold one or more out of training
python3 code/run_spectral_nfst_novelty.py --dataset BoT_IoT --limit 1000 --list-classes
python3 code/run_spectral_nfst_novelty.py --dataset BoT_IoT --limit 1000 \
    --unknown-labels theft scan --threshold 0.25 --samples-per-class 100
```

The old `code/run_spectral_nfst.py` command remains as a backward-compatible
alias for the classification CLI.

Parallel orchestrator (reproduces a preset grid):

```bash
# Paper Tables 2/3: CPAI-OvA/OvR across 8 kernels × 5 datasets × 3 scalers × 3 polys
python3 code/run_experiments.py --preset tables_2_3 --workers 8

# Paper Tables 4-8: baselines + KNFST
python3 code/run_experiments.py --preset tables_4_8 --workers 8

# Custom: BoT-IoT only, HHHv2 with all kernels
python3 code/run_experiments.py --datasets BoT_IoT --models HHHv2 \
    --kernels rbf abel laplacian sobolev --polys -1 --workers 4
```

Output: one CSV per `(dataset, model)` under `results/runs/<dataset>/<model>.csv`, with the same schema as `results/final/*.csv` so the columns line up with the Excel.

Notes:
- All three papers-but-missing kernels (Abel, Laplacian, Sobolev) are implemented in `cpai/kernels.py`. Formulas use standard definitions; adjust if the paper uses different ones.
- The legacy notebooks in `code/legacy/` still have hardcoded `/home/jupyter-hanx/...` paths and depend on an external `utils` module that is not in this repo. Treat them as reference, not runnable.

## Status

- **Label leakage fixed (BoT_IoT, CIC_IoT2023) — affected results need re-running.**
  `_DROP_ALWAYS` in `cpai/datasets.py` listed the lowercase `category` but not the
  capitalised `Category` column that both datasets also ship, and never listed
  CIC-IoT2023's `Binary` flag. Both are derived from the label, so they survived
  cleaning and entered the feature matrix. On CIC-IoT2023, `Category` maps 1:1 onto
  the 6 remapped classes — a feature that fully determines the target.
  Matched single-run check (HHHv2, rbf, QuantileTransformer, poly=-1, seed=42):

  | dataset | MCC before | MCC after |
  |---|---|---|
  | CIC_IoT2023 | 0.9836 | 0.8627 |
  | BoT_IoT     | 0.9928 | 0.9886 |

  Everything under `results/` for these two datasets predates the fix. The other six
  datasets are unaffected (verified: no alt-label column survives cleaning).
- `data/edge_iiotset_1000.csv` and `data/5G_NIDD_1000.csv` were built from raw sources
  outside this repo (`Papers/cnNFST`); no builder script is committed for them.
- `results/final/` contains the canonical CSVs that feed `TongHop_res_new.xlsx` (verified by cross-matching MCC + config keys).
- `results/backup/` holds everything else — superseded HHH.csv files, per-run exploratory runs, and duplicate dumps. Kept in case re-runs don't reproduce the Excel exactly.
- About 40–50 rows per dataset in the Excel (Abel/Laplacian/Sobolev kernel experiments) have **no CSV source file** in the project — likely run separately and pasted into Excel. Flag for reproducibility when re-running.
- The notebooks are queued for conversion to Python scripts.
