#!/usr/bin/env python3
"""Generate REPORT.md — a paper-style markdown report covering:

1. Environment (hardware + software).
2. Datasets (preprocessing, splits, sample counts, class counts).
3. Methodology recap (HHH / HHHv2 / kernels / baselines) + default hyperparameters.
4. Evaluation metrics (all reported).
5. Kernel study (Tables 2/3 analog): per-dataset × kernel best config 5-seed mean±std.
6. Baseline comparison (Tables 4-... analog): 7 per-dataset tables + summary, with
   Wilcoxon signed-rank and paired t-test p-values between CPAI (HHH/HHHv2) and each baseline.
7. Train/test timing (best config, mean over 5 seeds).
8. Confusion matrices for HHHv2 best-seed config per dataset.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

# Preload torch BEFORE scipy (macOS libomp fix)
try:
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpai.datasets import load_dataset, _DEFAULT_LIMIT, _CATEGORY_MAP, _LABEL_COLUMN
from cpai.paths import RESULTS_DIR, PROJECT_ROOT
from cpai.preprocessing import preprocess

warnings.filterwarnings("ignore")

REPORT_PATH = PROJECT_ROOT / "REPORT.md"

DATASETS = ["BoT_IoT", "IoTID20", "ToN_IoT", "N_BaIoT", "CIC_IoT2023", "Edge_IIoTset", "5G_NIDD"]

# Baselines to include (FTTransformer excluded per user request — kept in raw results)
BASELINES = ["KNN", "LDA", "NuSVC", "SGD", "GauNB", "NC", "RNC", "KNFST", "LGBM", "MLP", "TabNet", "SAINT"]

KERNELS = ["None", "linear", "poly", "rbf", "sigmoid", "abel", "laplacian", "sobolev"]
SCALERS = ["QuantileTransformer", "StandardScaler", "MinMaxScaler"]
SEEDS = [42, 43, 44, 45, 46]

METRICS_FULL = [
    ("MCC", "MCC", 4),
    ("ACC", "ACC (%)", 2),
    ("TPR_macro", "TPR macro (%)", 2),
    ("TPR_weighted", "TPR weighted (%)", 2),
    ("PPV_macro", "PPV macro (%)", 2),
    ("PPV_weighted", "PPV weighted (%)", 2),
    ("F1_macro", "F1 macro (%)", 2),
    ("F1_weighted", "F1 weighted (%)", 2),
    ("FPR", "FPR (%)", 3),
    ("AUROC_OvR_macro", "AUROC OvR macro (%)", 2),
    ("AUROC_OvR_weighted", "AUROC OvR weighted (%)", 2),
    ("AUROC_OvO_macro", "AUROC OvO macro (%)", 2),
    ("AUROC_binary", "AUROC binary (%)", 2),
]

METRICS_CORE = [
    ("MCC", "MCC", 4),
    ("ACC", "ACC", 2),
    ("TPR_macro", "TPR", 2),
    ("FPR", "FPR", 3),
    ("PPV_macro", "PPV", 2),
    ("F1_macro", "F1", 2),
]

# --------------------------------------------------------------------- helpers

def mean_std(series):
    m = np.mean(series)
    s = np.std(series, ddof=1) if len(series) > 1 else 0.0
    return m, s


def fmt(m, s, prec):
    return f"{m:.{prec}f} ± {s:.{prec}f}"


# ------------------------------------------------------------------ environment

def env_block() -> str:
    cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    pcores = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip()
    lcores = subprocess.check_output(["sysctl", "-n", "hw.ncpu"]).decode().strip()
    ram_gb = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip()) / 1024**3

    # package versions
    import numpy, scipy, sklearn
    import torch
    import lightgbm

    def _ver(pkg: str) -> str:
        try:
            return subprocess.check_output([sys.executable, "-m", "pip", "show", pkg],
                                           stderr=subprocess.DEVNULL).decode()
        except Exception:
            return ""

    tn_ver = [l.split(": ", 1)[1] for l in _ver("pytorch-tabnet").splitlines() if l.startswith("Version:")]
    ftt_ver = [l.split(": ", 1)[1] for l in _ver("tab-transformer-pytorch").splitlines() if l.startswith("Version:")]

    return f"""## Environment

| Category | Value |
|---|---|
| OS | {platform.platform()} |
| CPU | {cpu} |
| Physical cores | {pcores} |
| Logical cores | {lcores} |
| RAM | {ram_gb:.0f} GB |
| BLAS backend | Apple Accelerate |
| Python | {sys.version.split()[0]} |
| numpy | {numpy.__version__} |
| scipy | {scipy.__version__} |
| scikit-learn | {sklearn.__version__} |
| pytorch | {torch.__version__} |
| lightgbm | {lightgbm.__version__} |
| pytorch-tabnet | {tn_ver[0] if tn_ver else 'n/a'} |
| tab-transformer-pytorch (FTTransformer) | {ftt_ver[0] if ftt_ver else 'n/a'} |

**Thread configuration for experiments**: `VECLIB_MAXIMUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=6`
(limits BLAS to 6 threads per worker so 2 concurrent workers fit on 12 cores without oversubscription).
"""


# ---------------------------------------------------------------------- methods

def preprocessing_block() -> str:
    # Compute per-dataset sample counts (raw → after mapping → after split → after LOF)
    rows = []
    for ds in DATASETS:
        df, limit = load_dataset(ds)
        raw_n = len(df)
        n_classes = df["Label"].nunique()
        n_features = df.shape[1] - 1  # exclude label
        # Run preprocess to get actual train/test split sizes
        X_tr, y_tr, X_te, y_te, enc = preprocess(df, ds, poly=-1, kernel=None,
                                                  scaler="QuantileTransformer", seed=42)
        rows.append({
            "dataset": ds, "limit_per_class": limit, "raw_total": raw_n,
            "features": n_features, "n_classes": n_classes,
            "n_train_post_lof": len(y_tr), "n_test": len(y_te),
        })
    df = pd.DataFrame(rows)

    lines = []
    lines.append("## Datasets and Preprocessing")
    lines.append("")
    lines.append("### Data Sources and Label Remap")
    lines.append("")
    lines.append("Seven datasets — five from the original CPAI paper plus two added for this study (Edge-IIoTset and 5G-NIDD):")
    lines.append("")
    lines.append("| Dataset | Source | Label column (raw) | Label remap |")
    lines.append("|---|---|---|---|")
    lines.append("| BoT-IoT | Cyber Range Lab of UNSW Canberra | `subcategory` | Grouped into 6 classes: 0Normal, HTTP, TCP, UDP, scan (OS_Fingerprint + Service_Scan), theft (Data_Exfiltration + Keylogging) |")
    lines.append("| IoTID20 | Dept. of Computer Science, Kyungpook National Univ. | `Target` | No remap (5 native classes: 0Normal, DoS, MITM, Mirai, Scan) |")
    lines.append("| ToN-IoT | Cyber Range Lab of UNSW Canberra | `type` | Grouped into 8 classes: 0Normal, Malware (backdoor + ransomware), Scan, BruteForce, DDoS, WebAttack (xss + injection), DoS, MITM |")
    lines.append("| N-BaIoT | UCI ML Repository | per-device attack subtype | 9 classes: Benign + {ack, scan, syn, udp, udpplain, combo, junk, tcp} |")
    lines.append("| CIC-IoT2023 | Canadian Institute for Cybersecurity | `Label` | Grouped into 6 classes: 0Normal, DDoS/DoS, Mirai, Spoofing, Scan, Web |")
    lines.append("| Edge-IIoTset | Ferrag et al. 2022 | `Attack_type` | 15 native classes preserved |")
    lines.append("| 5G-NIDD | Samarakoon et al. 2022 | `Attack Type` | 9 native classes preserved |")
    lines.append("")
    lines.append("### Preprocessing Pipeline")
    lines.append("")
    lines.append("Same pipeline for every run (original CPAI procedure, `cpai/preprocessing.py::preprocess`):")
    lines.append("")
    lines.append("1. **Load and clean**: load CSV, apply per-dataset label remap above, drop identifier/payload string columns (IPs, URIs, MACs, timestamps). Label-encode remaining object columns. Drop all-NaN feature columns (e.g., BoT-IoT MAC fields).")
    lines.append("2. **Per-class subsampling**: sample `limit` rows per class (1 000 for all datasets except IoTID20 which uses 2 000) to produce a class-balanced dataset. Random state = 42.")
    lines.append("3. **Impute** (IoTID20 only): replace ±∞ with NaN, then impute NaNs with column mean.")
    lines.append("4. **Polynomial feature expansion** (conditional): when `poly > 1`, apply `PolynomialFeatures(poly, interaction_only=True)`.")
    lines.append("5. **Stratified 80/20 split**: `train_test_split(test_size=0.2, stratify=labels, random_state=seed)`. `seed` is one of {42, 43, 44, 45, 46}.")
    lines.append("6. **Label encoding**: `LabelEncoder` fitted on y_train, applied to y_test. `0Normal` always maps to class 0 (alphabetical sort).")
    lines.append("7. **Scaling**: fit the chosen scaler on X_train, transform both splits, then `nan_to_num(nan=0.0)`.")
    lines.append("8. **Per-class LOF outlier removal**: `LocalOutlierFactor(contamination=0.05)` applied separately to each attack class (class 0 / benign is passed through untouched).")
    lines.append("9. **Form-independent (poly ≠ -1 only)**: Gaussian elimination per class to keep a linearly independent subset of training rows. Skipped when `poly = -1` (the CPAI default).")
    lines.append("")
    lines.append("### Sample Counts")
    lines.append("")
    lines.append("Post-preprocessing sizes with `poly = -1`, `scaler = QuantileTransformer`, `seed = 42`. The test-set size is fixed at `total × 0.2` (LOF is applied only to the training half).")
    lines.append("")
    lines.append("| Dataset | Raw total | Per-class cap | Features | Classes | n_train (post-LOF) | n_test |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['raw_total']:,} | {r['limit_per_class']:,} | {r['features']} | {r['n_classes']} | {r['n_train_post_lof']:,} | {r['n_test']:,} |")
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------- method recap

def method_block() -> str:
    return """## Methodology Recap

### Proposed: CPAI-OvA (HHH) and CPAI-OvR (HHHv2)

Both solve a regularized kernel ridge-regression system with a **target-assignment** mechanism (not Fisher-criterion eigenvalue optimization as in NFST / KNFST):

For training data `X` with labels `y` (`c` classes, label-encoded to {0, …, c-1}):

**CPAI-OvA (HHH)** — single discriminant direction, target = class index:
- Kernel case: α = (K + λI)⁻¹ y, where K = k(X, X)
- Linear case: α = (X Xᵀ + λI)⁻¹ y, θ = Xᵀ α
- **Predict**: `round(f(x_test))`, clipped to [min(y), max(y)]; out-of-range → −1

**CPAI-OvR (HHHv2)** — one discriminant per class, targets = one-hot rows:
- α = (K + λI)⁻¹ Y, Y ∈ {0, 1}^(n × c) is the one-hot encoded y
- **Predict**: argminⱼ |fⱼ(x) − 1|

### Default Hyperparameters

| Hyperparameter | HHH | HHHv2 | Notes |
|---|---|---|---|
| Ridge λ (kernel path) | **1e-9** | **1e-5** | Matches the original notebook. HHH's tiny ridge works because target = class index is smoother. |
| Ridge λ (linear / kernel=None path) | 1e-5 | 1e-5 | |
| Solver | scipy.linalg.solve(assume_a='pos') | same | Cholesky, multi-RHS, BLAS-threaded |
| Solver fallback | LU with ridge boost to 10⁻³ | same | Triggered on LinAlgError (singular matrix) |

### Kernels

| Kernel | Formula | Notes |
|---|---|---|
| None | K(X, Y) = X Yᵀ | Linear Gram, dual formulation |
| linear | K(x, y) = xᵀy (sklearn) | |
| poly | K(x, y) = (γxᵀy + 1)ᵈ (sklearn default) | γ = 1/d, d = 3 |
| rbf | K(x, y) = exp(-γ‖x − y‖²) (sklearn) | γ = 1/d |
| sigmoid | K(x, y) = tanh(γxᵀy + 1) (sklearn) | γ = 1/d |
| abel | K(x, y) = exp(-α‖x − y‖₂) | **α = 0.1** (matches notebook) |
| laplacian | K(x, y) = exp(-α‖x − y‖₁) | **α = 0.1** (matches notebook; different from sklearn's default γ = 1/d_features) |
| sobolev | K(x, y) = rᵏ⁻ᵈ/² · K_{k−d/2}(r), r = ‖x − y‖₂ | Bessel-K (Matérn family), **k = 0.5, d = 1** |

### Baselines

Eleven baselines, split into classical and modern/lightweight. All run with a single default configuration (no hyperparameter search) under `scaler = QuantileTransformer`, `poly = -1`. Full hyperparameter details are in the section **Baseline Default Hyperparameters** below.

*FTTransformer was tested but excluded from the final report to match the 12-baseline roster. Raw results remain in `results/baselines_default/runs/`.*
"""


def baseline_params_block() -> str:
    """Dedicated section listing every baseline's full hyperparameter set (set + sklearn defaults we inherit)."""
    return """## Baseline Default Hyperparameters

All baselines are instantiated via `cpai/baselines.py::build_baseline(name)`. Only the parameters listed in the **Set** column below are passed to the constructor; every other knob uses the library default. The **Inherited defaults** column lists the most relevant ones we did **not** override so the table is self-contained.

### Classical baselines (sklearn)

| Model | Set | Inherited defaults (selected) |
|---|---|---|
| **KNN** (`KNeighborsClassifier`) | `n_neighbors=5` | `weights='uniform'`, `algorithm='auto'`, `p=2` (Euclidean), `metric='minkowski'`, `n_jobs=None` |
| **LDA** (`LinearDiscriminantAnalysis`) | `solver='svd'` | `shrinkage=None`, `priors=None`, `n_components=None`, `store_covariance=False`, `tol=1e-4` |
| **NuSVC** (`NuSVC`) | `nu=0.2`, `kernel='rbf'`, `random_state=42` | `degree=3` (unused for rbf), `gamma='scale'`, `coef0=0.0`, `shrinking=True`, `probability=False` (predict_proba falls back to Platt scaling when needed), `tol=1e-3`, `cache_size=200`, `class_weight=None`, `decision_function_shape='ovr'`, `break_ties=False` |
| **SGD** (`SGDClassifier`) | `random_state=42` | `loss='hinge'`, `penalty='l2'`, `alpha=1e-4`, `l1_ratio=0.15`, `fit_intercept=True`, `max_iter=1000`, `tol=1e-3`, `shuffle=True`, `epsilon=0.1`, `learning_rate='optimal'`, `eta0=0.0`, `power_t=0.5`, `early_stopping=False`, `class_weight=None` |
| **GauNB** (`GaussianNB`) | `var_smoothing=1e-9` | `priors=None` |
| **NC** (`NearestCentroid`) | `metric='euclidean'` | `shrink_threshold=None` |
| **RNC** (`RadiusNeighborsClassifier`) | `outlier_label='most_frequent'` | `radius=1.0`, `weights='uniform'`, `algorithm='auto'`, `leaf_size=30`, `p=2`, `metric='minkowski'`, `n_jobs=None` |
| **KNFST** (`cpai.models.KNFST`) | `kernel='rbf'` | Computes centered kernel via `sklearn.preprocessing.KernelCenterer`, dense `np.linalg.eig` for spectral basis, SVD (`eps=1e-12`) for null-space projection, argmin centroid distance at inference. No randomness — results depend only on the train/test split. |

### Modern baselines

| Model | Set | Inherited defaults (selected) |
|---|---|---|
| **LGBM** (`lightgbm.LGBMClassifier`) | `n_estimators=100`, `random_state=42`, `verbosity=-1` | `boosting_type='gbdt'`, `num_leaves=31`, `max_depth=-1` (no limit), `learning_rate=0.1`, `subsample=1.0`, `subsample_freq=0`, `colsample_bytree=1.0`, `reg_alpha=0.0`, `reg_lambda=0.0`, `objective=None` (auto multiclass), `class_weight=None`, `min_child_samples=20`, `n_jobs=-1` |
| **MLP** (`sklearn.neural_network.MLPClassifier`) | `random_state=42` | `hidden_layer_sizes=(100,)` (1 hidden layer), `activation='relu'`, `solver='adam'`, `alpha=1e-4` (L2 reg), `batch_size='auto'`, `learning_rate='constant'`, `learning_rate_init=1e-3`, `max_iter=200`, `shuffle=True`, `tol=1e-4`, `early_stopping=False`, `beta_1=0.9`, `beta_2=0.999`, `epsilon=1e-8`, `n_iter_no_change=10` |
| **TabNet** (`pytorch_tabnet.tab_model.TabNetClassifier` wrapped) | `n_d=16`, `n_a=16`, `n_steps=4`, `seed=42`, `max_epochs=50`, `patience=10`, `batch_size=256`, `virtual_batch_size=128`, `num_workers=0`, `device_name='cpu'` | `gamma=1.3` (attention relaxation), `lambda_sparse=1e-3`, `optimizer_fn=torch.optim.Adam`, `optimizer_params={'lr':0.02}`, `scheduler_fn=None`, `mask_type='sparsemax'`, `momentum=0.02`, `clip_value=None`, `drop_last=False`, `weights=0` (no class reweighting), `loss_fn=cross_entropy` |
| **SAINT** (custom implementation) | `dim=32`, `depth=3`, `heads=4`, `dim_head` (split from `dim` by heads), `dropout=0.1`, `max_epochs=30`, `batch_size=256`, `lr=1e-3`, `seed=42` | Adam optimizer, `betas=(0.9, 0.999)`, cross-entropy loss, `num_workers=0`, CPU, feature-tokenization via learnable (weight, bias) per feature, standard transformer FFN with hidden = 4×dim and GELU activation, LayerNorm before each attention sub-block |

### Notes on SAINT implementation

The `SAINTClassifier` in `cpai/baselines.py` is a from-scratch minimal implementation of Somepalli et al. (2021), not the official codebase. Concretely, each block contains:

1. **Feature-wise multi-head self-attention** over the `(batch, n_features+1, dim)` token tensor (a `[CLS]` token is prepended).
2. **Intersample multi-head self-attention** — the same tensor transposed to `(n_features+1, batch, dim)` so each feature position attends across all samples in the batch (the SAINT novelty).
3. **Position-wise FFN** of shape `(dim → 4·dim → dim)` with GELU + dropout.
4. Residual connections and LayerNorm around each sub-block.

Classification uses the `[CLS]` token after the final block, passed through `LayerNorm → Linear(dim → n_classes)`.

At inference time the intersample attention is computed over *test-set batches of `batch_size`* (no train context is injected), so predictions are deterministic given a fixed batch order and model weights.

### Removed from report

**FTTransformer** (`tab_transformer_pytorch.FTTransformer`) was evaluated (`dim=32`, `depth=3`, `heads=4`, `dim_head=16`, `max_epochs=30`, `batch_size=256`, `lr=1e-3`, `seed=42`) but is excluded from all tables and statistical tests in this report. Its raw JSONs are preserved in `results/baselines_default/runs/*FTTransformer*.json` if needed.
"""


# --------------------------------------------------------------------- metrics

def metrics_block() -> str:
    return """## Evaluation Metrics

All metrics below are computed per run (one per seed) and aggregated as mean ± standard deviation over five seeds.

| Metric | Definition | Notes |
|---|---|---|
| MCC | Matthews Correlation Coefficient | Range [-1, 1]; robust to class imbalance |
| ACC | (TP + TN) / all, aggregated (micro) across classes | |
| TPR macro | mean recall across classes | |
| TPR weighted | recall weighted by class support | |
| PPV macro | mean precision across classes | |
| PPV weighted | precision weighted by class support | |
| F1 macro | mean F1 across classes | Reported as percentage |
| F1 weighted | F1 weighted by class support | |
| FPR | aggregated FP / (FP + TN) across classes | |
| AUROC OvR macro | one-vs-rest multiclass AUROC, unweighted mean | Requires predict_proba |
| AUROC OvR weighted | one-vs-rest multiclass AUROC, support-weighted mean | |
| AUROC OvO macro | one-vs-one multiclass AUROC, unweighted mean | |
| AUROC binary | binary attack-vs-benign AUROC using score = 1 - P(class 0) | Benign is always class 0 (`0Normal`) |
| Train time (s) | wall-clock of `model.fit(X_train, y_train)` only | Excludes preprocessing |
| Test time (s) | wall-clock of `model.predict` + `model.predict_proba` | Excludes preprocessing |
"""


# --------------------------------------------------------------- scene_b loader

def load_scene_b() -> pd.DataFrame:
    # keep_default_na=False so the literal string "None" survives as a kernel name
    df = pd.read_csv(RESULTS_DIR / "scene_b" / "summary.csv", keep_default_na=False)
    # Replace empty strings (real missing values) with NaN where appropriate
    for col in ["MCC", "ACC", "TPR_macro", "TPR_weighted", "PPV_macro", "PPV_weighted",
                "F1_macro", "F1_weighted", "FPR", "AUROC_OvR_macro", "AUROC_OvR_weighted",
                "AUROC_OvO_macro", "AUROC_binary", "t_preprocess_s", "t_fit_s",
                "t_predict_s", "peak_rss_mb", "n_train", "n_test", "n_classes", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["poly"] = df["poly"].astype(int)
    return df


def load_baselines() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "baselines_default" / "summary.csv", keep_default_na=False)
    for col in ["MCC", "ACC", "TPR_macro", "TPR_weighted", "PPV_macro", "PPV_weighted",
                "F1_macro", "F1_weighted", "FPR", "AUROC_OvR_macro", "AUROC_OvR_weighted",
                "AUROC_OvO_macro", "AUROC_binary", "t_preprocess_s", "t_fit_s",
                "t_predict_s", "peak_rss_mb", "n_train", "n_test", "n_classes", "seed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[df["model"].isin(BASELINES)].copy()  # excludes FTTransformer


# --------------------------------------------------------------- kernel study

def _kernel_paper_table(sb: pd.DataFrame, model: str) -> list[str]:
    """Paper Table 2/3 analog: rows = kernels, super-cols = datasets, sub-cols = ACC/TPR/FPR.

    Each cell = "mean (std)" across 5 seeds at the best (scaler, poly) for that
    (dataset, kernel, model) triple.
    """
    metrics = [("ACC", 2), ("TPR_macro", 2), ("FPR", 3)]
    label = "CPAI-OvA (HHH)" if model == "HHH" else "CPAI-OvR (HHHv2)"

    out = [f"### {label} — Multiseed Summary (mean (std) over 5 seeds, best `(scaler, poly)` per cell)",
           ""]
    # Header row 1: dataset names spanning 3 columns each (markdown can't truly span,
    # so we repeat the metric labels with the dataset prefix)
    cols = []
    for ds in DATASETS:
        for m_name, _ in metrics:
            short = "ACC" if m_name == "ACC" else ("TPR" if m_name == "TPR_macro" else "FPR")
            cols.append(f"{ds[:8]}<br/>{short}")
    out.append("| Kernel | " + " | ".join(cols) + " |")
    out.append("|---" + "|---" * len(cols) + "|")

    for k in KERNELS:
        row_cells = [k]
        for ds in DATASETS:
            sub = sb[(sb["dataset"] == ds) & (sb["model"] == model) & (sb["kernel"] == k)]
            if sub.empty:
                row_cells.extend(["—"] * len(metrics))
                continue
            g = sub.groupby(["scaler", "poly"])["MCC"].mean().reset_index()
            best = g.loc[g["MCC"].idxmax()]
            seeds = sub[(sub["scaler"] == best["scaler"]) & (sub["poly"] == best["poly"])]
            for m_name, prec in metrics:
                v = pd.to_numeric(seeds[m_name], errors="coerce").values
                if len(v) == 0:
                    row_cells.append("—")
                else:
                    m, s = mean_std(v)
                    row_cells.append(f"{m:.{prec}f} ({s:.{prec}f})")
        out.append("| " + " | ".join(row_cells) + " |")
    out.append("")
    return out


def kernel_study_tables(sb: pd.DataFrame) -> str:
    """Tables 2/3 analog: best (scaler, poly) per (dataset, model, kernel), mean±std over 5 seeds."""
    parts = ["## Kernel Study — CPAI-OvA (HHH) and CPAI-OvR (HHHv2)",
             "",
             "For each dataset × kernel × model cell, the best `(scaler, poly)` configuration is "
             "selected by mean MCC over five seeds. Values are **mean ± std** across those five seeds.",
             "",
             "### Paper-Format Multiseed Summary (Table 2 / Table 3 analog)",
             "",
             "Mirrors the layout of the original paper's Tables 2 (CPAI-OvA) and 3 (CPAI-OvR): "
             "rows are kernels, super-columns are datasets, with `ACC` (%), `TPR macro` (%), and "
             "`FPR` (%) sub-columns per dataset. Each cell is `mean (std)` across 5 seeds; the best "
             "`(scaler, poly)` per `(dataset, kernel)` cell is selected by mean MCC.",
             ""]
    parts.extend(_kernel_paper_table(sb, "HHH"))
    parts.extend(_kernel_paper_table(sb, "HHHv2"))
    parts.append("### Per-Metric Detail Tables")
    parts.append("")
    parts.append("Same data, broken out one metric at a time so each table fits a normal viewport.")
    parts.append("")

    for model in ["HHH", "HHHv2"]:
        parts.append(f"### {'CPAI-OvA (HHH)' if model == 'HHH' else 'CPAI-OvR (HHHv2)'}")
        parts.append("")
        parts.append(f"#### Accuracy (ACC %) — best `(scaler, poly)` per cell")
        parts.append("")
        parts.append("| Kernel | " + " | ".join(DATASETS) + " |")
        parts.append("|---|" + "|".join(["---"] * len(DATASETS)) + "|")
        for k in KERNELS:
            row = [k]
            for ds in DATASETS:
                sub = sb[(sb["dataset"] == ds) & (sb["model"] == model) & (sb["kernel"] == k)]
                if sub.empty:
                    row.append("—")
                    continue
                # best (scaler, poly) by mean MCC
                g = sub.groupby(["scaler", "poly"])["MCC"].mean().reset_index()
                best_cfg = g.loc[g["MCC"].idxmax()]
                seeds = sub[(sub["scaler"] == best_cfg["scaler"]) & (sub["poly"] == best_cfg["poly"])]
                m, s = mean_std(seeds["ACC"].values)
                row.append(f"{m:.2f}±{s:.2f}")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

        parts.append(f"#### Matthews Correlation Coefficient (MCC)")
        parts.append("")
        parts.append("| Kernel | " + " | ".join(DATASETS) + " |")
        parts.append("|---|" + "|".join(["---"] * len(DATASETS)) + "|")
        for k in KERNELS:
            row = [k]
            for ds in DATASETS:
                sub = sb[(sb["dataset"] == ds) & (sb["model"] == model) & (sb["kernel"] == k)]
                if sub.empty:
                    row.append("—")
                    continue
                g = sub.groupby(["scaler", "poly"])["MCC"].mean().reset_index()
                best_cfg = g.loc[g["MCC"].idxmax()]
                seeds = sub[(sub["scaler"] == best_cfg["scaler"]) & (sub["poly"] == best_cfg["poly"])]
                m, s = mean_std(seeds["MCC"].values)
                row.append(f"{m:.4f}±{s:.4f}")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

        parts.append(f"#### F1 macro (%)")
        parts.append("")
        parts.append("| Kernel | " + " | ".join(DATASETS) + " |")
        parts.append("|---|" + "|".join(["---"] * len(DATASETS)) + "|")
        for k in KERNELS:
            row = [k]
            for ds in DATASETS:
                sub = sb[(sb["dataset"] == ds) & (sb["model"] == model) & (sb["kernel"] == k)]
                if sub.empty:
                    row.append("—")
                    continue
                g = sub.groupby(["scaler", "poly"])["MCC"].mean().reset_index()
                best_cfg = g.loc[g["MCC"].idxmax()]
                seeds = sub[(sub["scaler"] == best_cfg["scaler"]) & (sub["poly"] == best_cfg["poly"])]
                m, s = mean_std(seeds["F1_macro"].values)
                row.append(f"{m:.2f}±{s:.2f}")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

        parts.append(f"#### TPR macro (%)")
        parts.append("")
        parts.append("| Kernel | " + " | ".join(DATASETS) + " |")
        parts.append("|---|" + "|".join(["---"] * len(DATASETS)) + "|")
        for k in KERNELS:
            row = [k]
            for ds in DATASETS:
                sub = sb[(sb["dataset"] == ds) & (sb["model"] == model) & (sb["kernel"] == k)]
                if sub.empty:
                    row.append("—")
                    continue
                g = sub.groupby(["scaler", "poly"])["MCC"].mean().reset_index()
                best_cfg = g.loc[g["MCC"].idxmax()]
                seeds = sub[(sub["scaler"] == best_cfg["scaler"]) & (sub["poly"] == best_cfg["poly"])]
                m, s = mean_std(seeds["TPR_macro"].values)
                row.append(f"{m:.2f}±{s:.2f}")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

        parts.append(f"#### FPR (%)")
        parts.append("")
        parts.append("| Kernel | " + " | ".join(DATASETS) + " |")
        parts.append("|---|" + "|".join(["---"] * len(DATASETS)) + "|")
        for k in KERNELS:
            row = [k]
            for ds in DATASETS:
                sub = sb[(sb["dataset"] == ds) & (sb["model"] == model) & (sb["kernel"] == k)]
                if sub.empty:
                    row.append("—")
                    continue
                g = sub.groupby(["scaler", "poly"])["MCC"].mean().reset_index()
                best_cfg = g.loc[g["MCC"].idxmax()]
                seeds = sub[(sub["scaler"] == best_cfg["scaler"]) & (sub["poly"] == best_cfg["poly"])]
                m, s = mean_std(seeds["FPR"].values)
                row.append(f"{m:.3f}±{s:.3f}")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

        # Best config table
        parts.append(f"#### Best `(scaler, poly)` picked per cell")
        parts.append("")
        parts.append("| Kernel | " + " | ".join(DATASETS) + " |")
        parts.append("|---|" + "|".join(["---"] * len(DATASETS)) + "|")
        for k in KERNELS:
            row = [k]
            for ds in DATASETS:
                sub = sb[(sb["dataset"] == ds) & (sb["model"] == model) & (sb["kernel"] == k)]
                if sub.empty:
                    row.append("—")
                    continue
                g = sub.groupby(["scaler", "poly"])["MCC"].mean().reset_index()
                best_cfg = g.loc[g["MCC"].idxmax()]
                row.append(f"({best_cfg['scaler'][:4]}, p={best_cfg['poly']})")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------- baseline comparison

def _cpai_best_5seed_values(sb: pd.DataFrame, ds: str, model: str, metric: str = "MCC") -> tuple[np.ndarray, dict]:
    """Pick best (kernel, scaler, poly) by mean MCC over 5 seeds. Return the 5 per-seed values
    for `metric` plus the chosen config dict."""
    sub = sb[(sb["dataset"] == ds) & (sb["model"] == model)]
    g = sub.groupby(["kernel", "scaler", "poly"])["MCC"].mean().reset_index()
    best = g.loc[g["MCC"].idxmax()]
    seeds = sub[(sub["kernel"] == best["kernel"]) & (sub["scaler"] == best["scaler"]) & (sub["poly"] == best["poly"])]
    cfg = {"kernel": best["kernel"], "scaler": best["scaler"], "poly": int(best["poly"])}
    return seeds.sort_values("seed")[metric].values, cfg


def baseline_comparison_tables(sb: pd.DataFrame, bl: pd.DataFrame) -> str:
    parts = [
        "## Baseline Comparison — Per-Dataset Tables with Statistical Tests",
        "",
        "For every dataset we show mean ± std across five seeds of the following metrics:",
        "**MCC, ACC (%), TPR macro (%), FPR (%), PPV macro (%), F1 macro (%)**. "
        "Both CPAI models use the configuration (kernel, scaler, poly) with the highest mean MCC; "
        "baselines use the fixed default configuration (QuantileTransformer, poly = -1).",
        "",
        "**Statistical tests**: for every `(CPAI model, baseline)` pair we run a paired test across the "
        "five seeds comparing the MCC values and report two p-values — paired t-test (`p_t`) and Wilcoxon "
        "signed-rank (`p_w`). Lower p-value ⇒ stronger evidence that CPAI differs from the baseline. "
        "`p<0.05` is commonly used as the significance threshold. When CPAI beats the baseline by a "
        "non-zero margin for every seed the Wilcoxon test returns `p≈0.0625` (the lowest achievable "
        "p-value with n = 5 paired samples).",
        "",
    ]

    # For the summary table we collect each dataset's row.
    summary_rows = []

    for ds in DATASETS:
        parts.append(f"### {ds}")
        parts.append("")

        # Build rows: baselines then HHH best then HHHv2 best
        rows_data = {}
        for model in BASELINES:
            sub = bl[(bl["dataset"] == ds) & (bl["model"] == model)].sort_values("seed")
            if len(sub) != 5:
                continue
            rows_data[model] = sub[[c for c, _, _ in METRICS_CORE]].values  # (5, 6)

        hhh_seeds_mcc, hhh_cfg = _cpai_best_5seed_values(sb, ds, "HHH", "MCC")
        hhh_all = {c: _cpai_best_5seed_values(sb, ds, "HHH", c)[0] for c, _, _ in METRICS_CORE}
        hhhv2_seeds_mcc, hhhv2_cfg = _cpai_best_5seed_values(sb, ds, "HHHv2", "MCC")
        hhhv2_all = {c: _cpai_best_5seed_values(sb, ds, "HHHv2", c)[0] for c, _, _ in METRICS_CORE}

        # Header row
        parts.append("| Model | " + " | ".join([disp for _, disp, _ in METRICS_CORE]) +
                     " | p_t vs OvA | p_w vs OvA | p_t vs OvR | p_w vs OvR |")
        parts.append("|---|" + "|".join(["---"] * (len(METRICS_CORE) + 4)) + "|")

        for model, arr in rows_data.items():
            base_mcc = arr[:, 0]  # MCC column

            # Paired tests HHH vs baseline
            if np.allclose(hhh_seeds_mcc, base_mcc):
                p_t_ova = 1.0
                p_w_ova = 1.0
            else:
                p_t_ova = ttest_rel(hhh_seeds_mcc, base_mcc, nan_policy="omit")[1]
                try:
                    p_w_ova = wilcoxon(hhh_seeds_mcc, base_mcc, zero_method="wilcox", alternative="two-sided").pvalue
                except ValueError:
                    p_w_ova = float("nan")

            if np.allclose(hhhv2_seeds_mcc, base_mcc):
                p_t_ovr = 1.0
                p_w_ovr = 1.0
            else:
                p_t_ovr = ttest_rel(hhhv2_seeds_mcc, base_mcc, nan_policy="omit")[1]
                try:
                    p_w_ovr = wilcoxon(hhhv2_seeds_mcc, base_mcc, zero_method="wilcox", alternative="two-sided").pvalue
                except ValueError:
                    p_w_ovr = float("nan")

            cells = []
            for i, (c, _, prec) in enumerate(METRICS_CORE):
                m, s = mean_std(arr[:, i])
                cells.append(fmt(m, s, prec))
            row = [model] + cells + [f"{p_t_ova:.3f}", f"{p_w_ova:.3f}", f"{p_t_ovr:.3f}", f"{p_w_ovr:.3f}"]
            parts.append("| " + " | ".join(row) + " |")

        # HHH row
        cells = [fmt(*mean_std(hhh_all[c]), prec) for c, _, prec in METRICS_CORE]
        parts.append(f"| **CPAI-OvA (HHH)** — {hhh_cfg['kernel']}, {hhh_cfg['scaler']}, poly={hhh_cfg['poly']} | "
                     + " | ".join(cells) + " | — | — | — | — |")
        # HHHv2 row
        cells = [fmt(*mean_std(hhhv2_all[c]), prec) for c, _, prec in METRICS_CORE]
        parts.append(f"| **CPAI-OvR (HHHv2)** — {hhhv2_cfg['kernel']}, {hhhv2_cfg['scaler']}, poly={hhhv2_cfg['poly']} | "
                     + " | ".join(cells) + " | — | — | — | — |")
        parts.append("")

        # Collect for summary: per model mean-of-mean over this dataset
        for model in BASELINES:
            if model in rows_data:
                arr = rows_data[model]
                summary_rows.append(("baseline", model, ds,
                                     *[np.mean(arr[:, i]) for i, _ in enumerate(METRICS_CORE)]))
        summary_rows.append(("cpai", "HHH", ds,
                             *[np.mean(hhh_all[c]) for c, _, _ in METRICS_CORE]))
        summary_rows.append(("cpai", "HHHv2", ds,
                             *[np.mean(hhhv2_all[c]) for c, _, _ in METRICS_CORE]))

    # ---------------- Summary table (average of 7 datasets) ----------------
    parts.append("### Summary — Average Across All 7 Datasets")
    parts.append("")
    parts.append("Each cell is the unweighted mean of the per-dataset 5-seed mean. Significance tests are "
                 "performed on a paired array of *per-dataset means* (n = 7 paired samples) for CPAI vs "
                 "each baseline.")
    parts.append("")

    df_s = pd.DataFrame(summary_rows, columns=["kind", "model", "dataset"] + [c for c, _, _ in METRICS_CORE])
    # Average by model
    metrics = [c for c, _, _ in METRICS_CORE]
    avg = df_s.groupby(["model"])[metrics].mean().reset_index()

    # Compute paired tests across the 7 datasets (n=7)
    cpai_hhh_mcc = df_s[(df_s["model"] == "HHH")].sort_values("dataset")["MCC"].values
    cpai_hhhv2_mcc = df_s[(df_s["model"] == "HHHv2")].sort_values("dataset")["MCC"].values

    parts.append("| Model | " + " | ".join([d for _, d, _ in METRICS_CORE]) +
                 " | p_t vs OvA | p_w vs OvA | p_t vs OvR | p_w vs OvR |")
    parts.append("|---|" + "|".join(["---"] * (len(METRICS_CORE) + 4)) + "|")

    for model in BASELINES + ["HHH", "HHHv2"]:
        row = avg[avg["model"] == model]
        if row.empty:
            continue
        cells = []
        for c, _, prec in METRICS_CORE:
            cells.append(f"{row[c].values[0]:.{prec}f}")
        if model in BASELINES:
            base_mcc = df_s[df_s["model"] == model].sort_values("dataset")["MCC"].values
            if len(base_mcc) != 7:
                p_t_ova = p_w_ova = p_t_ovr = p_w_ovr = float("nan")
            else:
                p_t_ova = ttest_rel(cpai_hhh_mcc, base_mcc).pvalue
                p_t_ovr = ttest_rel(cpai_hhhv2_mcc, base_mcc).pvalue
                try:
                    p_w_ova = wilcoxon(cpai_hhh_mcc, base_mcc).pvalue
                except ValueError:
                    p_w_ova = float("nan")
                try:
                    p_w_ovr = wilcoxon(cpai_hhhv2_mcc, base_mcc).pvalue
                except ValueError:
                    p_w_ovr = float("nan")
            pvals = [f"{p_t_ova:.3f}", f"{p_w_ova:.3f}", f"{p_t_ovr:.3f}", f"{p_w_ovr:.3f}"]
            parts.append("| " + " | ".join([model] + cells + pvals) + " |")
        else:
            label = "**CPAI-OvA (HHH) — best-per-ds**" if model == "HHH" else "**CPAI-OvR (HHHv2) — best-per-ds**"
            parts.append("| " + " | ".join([label] + cells + ["—", "—", "—", "—"]) + " |")

    parts.append("")
    return "\n".join(parts)


# --------------------------------------------------------------------- timing

def timing_block(sb: pd.DataFrame, bl: pd.DataFrame) -> str:
    parts = [
        "## Average Train / Test Time (Best Config)",
        "",
        "Train time = wall-clock of `model.fit(X_train, y_train)` only. "
        "Test time = wall-clock of `model.predict` + `model.predict_proba`. "
        "Preprocessing time is **excluded**.",
        "",
        "Baselines are the 5-seed mean from `results/baselines_default/` (default "
        "scaler = QuantileTransformer, poly = -1). CPAI-OvA and CPAI-OvR rows are "
        "from `results/rerun_best/` at seed = 42 after the in-place kernel + "
        "`overwrite_a=True` solver optimizations (see **Performance Optimization "
        "Note** below) — MCC is bit-exact with the original 5-seed Scene B runs, "
        "so these timings are directly substitutable. Both were collected with "
        "`multiprocessing.Pool(processes=2)`.",
        "",
        "| Model | " + " | ".join(DATASETS) + " |",
        "|---|" + "|".join(["---"] * len(DATASETS)) + "|",
    ]

    def _fmt(t_fit, t_pred):
        return f"fit {t_fit:.2f}s / pred {t_pred:.3f}s"

    for model in BASELINES:
        row = [model]
        for ds in DATASETS:
            sub = bl[(bl["dataset"] == ds) & (bl["model"] == model)]
            if len(sub) != 5:
                row.append("—")
                continue
            row.append(_fmt(sub["t_fit_s"].mean(), sub["t_predict_s"].mean()))
        parts.append("| " + " | ".join(row) + " |")

    # CPAI rows: use rerun_best JSONs (post-memopt, seed=42, Pool(2), bit-exact MCC)
    import json
    runs_dir = RESULTS_DIR / "rerun_best"
    for model in ["HHH", "HHHv2"]:
        row = ["CPAI-OvA" if model == "HHH" else "CPAI-OvR"]
        for ds in DATASETS:
            jf = runs_dir / f"{ds}__{model}.json"
            if not jf.exists():
                row.append("—")
                continue
            d = json.loads(jf.read_text())
            row.append(_fmt(d["_t_fit"], d["_t_predict"]))
        parts.append("| " + " | ".join(row) + " |")

    parts.append("")
    parts.append("### Performance Optimization Note")
    parts.append("")
    parts.append("The CPAI implementation was tuned for memory and runtime after the initial Scene B sweep:")
    parts.append("")
    parts.append("1. **In-place kernel exp**: `abel_kernel`, `laplacian_kernel` now compute `cdist` then reuse the same buffer for `np.exp(-α·d, out=D)` — saves one n×n float64 allocation.")
    parts.append("2. **In-place ridge**: `A.reshape(-1)[::n+1] += ridge` instead of `A + np.eye(n) * ridge` — saves another n×n allocation.")
    parts.append("3. **`scipy.linalg.solve(..., overwrite_a=True)`**: LAPACK factors the regularized matrix destructively rather than copying it internally — saves an n×n and is faster because there's less memory traffic.")
    parts.append("")
    parts.append("The CPAI rows above reflect post-optimization timings. Bit-exact MCC preservation was verified on all 14 paper-validation runs (ΔMCC < 1.11×10⁻¹⁶ = float64 rounding noise on 4 runs, ΔMCC = 0 exactly on the other 10). Total fit-time on the 14 best-configs dropped from **86.8 s → 32.8 s (2.6×)**; per-run peak RSS dropped by ~**2.3×** (see the updated Resource Consumption table). Sum of 14 fits at different worker counts (post-opt, Pool-based):")
    parts.append("")
    parts.append("| Workers | Wall (s) | Sum fit (s) | Max peak / worker | Aggregate peak |")
    parts.append("|---|---|---|---|---|")
    parts.append("| 1 (serial) | 44.4 | 27.5 | 3 773 MB | 3.8 GB |")
    parts.append("| 2 | 28.4 | 34.7 | 3 755 MB | ~7.5 GB |")
    parts.append("| 4 | 19.7 | 44.8 | 3 664 MB | ~14.7 GB |")
    parts.append("")
    return "\n".join(parts)


# ---------------------------------------------------------- memory / peak RSS

def memory_block(sb: pd.DataFrame) -> str:
    """Peak RSS for CPAI-OvA / CPAI-OvR at their best-per-dataset config.

    Shows three per-dataset values:
      - Pre-optimization peak  (5-seed mean from Scene B summary.csv)
      - Post-optimization peak (rerun_best seed=42 under Pool(2), same context)
      - Fresh-subprocess peak  (profile_memory_v2: single serial run, no HWM accumulation)
    plus a cross-dataset summary row.
    """
    import json
    runs_dir = RESULTS_DIR / "rerun_best"
    parts = [
        "## Resource Consumption — CPAI-OvA (HHH) and CPAI-OvR (HHHv2)",
        "",
        "Peak resident set size (RSS) in **MB**, captured per-run via "
        "`resource.getrusage(RUSAGE_SELF).ru_maxrss`. Three measurements per (dataset, model):",
        "",
        "- **Pre-opt (Scene B)**: 5-seed mean from the original Scene B sweep "
        "(`multiprocessing.Pool(processes=2)`, pre in-place kernel / ridge / overwrite_a).",
        "- **Post-opt (rerun_best)**: seed = 42 rerun after the optimizations, same Pool(2) "
        "context. Bit-exact MCC — this is pure memory savings.",
        "- **Fresh subprocess**: single top-level Python invocation (no prior memory peak in the "
        "same process). This is the cleanest per-run peak; the Pool numbers are higher because "
        "each worker accumulates a high-water-mark across all the jobs it processed.",
        "",
    ]

    # Fresh-subprocess peaks from profile_memory_v2.py measurements (saved inline).
    fresh_peaks = {
        ("Edge_IIoTset", "HHHv2"): 2785,
        ("Edge_IIoTset", "HHH"):   2469,
        ("N_BaIoT",      "HHHv2"): 1292,
        ("N_BaIoT",      "HHH"):   1296,
        ("BoT_IoT",      "HHHv2"): 1461,
        ("5G_NIDD",      "HHHv2"): 1233,
    }

    for model_name, model_label in [("HHH", "CPAI-OvA (HHH)"), ("HHHv2", "CPAI-OvR (HHHv2)")]:
        parts.append(f"### {model_label}")
        parts.append("")
        parts.append("| Dataset | Config | n_train | Pre-opt Pool(2) | Post-opt Pool(2) | Fresh subprocess | Reduction (Pre/Post) |")
        parts.append("|---|---|---|---|---|---|---|")
        pre_vals, post_vals = [], []
        for ds in DATASETS:
            sub = sb[(sb["dataset"] == ds) & (sb["model"] == model_name)]
            agg = sub.groupby(["kernel", "scaler", "poly"])["MCC"].mean().reset_index()
            best = agg.loc[agg["MCC"].idxmax()]
            seeds = sub[(sub["kernel"] == best["kernel"]) &
                        (sub["scaler"] == best["scaler"]) &
                        (sub["poly"] == best["poly"])]
            pre_peak = seeds["peak_rss_mb"].mean()
            n_train = int(seeds["n_train"].iloc[0])
            cfg = f"{best['kernel']}, {best['scaler']}, poly={int(best['poly'])}"

            jf = runs_dir / f"{ds}__{model_name}.json"
            post_peak = json.loads(jf.read_text())["_peak_rss_mb"] if jf.exists() else None

            fresh = fresh_peaks.get((ds, model_name), None)
            fresh_str = f"{fresh:.0f}" if fresh is not None else "—"
            post_str = f"{post_peak:.0f}" if post_peak is not None else "—"
            ratio = pre_peak / post_peak if post_peak else float("nan")
            parts.append(f"| {ds} | {cfg} | {n_train:,} | {pre_peak:.0f} | {post_str} | {fresh_str} | {ratio:.2f}× |")
            pre_vals.append(pre_peak)
            if post_peak:
                post_vals.append(post_peak)
        pre_arr = np.array(pre_vals); post_arr = np.array(post_vals)
        parts.append(f"| **Mean across datasets** | — | — | {pre_arr.mean():.0f} | "
                     f"{post_arr.mean():.0f} | — | {pre_arr.mean()/post_arr.mean():.2f}× |")
        parts.append("")

    parts.append("*The post-opt Pool(2) numbers still show some cumulative high-water-mark* "
                 "*inflation when a worker inherits a peak from a previous job — see 5G_NIDD's* "
                 "*~3.7 GB post-opt Pool peak vs ~1.2 GB fresh-subprocess peak for the same job.*")
    parts.append("")
    return "\n".join(parts)


# ------------------------------------------------------------------- figures

def figures_block() -> str:
    """Reference all four generated figures with interpretation."""
    return """## Figures

All figures are generated by `code/make_figures.py` from `results/scene_b/summary.csv`
(and `results/scene_b/runs/*.json` for the ROC `y_score` data).

### Figure 4 — Mean MCC by kernel, CPAI-OvA vs CPAI-OvR

![MCC bars](paper/figures/MeanMCC_kernels_2_approach_v3.png)

Bars are the mean MCC across the **7 datasets × 5 seeds** (per (dataset, kernel, model) cell we pick the best `(scaler, poly)` by mean MCC, then average across datasets). Error bars show the cross-dataset standard deviation. Observations:

- **CPAI-OvR dominates CPAI-OvA on every kernel.** The gap is widest on non-positive-definite kernels (None, linear, sigmoid): +0.37, +0.39, +0.18 MCC — there the OvA single-scalar discriminant collapses while OvR's one-discriminant-per-class framework still separates the majority of classes.
- **Laplacian is the top kernel** on both models: MCC = 0.91 (OvA) / **0.97 (OvR)**. Abel is a close second; RBF is third.
- Sobolev and sigmoid show the highest cross-dataset std (error bars up to ±0.22), indicating unstable behavior.

### Figure 5 — Mean F1 macro by kernel, CPAI-OvA vs CPAI-OvR

![F1 bars](paper/figures/MeanF1_kernels_2_approach_v3.png)

Same setup as Figure 4 but F1 macro (%) on the y-axis. The F1 gap between OvA and OvR is even more pronounced than MCC (OvR beats OvA by 9.2–56.0 F1 points across kernels) — OvA's "one-class-bombed" failure mode shows up heavily in macro-averaged F1.

### Figure 6 — Binary ROC (attack vs benign): CPAI-OvR vs modern DL baselines

![ROC vs baselines](paper/figures/binary_roc_by_dataset.png)

For each dataset we take CPAI-OvR at its best `(kernel, scaler, poly)` config (from Scene B), LGBM at its default config, and SAINT / TabNet at their default configs — all at seed = 42. Binary attack-vs-benign label: `y_bin = (y_true != 0)`. Attack score: `1 − P(class 0)`. **FPR is on log scale** to emphasize the low-false-positive regime where IDS deployment actually lives (full-linear ROC is saturated at TPR ≈ 1 for every model on every dataset).

Observations:

- **BoT_IoT, N_BaIoT, CIC_IoT2023, 5G_NIDD**: all four models reach TPR = 1 at FPR < 10⁻⁴ — the binary detection problem is trivially separable on these datasets.
- **IoTID20, ToN_IoT, Edge_IIoTset**: real separation emerges in the low-FPR region. CPAI-OvR (orange) Pareto-dominates the DL baselines at FPR ≤ 10⁻² on IoTID20; on Edge_IIoTset the curves are closer but CPAI-OvR maintains ≥5 percentage-point TPR margin from FPR = 10⁻³ onward.
- **SAINT** and **TabNet** are comparable; LGBM is the strongest classical competitor.

### Figure 7 — Binary ROC: CPAI-OvA vs CPAI-OvR only

![ROC CPAI only](paper/figures/binary_roc_cpai_only.png)

Head-to-head between our two proposed models. Each subplot caption lists the best-per-model config for that dataset. The pattern mirrors Figure 5: on easy datasets (BoT_IoT, N_BaIoT, CIC_IoT2023, 5G_NIDD) both models saturate to perfect binary detection; on the three harder datasets (IoTID20, ToN_IoT, Edge_IIoTset) CPAI-OvR reaches high TPR at consistently lower FPR than CPAI-OvA. **Edge_IIoTset shows the largest gap**: CPAI-OvR reaches TPR ≈ 0.9 at FPR = 10⁻² while CPAI-OvA needs FPR ≈ 10⁻¹.
"""


# -------------------------------------------------------------- ablation: ridge

def ablation_ridge_block() -> str:
    abl_csv = RESULTS_DIR / "ablation_ridge" / "summary.csv"
    if not abl_csv.exists():
        return ""
    df = pd.read_csv(abl_csv)
    df["ridge"] = df["ridge"].astype(float)
    ridges = sorted(df["ridge"].unique())
    ridge_cols = [f"λ={r:g}" for r in ridges]

    parts = [
        "## Ablation Study — Ridge Regularization λ",
        "",
        "For each dataset the best `(kernel, scaler, poly)` configuration from the Scene B "
        "multiseed sweep is held fixed, and only the ridge λ is varied across five values: "
        "`{1e-9, 1e-7, 1e-5, 1e-3, 1e-1}`. Single seed (`seed = 42`) per configuration — "
        "70 runs total.",
        "",
        "**Default ridge values used elsewhere in this report**: CPAI-OvA (HHH) uses λ = 1e-9 for "
        "the kernel path and λ = 1e-5 for the linear/None path; CPAI-OvR (HHHv2) uses λ = 1e-5 "
        "on both paths.",
        "",
    ]

    for model_name, model_label in [("HHH", "CPAI-OvA (HHH)"), ("HHHv2", "CPAI-OvR (HHHv2)")]:
        sub = df[df["model"] == model_name]

        # Best config used per dataset
        cfg_rows = sub.drop_duplicates("dataset")[["dataset", "kernel", "scaler", "poly"]]
        cfg_map = {r["dataset"]: f"{r['kernel']}, {r['scaler']}, poly={r['poly']}" for _, r in cfg_rows.iterrows()}

        parts.append(f"### {model_label}")
        parts.append("")
        parts.append(f"#### MCC as a function of λ")
        parts.append("")
        parts.append("| Dataset | Config (kernel, scaler, poly) | " + " | ".join(ridge_cols) + " |")
        parts.append("|---|---|" + "|".join(["---"] * len(ridges)) + "|")
        for ds in DATASETS:
            row = [ds, cfg_map.get(ds, "—")]
            for r in ridges:
                v = sub[(sub["dataset"] == ds) & (np.isclose(sub["ridge"], r))]["MCC"]
                row.append(f"{v.values[0]:.4f}" if len(v) else "—")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

        parts.append(f"#### F1 macro (%) as a function of λ")
        parts.append("")
        parts.append("| Dataset | " + " | ".join(ridge_cols) + " |")
        parts.append("|---|" + "|".join(["---"] * len(ridges)) + "|")
        for ds in DATASETS:
            row = [ds]
            for r in ridges:
                v = sub[(sub["dataset"] == ds) & (np.isclose(sub["ridge"], r))]["F1_macro"]
                row.append(f"{v.values[0]:.2f}" if len(v) else "—")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

    # Interpretation
    parts.append("### Observations")
    parts.append("")
    parts.append("- **CPAI-OvR (HHHv2)** is extremely robust — λ in `[1e-9, 1e-3]` yields essentially identical MCC on most datasets (≤ 0.005 variation). Performance only degrades noticeably at `λ = 1e-1`, and even there the drop is modest (≤ 0.05 MCC on most datasets).")
    parts.append("- **CPAI-OvA (HHH)** is more sensitive to λ. On `IoTID20`, `N_BaIoT`, and `Edge_IIoTset` MCC drops by 0.05–0.15 as λ grows from 1e-5 to 1e-1. The paper's default (`λ = 1e-9` on the kernel path) lies in the safe flat region of the curve on every dataset.")
    parts.append("- The conditioning of the kernel Gram matrix `K` dominates the λ sensitivity: `N_BaIoT` and `Edge_IIoTset` have larger training sets (`n_train ≈ 7k–12k`) and more features, producing a worse-conditioned `K` — the extra regularization at `λ = 1e-1` over-smooths and costs MCC.")
    parts.append("")
    return "\n".join(parts)


# ----------------------------------------------------------- ablation: gamma

def ablation_gamma_block() -> str:
    abl_csv = RESULTS_DIR / "ablation_gamma" / "summary.csv"
    if not abl_csv.exists():
        return ""
    df = pd.read_csv(abl_csv)
    df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce")
    df["F1_macro"] = pd.to_numeric(df["F1_macro"], errors="coerce")
    df["gamma_used"] = pd.to_numeric(df["gamma_used"], errors="coerce")

    parts = [
        "## Ablation Study — Bandwidth γ (RBF / Poly / Sigmoid)",
        "",
        "Section 3.4 of the revised paper proposes a data-driven bandwidth heuristic:",
        "",
        "> σ = median over training samples of the mean distance to its *k* nearest neighbors;  \n"
        "> γ = 1 / (2 σ²)",
        "",
        "We implement this with **k = 5** (matching the KNN baseline). For each "
        "(dataset, model, kernel) triple we freeze the best `(scaler, poly)` "
        "configuration from Scene B and vary only γ between two settings:",
        "",
        "- **γ = 1/n_features** — sklearn's default (used throughout Scene B).",
        "- **γ = 1/(2σ²)** — the proposed heuristic, computed per-dataset from X_train.",
        "",
        "Single seed (`seed = 42`), 7 datasets × 2 models × 3 kernels × 2 γ settings = 84 runs.",
        "",
    ]

    # ---------------- Condensed MCC overview ----------------
    for model_name, model_label in [("HHH", "CPAI-OvA (HHH)"), ("HHHv2", "CPAI-OvR (HHHv2)")]:
        parts.append(f"### MCC overview — {model_label}")
        parts.append("")
        parts.append("| Dataset | rbf default | rbf heuristic | poly default | poly heuristic | sigmoid default | sigmoid heuristic |")
        parts.append("|---|---|---|---|---|---|---|")
        for ds in DATASETS:
            row = [ds]
            for k in ["rbf", "poly", "sigmoid"]:
                for gs in ["default", "heuristic"]:
                    r = df[(df.dataset == ds) & (df.model == model_name) &
                           (df.kernel == k) & (df.gamma_setting == gs)]
                    if r.empty or pd.isna(r["MCC"].iloc[0]):
                        row.append("—")
                    else:
                        row.append(f"{r['MCC'].iloc[0]:.4f}")
            parts.append("| " + " | ".join(row) + " |")
        parts.append("")

    # ---------------- Per-kernel detailed comparison ----------------
    parts.append("### Per-Kernel Detailed Comparison")
    parts.append("")
    parts.append("Each row is one `(dataset, model)` pair. Columns give MCC, ACC (%), F1 macro (%), "
                 "TPR macro (%), and FPR (%) under both γ settings, plus the signed delta (heuristic − default) for MCC and F1.")
    parts.append("")

    all_metrics = [("MCC", 4), ("ACC", 2), ("F1_macro", 2), ("TPR_macro", 2), ("FPR", 3)]

    for kernel in ["rbf", "poly", "sigmoid"]:
        parts.append(f"#### Kernel: {kernel}")
        parts.append("")
        header = (
            "| Dataset | Model | "
            + " | ".join([f"{nm} def" for nm, _ in all_metrics])
            + " | "
            + " | ".join([f"{nm} heur" for nm, _ in all_metrics])
            + " | ΔMCC | ΔF1 |"
        )
        parts.append(header)
        parts.append("|---|---|" + "|".join(["---"] * (2 * len(all_metrics) + 2)) + "|")

        for ds in DATASETS:
            for model in ["HHH", "HHHv2"]:
                r_def = df[(df.dataset == ds) & (df.model == model) &
                           (df.kernel == kernel) & (df.gamma_setting == "default")]
                r_heu = df[(df.dataset == ds) & (df.model == model) &
                           (df.kernel == kernel) & (df.gamma_setting == "heuristic")]
                if r_def.empty or r_heu.empty:
                    continue
                def _fmtval(row, key, prec):
                    v = pd.to_numeric(row[key], errors="coerce").iloc[0]
                    return f"{v:.{prec}f}" if pd.notna(v) else "—"
                cells_def = [_fmtval(r_def, m, p) for m, p in all_metrics]
                cells_heu = [_fmtval(r_heu, m, p) for m, p in all_metrics]
                dmcc = (pd.to_numeric(r_heu["MCC"]).iloc[0] - pd.to_numeric(r_def["MCC"]).iloc[0])
                df1 = (pd.to_numeric(r_heu["F1_macro"]).iloc[0] - pd.to_numeric(r_def["F1_macro"]).iloc[0])
                mdisp = "OvA" if model == "HHH" else "OvR"
                parts.append(f"| {ds} | {mdisp} | " + " | ".join(cells_def)
                             + " | " + " | ".join(cells_heu)
                             + f" | {dmcc:+.4f} | {df1:+.2f} |")
        parts.append("")

    # ---------------- Win / loss summary ----------------
    parts.append("### Win / Loss Summary (ΔMCC per cell)")
    parts.append("")
    parts.append("Count of datasets where the heuristic wins (> default), ties (|Δ| < 0.001), or loses.")
    parts.append("")
    parts.append("| Kernel | Model | Wins (heur > def) | Ties | Losses (def > heur) |")
    parts.append("|---|---|---|---|---|")
    for kernel in ["rbf", "poly", "sigmoid"]:
        for model in ["HHH", "HHHv2"]:
            wins = ties = losses = 0
            for ds in DATASETS:
                r_def = df[(df.dataset == ds) & (df.model == model) &
                           (df.kernel == kernel) & (df.gamma_setting == "default")]
                r_heu = df[(df.dataset == ds) & (df.model == model) &
                           (df.kernel == kernel) & (df.gamma_setting == "heuristic")]
                if r_def.empty or r_heu.empty:
                    continue
                d = pd.to_numeric(r_heu["MCC"]).iloc[0] - pd.to_numeric(r_def["MCC"]).iloc[0]
                if abs(d) < 0.001:
                    ties += 1
                elif d > 0:
                    wins += 1
                else:
                    losses += 1
            mdisp = "CPAI-OvA" if model == "HHH" else "CPAI-OvR"
            parts.append(f"| {kernel} | {mdisp} | {wins} | {ties} | {losses} |")
    parts.append("")

    # γ scale comparison
    parts.append("### γ Values Produced by the Heuristic vs sklearn Default")
    parts.append("")
    parts.append("The heuristic γ is computed once per dataset (same X_train shape across models). "
                 "The ratio column is `γ_heuristic / γ_default`.")
    parts.append("")
    parts.append("| Dataset | n_features | γ_default (=1/n) | γ_heuristic | ratio |")
    parts.append("|---|---|---|---|---|")
    for ds in DATASETS:
        r = df[(df.dataset == ds) & (df.kernel == "rbf") & (df.model == "HHHv2") &
               (df.gamma_setting == "heuristic")]
        if r.empty:
            continue
        n_feat = int(pd.to_numeric(r["n_train"].iloc[0]))  # fallback, not quite right — use from the run
        # Better: look up n_features from a default-γ row's JSON  (we saved n_classes etc., not n_features)
        # Estimate via 1/γ_default relationship
        default_row = df[(df.dataset == ds) & (df.kernel == "rbf") & (df.model == "HHHv2") &
                         (df.gamma_setting == "default")]
        gh = r["gamma_used"].iloc[0]
        # We don't have γ_default stored (gamma_used = None when default), so infer from n_features
        # by inspecting one of our feature-count mappings. Use the JSON directly.
        import json
        jf = RESULTS_DIR / "ablation_gamma" / "runs" / f"{ds}__HHHv2__rbf__gamma_default.json"
        if jf.exists():
            d = json.loads(jf.read_text())
            # the run's config doesn't carry n_features but the JSON stores y_score_raw of shape (n_test, n_classes)
            # We need n_features separately. Use a hardcoded map from dataset stats:
            pass
        nf_map = {"BoT_IoT":23,"IoTID20":79,"ToN_IoT":41,"N_BaIoT":115,"CIC_IoT2023":48,"Edge_IIoTset":42,"5G_NIDD":49}
        n_feat = nf_map.get(ds, None)
        if n_feat is None:
            continue
        g_def = 1.0 / n_feat
        parts.append(f"| {ds} | {n_feat} | {g_def:.5f} | {gh:.4f} | **{gh/g_def:.0f}×** |")
    parts.append("")

    parts.append("### Observations")
    parts.append("")
    parts.append("- **The heuristic produces γ values 50×–3000× larger than sklearn's default** on these datasets. With `QuantileTransformer(output_distribution='normal')` or MinMax scaling, kNN distances collapse to a narrow range (σ ≲ 0.1), so `1/(2σ²)` explodes.")
    parts.append("- **RBF**: the heuristic γ over-localizes the kernel — on BoT_IoT, CIC_IoT2023, N_BaIoT, and Edge_IIoTset the HHHv2 MCC drop is modest (≤ 0.015), but on 5G_NIDD MCC drops from 0.99 to 0.87. CPAI-OvA (HHH) is more sensitive (up to −0.29 MCC on N_BaIoT).")
    parts.append("- **Poly**: heuristic is within ±0.03 MCC of the default on most datasets for HHHv2; Edge_IIoTset HHHv2 shows a large gap (MCC 0.77 → 0.17) because the large γ dominates the polynomial's additive `+1` offset, destroying the inner-product signal.")
    parts.append("- **Sigmoid**: both γ settings produce unstable MCC (0–0.6). Sigmoid is a known-bad kernel for multiclass and the heuristic doesn't rescue it.")
    parts.append("- **Practical recommendation**: the heuristic as formulated would need per-dataset rescaling (e.g., using the mean rather than median kNN distance, or applying a multiplier) to be competitive with `γ = 1/n_features` on the CPAI preprocessing pipeline. This is a finding worth discussing in the paper's limitations section.")
    parts.append("")
    return "\n".join(parts)


# ------------------------------------------------------------ confusion matrices

def confusion_matrices_block(sb: pd.DataFrame) -> str:
    parts = [
        "## Confusion Matrices — CPAI-OvA (HHH) and CPAI-OvR (HHHv2), Best Seed per Dataset",
        "",
        "For each dataset and each model, we select the single run with the highest MCC across "
        "all of Scene B and render the confusion matrix as a heatmap (rows = true class, columns = "
        "predicted class). Color encodes count; cells with `0` predictions are blank.",
        "",
        "Rendered PNGs live under `paper/figures/cm/` (one file per (dataset, model)) and two "
        "combined grids split for readability at `paper/figures/confusion_matrices_grid_part1.png` "
        "(3 datasets) and `paper/figures/confusion_matrices_grid_part2.png` (4 datasets).",
        "",
        "### Combined grids",
        "",
        "![Confusion matrices grid part 1](paper/figures/confusion_matrices_grid_part1.png)",
        "",
        "![Confusion matrices grid part 2](paper/figures/confusion_matrices_grid_part2.png)",
        "",
        "*2 columns (CPAI-OvA on the left, CPAI-OvR on the right); part 1 holds 3 datasets and "
        "part 2 holds the remaining 4. Each subplot title lists the winning kernel/scaler/poly + "
        "seed and its MCC.*",
        "",
        "### Individual figures + ASCII tables",
        "",
    ]
    runs_dir = RESULTS_DIR / "scene_b" / "runs"
    for ds in DATASETS:
        parts.append(f"#### {ds}")
        parts.append("")
        for model_name, mlbl in [("HHH", "CPAI-OvA (HHH)"), ("HHHv2", "CPAI-OvR (HHHv2)")]:
            sub = sb[(sb["dataset"] == ds) & (sb["model"] == model_name)]
            if sub.empty:
                continue
            best = sub.loc[sub["MCC"].idxmax()]
            run_id = (f"{best['dataset']}__{best['model']}__poly{int(best['poly'])}__"
                      f"{best['kernel']}__{best['scaler']}__seed{int(best['seed'])}")
            jf = runs_dir / f"{run_id}.json"
            if not jf.exists():
                continue
            data = json.loads(jf.read_text())
            cm = np.array(data["CFS Matrix"], dtype=int)
            classes = data["_classes"]
            # Rename 0Normal → Normal; if there's an extra -1 row (at index 0 from sklearn's
            # sorted labels), roll it to the end so Normal stays on the first row.
            renamed = ["Normal" if c == "0Normal" else c for c in classes]
            n_extra = cm.shape[0] - len(renamed)
            if n_extra > 0:
                cm = np.roll(cm, -n_extra, axis=0)
                cm = np.roll(cm, -n_extra, axis=1)
                cls_full = renamed + (["<other>"] * n_extra)
            else:
                cls_full = renamed
            parts.append(f"**{mlbl}** — kernel={best['kernel']}, scaler={best['scaler']}, "
                         f"poly={int(best['poly'])}, seed={int(best['seed'])}, MCC={best['MCC']:.4f}")
            parts.append("")
            parts.append(f"![CM {ds} {model_name}](paper/figures/cm/cm_{ds}_{model_name}.png)")
            parts.append("")
            header = ["true \\ pred"] + [str(c)[:14] for c in cls_full]
            parts.append("| " + " | ".join(header) + " |")
            parts.append("|---" * len(header) + "|")
            for i, lbl in enumerate(cls_full):
                parts.append("| " + str(lbl)[:14] + " | " + " | ".join(str(x) for x in cm[i]) + " |")
            parts.append("")
    return "\n".join(parts)


# ----------------------------------------------------------------------- main

def main() -> int:
    sb = load_scene_b()
    bl = load_baselines()

    parts = []
    parts.append("# CPAI: Revised Experimental Report\n")
    parts.append("A comprehensive update to the experimental section of "
                 "*CPAI: A Controlled-Point Discriminant Analysis for IoT Network Intrusion Detection* "
                 "with an expanded evaluation protocol (7 datasets, multi-seed, full baseline roster, "
                 "statistical significance tests).\n")
    parts.append(env_block())
    parts.append(preprocessing_block())
    parts.append(method_block())
    parts.append(baseline_params_block())
    parts.append(metrics_block())
    parts.append(kernel_study_tables(sb))
    parts.append(baseline_comparison_tables(sb, bl))
    parts.append(timing_block(sb, bl))
    parts.append(memory_block(sb))
    parts.append(figures_block())
    parts.append(ablation_ridge_block())
    parts.append(ablation_gamma_block())
    parts.append(confusion_matrices_block(sb))

    REPORT_PATH.write_text("\n\n".join(parts))
    print(f"[report] wrote {REPORT_PATH}  ({REPORT_PATH.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
