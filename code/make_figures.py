#!/usr/bin/env python3
"""Generate paper figures from the Scene B and baseline result summaries.

Outputs under paper/figures/:
  - MeanMCC_kernels_2_approach_v3.png : Figure 4 analog (mean MCC across kernels,
    OvA vs OvR, averaged over all 7 datasets x 5 seeds)
  - MeanF1_kernels_2_approach_v3.png  : Figure 5 analog (mean F1 macro, same layout)
  - binary_roc_by_dataset.png          : 7-subplot binary ROC (attack vs benign)
    comparing CPAI-OvR against the top baselines (LGBM, SAINT, TabNet)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Preload torch before scipy/matplotlib (macOS libomp)
try:
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpai.paths import RESULTS_DIR, PROJECT_ROOT

# Output dir: defaults to paper/figures; set FIGS_SUBDIR (e.g. "review_R2") to write
# into a subfolder without touching the existing figures.
_FIGS_SUBDIR = os.environ.get("FIGS_SUBDIR", "").strip()
FIGS_DIR = (PROJECT_ROOT / "paper" / "figures" / _FIGS_SUBDIR if _FIGS_SUBDIR
            else PROJECT_ROOT / "paper" / "figures")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["BoT_IoT", "IoTID20", "ToN_IoT", "N_BaIoT", "CIC_IoT2023", "Edge_IIoTset", "5G_NIDD"]
# Canonical kernel keys (must match the lowercase values in summary.csv — used for grouping).
KERNELS = ["None", "linear", "poly", "rbf", "sigmoid", "abel", "laplacian", "sobolev"]
# Display labels: RBF is an acronym; Abel/Laplacian/Sobolev are named after people.
KERNEL_DISPLAY = {"rbf": "RBF", "abel": "Abel", "laplacian": "Laplacian", "sobolev": "Sobolev"}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 140,
})


# ------------------------------------------------------------ Figures 4 and 5

def _best_per_kernel(sb: pd.DataFrame, metric: str) -> pd.DataFrame:
    """For each (dataset, model, kernel), pick best (scaler, poly) by mean metric
    over 5 seeds. Returns a long-form dataframe with the 5-seed mean of `metric`."""
    g = sb.groupby(["dataset", "model", "kernel", "scaler", "poly"])[metric].mean().reset_index()
    # For each (dataset, model, kernel) pick the (scaler, poly) max
    idx = g.groupby(["dataset", "model", "kernel"])[metric].idxmax()
    return g.loc[idx].reset_index(drop=True)


def fig_kernel_bars(sb: pd.DataFrame, metric: str, ylabel: str, title: str, fname: str,
                    ymin: float = 0.0) -> Path:
    best = _best_per_kernel(sb, metric)
    # Average across the 7 datasets per (model, kernel)
    mean_per = best.groupby(["model", "kernel"])[metric].agg(["mean", "std"]).reset_index()
    # Order kernels
    mean_per["kernel"] = pd.Categorical(mean_per["kernel"], KERNELS, ordered=True)
    mean_per = mean_per.sort_values(["kernel", "model"])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(KERNELS))
    width = 0.38
    ova = mean_per[mean_per["model"] == "HHH"].set_index("kernel").reindex(KERNELS)
    ovr = mean_per[mean_per["model"] == "HHHv2"].set_index("kernel").reindex(KERNELS)

    b1 = ax.bar(x - width/2, ova["mean"], width, yerr=ova["std"],
                label="CPAI-OvA", color="#4C72B0", edgecolor="black", capsize=3)
    b2 = ax.bar(x + width/2, ovr["mean"], width, yerr=ovr["std"],
                label="CPAI-OvR", color="#DD8452", edgecolor="black", capsize=3)

    # Annotate bar heights
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f"{h:.2f}" if metric == "MCC" else f"{h:.1f}",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([KERNEL_DISPLAY.get(k, k) for k in KERNELS], rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ymin, 1.05 if metric == "MCC" else 100 * 1.06)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = FIGS_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ------------------------------------------------------------ binary ROC panels

def load_y_for_run(run_dir: Path, dataset: str, model: str,
                   cfg_filter: dict | None = None,
                   seed: int = 42) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a per-run JSON matching (dataset, model, cfg_filter, seed) and return
    (y_true_binary, attack_score). Returns None if no matching JSON."""
    # Search pattern (both Scene B and baselines have uniform naming)
    for jf in run_dir.glob(f"{dataset}__{model}*.json"):
        d = json.loads(jf.read_text())
        c = d.get("_config") or {}
        if c.get("seed") != seed:
            continue
        if cfg_filter is not None:
            if not all(c.get(k) == v for k, v in cfg_filter.items()):
                continue
        y_true = np.array(d.get("y_true", []))
        # For CPAI runs y_score_attack_binary; for baselines we construct from y_score_raw (proba)
        if "y_score_attack_binary" in d:
            score = np.array(d["y_score_attack_binary"])
        elif "y_score_raw" in d:
            raw = np.array(d["y_score_raw"])
            if raw.ndim == 2 and raw.shape[1] >= 2:
                # attack score = 1 - P(class 0)
                score = 1.0 - raw[:, 0]
            else:
                score = raw
        else:
            return None
        if len(y_true) != len(score):
            return None
        y_bin = (y_true != 0).astype(int)
        return y_bin, score
    return None


# ----------------------------------------------------------- confusion matrices

def _best_run_json(sb: pd.DataFrame, dataset: str, model: str) -> dict | None:
    """Return the per-run JSON of the (dataset, model) row with the highest MCC overall."""
    sub = sb[(sb["dataset"] == dataset) & (sb["model"] == model)]
    if sub.empty:
        return None
    best = sub.loc[sub["MCC"].idxmax()]
    run_id = (f"{best['dataset']}__{best['model']}__poly{int(best['poly'])}__{best['kernel']}__"
              f"{best['scaler']}__seed{int(best['seed'])}")
    jf = RESULTS_DIR / "scene_b" / "runs" / f"{run_id}.json"
    if not jf.exists():
        return None
    return json.loads(jf.read_text())


def _prepare_cm_display(cm: np.ndarray, classes: list[str]) -> tuple[np.ndarray, list[str]]:
    """Reorder the confusion matrix for display:

    1. Rename `0Normal` → `Normal` (the `0` prefix was only there to force alphabetical
       sort so that benign is LabelEncoder class 0).
    2. If the cm has an extra row/col for the CPAI-OvA `-1` ("<other>") prediction, sklearn's
       `confusion_matrix(labels=sorted)` puts it at index 0. Move it to the **last** position
       so `Normal` stays on the first row/column.
    """
    n = cm.shape[0]
    renamed = ["Normal" if c == "0Normal" else c for c in classes]
    if n == len(renamed):
        return cm, renamed
    # n > len(classes) → there is a -1 <other> row/col at index 0; roll it to the end.
    n_extra = n - len(renamed)
    # Single-step roll by -n_extra moves the first n_extra rows/cols to the bottom/right.
    cm_rolled = np.roll(cm, -n_extra, axis=0)
    cm_rolled = np.roll(cm_rolled, -n_extra, axis=1)
    return cm_rolled, renamed + (["<other>"] * n_extra)


def _draw_cm(ax, cm: np.ndarray, classes: list[str], title: str, max_cells_for_text: int = 16):
    """Render a confusion matrix on `ax`. Annotate cells with counts when small enough."""
    cm, cls = _prepare_cm_display(cm, classes)
    n = cm.shape[0]
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    cls_short = [c[:14] for c in cls]
    ax.set_xticklabels(cls_short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cls_short, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    ax.set_title(title, fontsize=10)
    if n <= max_cells_for_text:
        # Annotate with counts (white text on dark cells, black on light)
        thresh = cm.max() / 2 if cm.max() > 0 else 1
        for i in range(n):
            for j in range(n):
                v = cm[i, j]
                if v == 0:
                    continue
                color = "white" if v > thresh else "black"
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=7, color=color)
    return im


def fig_confusion_matrices_individual(sb: pd.DataFrame) -> list[Path]:
    """One PNG per (dataset, model) — 14 files under paper/figures/cm/."""
    out_dir = FIGS_DIR / "cm"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ds in DATASETS:
        for model in ["HHH", "HHHv2"]:
            d = _best_run_json(sb, ds, model)
            if d is None:
                continue
            cm = np.array(d["CFS Matrix"], dtype=int)
            classes = d["_classes"]
            mlbl = "CPAI-OvA" if model == "HHH" else "CPAI-OvR"
            title = f"{ds} — {mlbl}"
            fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(classes) + 2),
                                            max(4.5, 0.40 * len(classes) + 2)))
            im = _draw_cm(ax, cm, classes, title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            out = out_dir / f"cm_{ds}_{model}.png"
            fig.savefig(out, bbox_inches="tight")
            plt.close(fig)
            written.append(out)
    return written


def fig_confusion_matrices_grid(sb: pd.DataFrame) -> list[Path]:
    """Two combined grids (datasets x 2 models). Part 1 holds 3 datasets, part 2
    holds the remaining 4. Splitting the old 7-row figure into two shorter PNGs
    lets each subplot (and its labels) render larger on the page."""
    splits = [
        ("confusion_matrices_grid_part1.png", DATASETS[:3]),
        ("confusion_matrices_grid_part2.png", DATASETS[3:]),
    ]
    written = []
    for fname, ds_group in splits:
        fig, axes = plt.subplots(len(ds_group), 2,
                                 figsize=(13, 4.4 * len(ds_group)))
        axes = np.atleast_2d(axes)
        for i, ds in enumerate(ds_group):
            for j, model in enumerate(["HHH", "HHHv2"]):
                ax = axes[i, j]
                d = _best_run_json(sb, ds, model)
                if d is None:
                    ax.axis("off")
                    continue
                cm = np.array(d["CFS Matrix"], dtype=int)
                classes = d["_classes"]
                mlbl = "CPAI-OvA" if model == "HHH" else "CPAI-OvR"
                title = f"{ds} - {mlbl}"
                im = _draw_cm(ax, cm, classes, title)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Confusion matrices: best per-dataset run for CPAI-OvA (left) and CPAI-OvR (right)",
                     fontsize=13, y=1.0)
        fig.tight_layout()
        out = FIGS_DIR / fname
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        written.append(out)
    return written


def fig_binary_roc_cpai_only(sb: pd.DataFrame) -> Path:
    """7-subplot binary ROC comparing CPAI-OvA vs CPAI-OvR, each at its best-per-dataset
    config (from Scene B, seed=42). No baselines — focus on the two proposed models."""
    sb_runs = RESULTS_DIR / "scene_b" / "runs"

    # Best-per-dataset config for each model
    best_cfg = {}
    for ds in DATASETS:
        for model in ("HHH", "HHHv2"):
            sub = sb[(sb["dataset"] == ds) & (sb["model"] == model)]
            agg = sub.groupby(["kernel", "scaler", "poly"])["MCC"].mean().reset_index()
            r = agg.loc[agg["MCC"].idxmax()]
            best_cfg[(ds, model)] = {"kernel": r["kernel"], "scaler": r["scaler"],
                                     "poly": int(r["poly"])}

    colors = {"HHH": "#4C72B0", "HHHv2": "#DD8452"}
    labels = {"HHH": "CPAI-OvA", "HHHv2": "CPAI-OvR"}

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        for model in ("HHH", "HHHv2"):
            cfg = best_cfg[(ds, model)]
            data = load_y_for_run(sb_runs, ds, model, cfg_filter=cfg, seed=42)
            if data is None:
                continue
            y_bin, score = data
            if len(np.unique(y_bin)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin, score)
            auc_val = auc(fpr, tpr)
            fpr_plot = np.clip(fpr, 1e-4, 1.0)
            ax.plot(fpr_plot, tpr, label=f"{labels[model]} (AUC={auc_val:.3f})",
                    color=colors[model], lw=2.2)
        # Diagonal
        diag_fpr = np.logspace(-4, 0, 200)
        ax.plot(diag_fpr, diag_fpr, "k--", lw=0.8, alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate (log scale)")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(ds)
        ax.grid(which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    axes[7].axis("off")
    fig.suptitle("Binary ROC — CPAI-OvA vs CPAI-OvR (log-scale FPR)",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIGS_DIR / "binary_roc_cpai_only.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_binary_roc(sb: pd.DataFrame) -> Path:
    """7-subplot grid: one binary ROC panel per dataset comparing CPAI-OvR,
    LGBM, SAINT, TabNet at seed=42."""
    sb_runs = RESULTS_DIR / "scene_b" / "runs"
    bl_runs = RESULTS_DIR / "baselines_default" / "runs"

    # Per-dataset: find the HHHv2 best config from Scene B summary
    best_cfg = {}
    for ds in DATASETS:
        sub = sb[(sb["dataset"] == ds) & (sb["model"] == "HHHv2")]
        agg = sub.groupby(["kernel", "scaler", "poly"])["MCC"].mean().reset_index()
        r = agg.loc[agg["MCC"].idxmax()]
        best_cfg[ds] = {"kernel": r["kernel"], "scaler": r["scaler"], "poly": int(r["poly"])}

    models = [
        ("CPAI-OvR", "HHHv2", sb_runs, True),
        ("LGBM",     "LGBM",   bl_runs, False),
        ("SAINT",    "SAINT",  bl_runs, False),
        ("TabNet",   "TabNet", bl_runs, False),
    ]
    colors = {"CPAI-OvR": "#DD8452", "LGBM": "#55A868", "SAINT": "#C44E52",
              "TabNet": "#8172B3"}
    linewidths = {"CPAI-OvR": 2.5, "LGBM": 1.5, "SAINT": 1.5, "TabNet": 1.5}

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, ds in enumerate(DATASETS):
        ax = axes[i]
        for label, model, run_dir, is_cpai in models:
            cfg = best_cfg[ds] if is_cpai else None
            data = load_y_for_run(run_dir, ds, model, cfg_filter=cfg, seed=42)
            if data is None:
                continue
            y_bin, score = data
            if len(np.unique(y_bin)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin, score)
            auc_val = auc(fpr, tpr)
            # Log-scale FPR so the low-false-positive region (deployment sweet spot) is visible.
            # Clamp FPR below 1e-4 to avoid log(0). Curves are monotonic so this is safe.
            fpr_plot = np.clip(fpr, 1e-4, 1.0)
            ax.plot(fpr_plot, tpr, label=f"{label} (AUC={auc_val:.3f})",
                    color=colors[label], lw=linewidths[label])
        # Random-classifier diagonal in log-linear space
        diag_fpr = np.logspace(-4, 0, 200)
        ax.plot(diag_fpr, diag_fpr, "k--", lw=0.8, alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1.0)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("False Positive Rate (log scale)")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(ds)
        ax.grid(which="both", linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", fontsize=10, framealpha=0.95)

    axes[7].axis("off")

    fig.suptitle("Binary ROC (attack vs benign), log-scale FPR",
                 fontsize=13, y=1.01)
    fig.tight_layout()
    out = FIGS_DIR / "binary_roc_by_dataset.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ----------------------------------------------------------------------- main

def main() -> int:
    # Scene B summary
    sb = pd.read_csv(RESULTS_DIR / "scene_b" / "summary.csv", keep_default_na=False)
    for c in ["MCC", "F1_macro", "AUROC_binary", "t_fit_s", "peak_rss_mb"]:
        sb[c] = pd.to_numeric(sb[c], errors="coerce")
    sb["poly"] = sb["poly"].astype(int)
    sb["seed"] = pd.to_numeric(sb["seed"], errors="coerce").astype(int)

    out1 = fig_kernel_bars(sb, "MCC",
                           "Mean MCC (over 7 datasets × 5 seeds)",
                           "Mean MCC by kernel — CPAI-OvA vs CPAI-OvR",
                           "MeanMCC_kernels_2_approach_v3.png", ymin=0.0)
    print(f"[figure] wrote {out1}")

    out2 = fig_kernel_bars(sb, "F1_macro",
                           "Mean F1 macro [%] (over 7 datasets × 5 seeds)",
                           "Mean F1 macro by kernel — CPAI-OvA vs CPAI-OvR",
                           "MeanF1_kernels_2_approach_v3.png", ymin=0.0)
    print(f"[figure] wrote {out2}")

    out3 = fig_binary_roc(sb)
    print(f"[figure] wrote {out3}")

    out4 = fig_binary_roc_cpai_only(sb)
    print(f"[figure] wrote {out4}")

    out5 = fig_confusion_matrices_individual(sb)
    print(f"[figure] wrote {len(out5)} confusion-matrix PNGs to {FIGS_DIR / 'cm'}")

    out6 = fig_confusion_matrices_grid(sb)
    for p in out6:
        print(f"[figure] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
