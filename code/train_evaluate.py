#!/usr/bin/env python3
"""Single-run CLI: train one (dataset, kernel, scaler, poly, model) combo and append to output CSV.

Usage:
    python3 train_evaluate.py --dataset BoT_IoT --model HHHv2 --kernel rbf \
        --scaler QuantileTransformer --poly -1 --output results/runs/out.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpai import HHH, HHHv2, KNFST, SpectralNFST, build_baseline
from cpai.baselines import BASELINE_NAMES
from cpai.datasets import DATASETS, load_dataset
from cpai.kernels import KERNEL_CHOICES
from cpai.metrics import evaluate
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import SCALERS, preprocess

MODELS = ("HHH", "HHHv2", "KNFST", "SpectralNFST", *BASELINE_NAMES)

OUTPUT_COLUMNS = [
    "Data Type", "Poly", "Kernel", "SCALER", "Model",
    "MCC", "NDR", "AUC_N", "ACC", "TPR Macro",
    "FPR", "PPV Macro", "F1 Macro", "AUC", "CLS Report",
    "CFS Matrix", "Training time", "Test time",
]


def _build_model(name: str, kernel: str | None, components_per_class: int = 2):
    if name == "HHH":
        return HHH(kernel=kernel)
    if name == "HHHv2":
        return HHHv2(kernel=kernel)
    if name == "KNFST":
        return KNFST(kernel=kernel or "rbf")
    if name == "SpectralNFST":
        return SpectralNFST(
            n_components=components_per_class, kernel=kernel or "rbf"
        )
    return build_baseline(name)


def run_one(
    dataset: str,
    model_name: str,
    kernel: str | None,
    scaler: str,
    poly: int,
    seed: int = 42,
    components_per_class: int = 2,
    limit: int | None = None,
    samples_per_class: int | None = None,
) -> dict:
    df, resolved_limit = load_dataset(dataset, limit=limit)
    if samples_per_class is not None:
        if samples_per_class < 2:
            raise ValueError("samples_per_class must be at least 2")
        sampled = []
        for _, group in df.groupby("Label", sort=True):
            sampled.append(
                group.sample(n=min(samples_per_class, len(group)), random_state=seed)
            )
        df = pd.concat(sampled, ignore_index=True)
    X_train, y_train, X_test, y_test, _ = preprocess(
        df, dataset, poly=poly, kernel=kernel, scaler=scaler, seed=seed,
    )

    model = _build_model(model_name, kernel, components_per_class)

    t0 = time.time()
    model.fit(X_train, y_train)
    t1 = time.time()
    y_pred = model.predict(X_test)
    t2 = time.time()

    result = evaluate(y_test, y_pred)
    return {
        "Data Type": f"{dataset}_{resolved_limit}" if limit is not None else dataset,
        "Poly": poly,
        "Kernel": "None" if (kernel is None or kernel == "none") else kernel,
        "SCALER": scaler,
        "Model": (
            f"SpectralNFST-Q{components_per_class}"
            if model_name == "SpectralNFST" else model_name
        ),
        "MCC": result["MCC"],
        "NDR": -1,
        "AUC_N": -1,
        "ACC": result["ACC"],
        "TPR Macro": result["TPR Macro"],
        "FPR": result["FPR"],
        "PPV Macro": result["PPV Macro"],
        "F1 Macro": result["F1 Macro"],
        "AUC": result["AUC"],
        "CLS Report": "",
        "CFS Matrix": np.array2string(result["CFS Matrix"]),
        "Training time": t1 - t0,
        "Test time": t2 - t1,
    }


def append_row(row: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=OUTPUT_COLUMNS)
    header = not output_path.exists()
    df.to_csv(output_path, mode="a", header=header, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--kernel", default="none", choices=KERNEL_CHOICES)
    parser.add_argument("--scaler", default="QuantileTransformer", choices=SCALERS)
    parser.add_argument("--poly", type=int, default=-1, choices=[-1, 0, 2, 3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, choices=[1000, 2000],
        help="Select a registered dataset-size variant, where available.",
    )
    parser.add_argument(
        "--samples-per-class", type=int,
        help="Optional deterministic cap per class before the train/test split.",
    )
    parser.add_argument(
        "--components-per-class", type=int, default=2,
        help="SpectralNFST target components Q_j for every class (default: 2).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=RESULTS_DIR / "runs" / "runs.csv",
        help="CSV to append this row to (created if missing).",
    )
    args = parser.parse_args()

    kernel = None if args.kernel == "none" else args.kernel
    print(f"[run] dataset={args.dataset} model={args.model} kernel={kernel} "
          f"scaler={args.scaler} poly={args.poly}")
    row = run_one(
        args.dataset, args.model, kernel, args.scaler, args.poly,
        seed=args.seed, components_per_class=args.components_per_class,
        limit=args.limit, samples_per_class=args.samples_per_class,
    )
    print(f"[run] MCC={row['MCC']:.4f} ACC={row['ACC']:.2f} TPR={row['TPR Macro']:.2f}")
    append_row(row, args.output)
    print(f"[run] appended to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
