#!/usr/bin/env python3
"""CLI 2/2: multiclass multi-novelty detection with Spectral NFST."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpai.datasets import DATASETS
from cpai.kernels import KERNEL_CHOICES
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import SCALERS
from spectral_nfst_experiments import (
    dataset_class_labels,
    run_spectral_nfst_multinovelty,
)


def main() -> int:
    """Parse unknown labels and run one fixed-threshold novelty experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--limit", type=int, choices=[1000, 2000])
    parser.add_argument(
        "--unknown-labels", nargs="+",
        help="One or more cleaned original labels to exclude from training.",
    )
    parser.add_argument(
        "--list-classes", action="store_true",
        help="Print valid cleaned labels for this dataset and exit.",
    )
    parser.add_argument(
        "--threshold", type=float,
        help="Externally selected novelty threshold tau (required for a run).",
    )
    parser.add_argument("--kernel", default="rbf", choices=KERNEL_CHOICES)
    parser.add_argument("--scaler", default="QuantileTransformer", choices=SCALERS)
    parser.add_argument("--poly", type=int, default=-1, choices=[-1, 0, 2, 3])
    parser.add_argument("--components-per-class", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=RESULTS_DIR / "spectral_nfst" / "multinovelty.csv",
    )
    args = parser.parse_args()

    if args.list_classes:
        for index, label in enumerate(dataset_class_labels(args.dataset, args.limit)):
            print(f"[{index}] {label}")
        return 0
    if not args.unknown_labels:
        parser.error("--unknown-labels is required unless --list-classes is used")
    if args.threshold is None:
        parser.error("--threshold is required and must be selected outside the test set")

    try:
        row = run_spectral_nfst_multinovelty(
            dataset=args.dataset,
            limit=args.limit,
            unknown_labels=args.unknown_labels,
            threshold=args.threshold,
            kernel=args.kernel,
            scaler=args.scaler,
            poly=args.poly,
            components_per_class=args.components_per_class,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            output=args.output,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Dataset: {row['Data Type']}")
    print(f"Unknown labels: {row['Unknown labels']}")
    print(
        f"MCC={row['MCC']:.4f} F1={row['F1 Macro']:.2f} "
        f"NDR={row['NDR']:.2f} FPR_N={row['FPR_N']:.2f} "
        f"AUC_N={row['AUC_N']:.2f} AUPRC_N={row['AUPRC_N']:.2f}"
    )
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
