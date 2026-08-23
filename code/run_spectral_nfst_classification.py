#!/usr/bin/env python3
"""CLI 1/2: closed-set multiclass classification with Spectral NFST."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpai.kernels import KERNEL_CHOICES
from cpai.datasets import DATASETS
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import SCALERS
from spectral_nfst_experiments import run_spectral_nfst_classification


def main(default_output: Path | None = None) -> int:
    """Parse CLI options and call the reusable classification experiment."""
    default_output = default_output or RESULTS_DIR / "spectral_nfst" / "classification.csv"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=DATASETS,
        help="Run only this dataset. Omit this option to run all 10 variants.",
    )
    parser.add_argument(
        "--limit", type=int, choices=[1000, 2000], default=1000,
        help="Dataset-size variant used with --dataset (default: 1000).",
    )
    parser.add_argument("--kernel", default="rbf", choices=KERNEL_CHOICES)
    parser.add_argument("--scaler", default="QuantileTransformer", choices=SCALERS)
    parser.add_argument("--poly", type=int, default=-1, choices=[-1, 0, 2, 3])
    parser.add_argument("--components-per-class", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=default_output,
    )
    args = parser.parse_args()

    variants = None
    if args.dataset is not None:
        variants = ((args.dataset, args.limit),)

    try:
        results = run_spectral_nfst_classification(
            kernel=args.kernel,
            scaler=args.scaler,
            poly=args.poly,
            components_per_class=args.components_per_class,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
            output=args.output,
            variants=variants,
        )
    except ValueError as exc:
        parser.error(str(exc))

    failures = [row for row in results if row["_status"] == "error"]
    print(
        f"Completed classification: {len(results)-len(failures)}/{len(results)}; "
        f"failed: {len(failures)}; output: {args.output}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
