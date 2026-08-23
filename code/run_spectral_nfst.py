#!/usr/bin/env python3
"""Run Spectral NFST over all ten registered dataset variants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpai.kernels import KERNEL_CHOICES
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import SCALERS
from train_evaluate import append_row, run_one


DATASET_VARIANTS = (
    ("BoT_IoT", 1000),
    ("CIC_IoT2023", 1000),
    ("ToN_IoT", 1000),
    ("ToN_IoT", 2000),
    ("UNSW_NB15", 1000),
    ("IoTID20", 1000),
    ("IoTID20", 2000),
    ("N_BaIoT", 1000),
    ("Edge_IIoTset", 1000),
    ("5G_NIDD", 1000),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", default="rbf", choices=KERNEL_CHOICES)
    parser.add_argument("--scaler", default="QuantileTransformer", choices=SCALERS)
    parser.add_argument("--poly", type=int, default=-1, choices=[-1, 0, 2, 3])
    parser.add_argument("--components-per-class", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=RESULTS_DIR / "spectral_nfst" / "summary.csv",
    )
    args = parser.parse_args()

    if args.kernel == "none":
        parser.error("SpectralNFST requires a kernel; use rbf (recommended).")

    failures = 0
    for index, (dataset, limit) in enumerate(DATASET_VARIANTS, start=1):
        variant = f"{dataset}_{limit}"
        print(f"[{index:02d}/10] {variant}: training", flush=True)
        try:
            row = run_one(
                dataset=dataset,
                model_name="SpectralNFST",
                kernel=args.kernel,
                scaler=args.scaler,
                poly=args.poly,
                seed=args.seed,
                components_per_class=args.components_per_class,
                limit=limit,
                samples_per_class=args.samples_per_class,
            )
            append_row(row, args.output)
            print(
                f"          MCC={row['MCC']:.4f} ACC={row['ACC']:.2f} "
                f"F1={row['F1 Macro']:.2f}",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            print(f"          FAILED: {type(exc).__name__}: {exc}", flush=True)

    print(f"Completed: {10 - failures}/10; failed: {failures}; output: {args.output}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
