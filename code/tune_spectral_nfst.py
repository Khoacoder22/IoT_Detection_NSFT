#!/usr/bin/env python3
"""Run 12 Spectral NFST experiments: 1 exponential kernel x 6 gamma x 2 features."""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpai import SpectralNFST
from cpai.datasets import DATASETS, load_dataset
from cpai.metrics import evaluate
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import SCALERS, preprocess


KERNEL = "l05_exponential_kernel"
GAMMAS = (
    ("heuristic", "heuristic"),
    ("0.001", 0.001),
    ("0.01", 0.01),
    ("0.1", 0.1),
    ("1", 1.0),
    ("10", 10.0),
)
FEATURES = (-1, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--limit", type=int, default=1000, choices=[1000, 2000])
    parser.add_argument("--scaler", default="QuantileTransformer", choices=SCALERS)
    parser.add_argument("--q", type=int, default=2)
    parser.add_argument("--samples-per-class", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seed = 42

    if args.limit == 2000 and args.dataset not in {"ToN_IoT", "IoTID20"}:
        parser.error("limit 2000 is available only for ToN_IoT and IoTID20")
    if args.q < 1:
        parser.error("q must be at least 1")
    if args.samples_per_class == 1 or args.samples_per_class < 0:
        parser.error("samples-per-class must be 0 (full data) or at least 2")

    sample_tag = "full" if args.samples_per_class == 0 else f"sample{args.samples_per_class}"
    output = args.output or (
        RESULTS_DIR / "spectral_nfst" /
        f"gamma_{args.dataset}_{args.limit}_{sample_tag}_seed{seed}.csv"
    )
    error_output = output.with_name(f"{output.stem}_errors.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    completed = set()
    if output.exists():
        old = pd.read_csv(output, dtype=str)
        completed.update(zip(old["Feature"], old["Gamma"]))
    if error_output.exists():
        old_errors = pd.read_csv(error_output, dtype=str)
        completed.update(zip(old_errors["Feature"], old_errors["Gamma"]))

    df, _ = load_dataset(args.dataset, limit=args.limit)
    if args.samples_per_class:
        df = pd.concat(
            group.sample(
                n=min(args.samples_per_class, len(group)), random_state=seed
            )
            for _, group in df.groupby("Label", sort=True)
        ).reset_index(drop=True)

    total = len(GAMMAS) * len(FEATURES)
    position = 0
    for feature in FEATURES:
        X_train, y_train, X_test, y_test, _ = preprocess(
            df,
            args.dataset,
            poly=feature,
            kernel=KERNEL,
            scaler=args.scaler,
            seed=seed,
        )
        for gamma_label, gamma in GAMMAS:
            position += 1
            key = (str(feature), gamma_label)
            if key in completed:
                print(f"[{position}/{total}] SKIP feature={feature}, gamma={gamma_label}")
                continue

            print(f"[{position}/{total}] RUN  feature={feature}, gamma={gamma_label}")
            try:
                model = SpectralNFST(
                    n_components=args.q,
                    kernel=KERNEL,
                    gamma=gamma,
                )
                started = time.perf_counter()
                # Keep the model unchanged; suppress its emoji debug line because
                # some Windows consoles cannot encode it.
                with redirect_stdout(io.StringIO()):
                    model.fit(X_train, y_train)
                train_time = time.perf_counter() - started

                started = time.perf_counter()
                prediction = model.predict(X_test)
                test_time = time.perf_counter() - started
                metrics = evaluate(y_test, prediction)
            except Exception as exc:
                error_row = {
                    "Dataset": f"{args.dataset}_{args.limit}",
                    "Kernel": KERNEL,
                    "Gamma": gamma_label,
                    "Feature": feature,
                    "Scaler": args.scaler,
                    "Q": args.q,
                    "Seed": seed,
                    "Error": f"{type(exc).__name__}: {exc}",
                }
                pd.DataFrame([error_row]).to_csv(
                    error_output,
                    mode="a",
                    header=not error_output.exists(),
                    index=False,
                )
                print(f"    FAILED: {error_row['Error']}; continuing")
                continue
            row = {
                "Dataset": f"{args.dataset}_{args.limit}",
                "Kernel": KERNEL,
                "Gamma": gamma_label,
                "Gamma used": model._gamma_used,
                "Feature": feature,
                "Scaler": args.scaler,
                "Q": args.q,
                "Seed": seed,
                "MCC": metrics["MCC"],
                "ACC": metrics["ACC"],
                "F1 Macro": metrics["F1 Macro"],
                "TPR Macro": metrics["TPR Macro"],
                "FPR": metrics["FPR"],
                "PPV Macro": metrics["PPV Macro"],
                "Training time": train_time,
                "Test time": test_time,
            }
            pd.DataFrame([row]).to_csv(
                output, mode="a", header=not output.exists(), index=False
            )
            print(f"    MCC={row['MCC']:.6f}, F1={row['F1 Macro']:.2f}")

    if output.exists():
        results = pd.read_csv(output).sort_values(
            ["MCC", "F1 Macro"], ascending=False
        )
        print(f"\nSaved: {output}")
        current_gammas = {label for label, _ in GAMMAS}
        results = results[results["Gamma"].astype(str).isin(current_gammas)]
        print(results[["Feature", "Gamma", "MCC", "F1 Macro", "ACC"]].head(12).to_string(index=False))
    if error_output.exists():
        print(f"Failed configurations: {error_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
