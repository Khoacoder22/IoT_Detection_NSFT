#!/usr/bin/env python3
"""Reuse completed tuning results, run only missing final SpectralNFST configs,
then compare them with all complete seed-42 competitor CSVs.
"""

from __future__ import annotations

import io
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpai import SpectralNFST
from cpai.datasets import load_dataset
from cpai.metrics import evaluate
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import preprocess


SEED = 42
KERNEL = "l05_exponential_kernel"
SCALER = "QuantileTransformer"
DATASETS = (
    ("BoT_IoT", 1000),
    ("CIC_IoT2023", 1000),
    ("ToN_IoT", 1000),
    ("UNSW_NB15", 1000),
    ("IoTID20", 2000),
    ("N_BaIoT", 1000),
    ("Edge_IIoTset", 1000),
    ("5G_NIDD", 1000),
)
Q_FILES = {
    "BoT_IoT_1000": ("grid_BoT_IoT_1000_sample1000_seed424344.csv",),
    "CIC_IoT2023_1000": ("grid_CIC_IoT2023_1000_sample1000_seed424344.csv",),
    "ToN_IoT_1000": ("grid_ToN_IoT_1000_sample1000_seed42.csv",),
    "UNSW_NB15_1000": ("grid_UNSW_NB15_1000_sample1000_seed42.csv",),
    "IoTID20_2000": tuple(
        f"grid_IoTID20_2000_sample1000_seed{seed}.csv" for seed in (42, 43, 44)
    ),
    "N_BaIoT_1000": tuple(
        f"grid_N_BaIoT_1000_sample1000_seed{seed}.csv" for seed in (42, 43, 44)
    ),
    "Edge_IIoTset_1000": tuple(
        f"grid_Edge_IIoTset_1000_sample1000_seed{seed}.csv" for seed in (42, 43, 44)
    ),
    "5G_NIDD_1000": ("grid_5G_NIDD_1000_sample1000_seed424344.csv",),
}
METRICS = ("MCC", "F1 Macro", "ACC", "FPR", "Training time", "Test time")
EXPECTED_COMPETITORS = {
    "KNFST", "HHH", "HHHv2", "KNN", "LDA", "NuSVC", "SGD", "GauNB",
    "NC", "RNC", "LGBM", "MLP", "TabNet", "FTTransformer", "SAINT",
}


def _best_gamma_feature(dataset: str, limit: int) -> pd.Series:
    path = (
        RESULTS_DIR / "spectral_nfst" /
        f"gamma_{dataset}_{limit}_full_seed{SEED}.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing gamma result: {path}")
    rows = pd.read_csv(path)
    rows = rows[rows["Gamma"].astype(str) != "default"].copy()
    rows["MCC"] = pd.to_numeric(rows["MCC"], errors="coerce")
    rows["F1 Macro"] = pd.to_numeric(rows["F1 Macro"], errors="coerce")
    rows["Feature"] = pd.to_numeric(rows["Feature"], errors="coerce")
    rows["Training time"] = pd.to_numeric(rows["Training time"], errors="coerce")
    rows = rows.dropna(subset=["MCC", "F1 Macro", "Feature"])
    if rows.empty:
        raise ValueError(f"No successful gamma/feature result: {path}")
    # Prefer the simpler feature=-1 pipeline when metrics tie.
    return rows.sort_values(
        ["MCC", "F1 Macro", "Feature", "Training time"],
        ascending=[False, False, True, True],
    ).iloc[0]


def _best_q(data_type: str) -> int:
    frames = []
    for filename in Q_FILES[data_type]:
        path = RESULTS_DIR / "spectral_nfst" / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing Q result: {path}")
        frames.append(pd.read_csv(path))
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[
        (rows["Data Type"] == data_type)
        & (rows["Kernel"] == KERNEL)
        & (rows["SCALER"] == SCALER)
        & (pd.to_numeric(rows["Poly"], errors="coerce") == -1)
    ].copy()
    rows["Q"] = pd.to_numeric(
        rows["Model"].astype(str).str.extract(r"Q(\d+)")[0], errors="coerce"
    )
    rows["MCC"] = pd.to_numeric(rows["MCC"], errors="coerce")
    rows["F1 Macro"] = pd.to_numeric(rows["F1 Macro"], errors="coerce")
    rows = rows.dropna(subset=["Q", "MCC", "F1 Macro"])
    if rows.empty:
        raise ValueError(f"No matching Q result for {data_type}")
    summary = (
        rows.groupby("Q", as_index=False)
        .agg(MCC=("MCC", "mean"), F1=("F1 Macro", "mean"))
        .sort_values(["MCC", "F1", "Q"], ascending=[False, False, True])
    )
    return int(summary.iloc[0]["Q"])


def _gamma_value(label: str):
    return "heuristic" if label == "heuristic" else float(label)


def _best_configs() -> list[dict]:
    configs = []
    for dataset, limit in DATASETS:
        data_type = f"{dataset}_{limit}"
        gamma_row = _best_gamma_feature(dataset, limit)
        configs.append({
            "dataset": dataset,
            "limit": limit,
            "data_type": data_type,
            "gamma_label": str(gamma_row["Gamma"]),
            "gamma": _gamma_value(str(gamma_row["Gamma"])),
            "feature": int(gamma_row["Feature"]),
            "q": _best_q(data_type),
            "gamma_row": gamma_row,
        })
    return configs


def _cached_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = pd.read_csv(path).to_dict("records")
    return {str(row["Data Type"]): row for row in rows}


def _cache_matches(row: dict, config: dict) -> bool:
    return (
        str(row.get("Gamma")) == config["gamma_label"]
        and int(float(row.get("Poly"))) == config["feature"]
        and int(float(row.get("Q"))) == config["q"]
    )


def _result_row(config: dict, metrics: dict, train_time, test_time, source: str) -> dict:
    return {
        "Seed": SEED,
        "Input Model": "SpectralNFST-Tuned",
        "Data Type": config["data_type"],
        "Poly": config["feature"],
        "Kernel": KERNEL,
        "SCALER": SCALER,
        "Model": "SpectralNFST-Tuned",
        "Q": config["q"],
        "Gamma": config["gamma_label"],
        "MCC": metrics["MCC"],
        "ACC": metrics["ACC"],
        "TPR Macro": metrics["TPR Macro"],
        "FPR": metrics["FPR"],
        "PPV Macro": metrics["PPV Macro"],
        "F1 Macro": metrics["F1 Macro"],
        "Training time": train_time,
        "Test time": test_time,
        "CFS Matrix": metrics.get("CFS Matrix", ""),
        "Source": source,
    }


def _run_final_spectral(configs: list[dict], output: Path) -> pd.DataFrame:
    cache = _cached_rows(output)
    final_rows = []
    for position, config in enumerate(configs, start=1):
        cached = cache.get(config["data_type"])
        if cached is not None and _cache_matches(cached, config):
            print(f"[{position}/8] SKIP {config['data_type']} (cached final result)")
            final_rows.append(cached)
            continue

        print(
            f"[{position}/8] FINAL {config['data_type']}: gamma={config['gamma_label']}, "
            f"feature={config['feature']}, Q={config['q']}"
        )
        if config["q"] == 2:
            old = config["gamma_row"]
            metrics = {name: old[name] for name in (
                "MCC", "ACC", "TPR Macro", "FPR", "PPV Macro", "F1 Macro"
            )}
            row = _result_row(
                config,
                metrics,
                old["Training time"],
                old["Test time"],
                "reused-gamma-grid",
            )
            print("    reused existing full-data Q=2 result")
        else:
            df, _ = load_dataset(config["dataset"], limit=config["limit"])
            X_train, y_train, X_test, y_test, _ = preprocess(
                df,
                config["dataset"],
                poly=config["feature"],
                kernel=KERNEL,
                scaler=SCALER,
                seed=SEED,
            )
            model = SpectralNFST(
                n_components=config["q"],
                kernel=KERNEL,
                gamma=config["gamma"],
            )
            started = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                model.fit(X_train, y_train)
            train_time = time.perf_counter() - started
            started = time.perf_counter()
            prediction = model.predict(X_test)
            test_time = time.perf_counter() - started
            row = _result_row(
                config,
                evaluate(y_test, prediction),
                train_time,
                test_time,
                "final-best-config",
            )
            print(f"    MCC={float(row['MCC']):.6f}, F1={float(row['F1 Macro']):.2f}")
        final_rows.append(row)
        pd.DataFrame(final_rows).to_csv(output, index=False)

    ordered = pd.DataFrame(final_rows)
    ordered["_order"] = ordered["Data Type"].map(
        {f"{dataset}_{limit}": i for i, (dataset, limit) in enumerate(DATASETS)}
    )
    ordered = ordered.sort_values("_order").drop(columns="_order")
    ordered.to_csv(output, index=False)
    return ordered


def _competitor_rows(models_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    frames = []
    excluded = []
    wanted = {f"{dataset}_{limit}" for dataset, limit in DATASETS}
    for path in sorted(models_dir.glob("*.csv")):
        if path.stem.lower().startswith("spectralnfst"):
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            excluded.append(f"{path.name}: unreadable ({type(exc).__name__})")
            continue
        required = {"Seed", "Data Type", *METRICS}
        if not required.issubset(frame.columns):
            excluded.append(f"{path.name}: missing required columns")
            continue
        frame["Seed"] = pd.to_numeric(frame["Seed"], errors="coerce")
        frame = frame[
            (frame["Seed"] == SEED) & frame["Data Type"].isin(wanted)
        ].copy()
        for column in METRICS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=list(METRICS))
        frame = frame.drop_duplicates(subset=["Data Type"], keep="last")
        if set(frame["Data Type"]) != wanted:
            excluded.append(
                f"{path.name}: has {frame['Data Type'].nunique()}/8 seed-42 datasets"
            )
            continue
        frame["Comparison Model"] = path.stem
        frames.append(frame)
    if not frames:
        raise ValueError("No complete seed-42 competitor file was found")
    return pd.concat(frames, ignore_index=True), excluded


def _print_per_dataset_comparison(all_rows: pd.DataFrame) -> None:
    """Print the tuned SpectralNFST parameters and full ranking per dataset."""
    summary_rows = []
    for dataset, limit in DATASETS:
        data_type = f"{dataset}_{limit}"
        rows = all_rows[all_rows["Data Type"] == data_type].copy()
        rows = rows.sort_values(
            ["MCC", "F1 Macro"], ascending=[False, False]
        ).reset_index(drop=True)
        rows.insert(0, "Rank", range(1, len(rows) + 1))

        spectral_rows = rows[
            rows["Comparison Model"] == "SpectralNFST-Tuned"
        ]
        if spectral_rows.empty:
            print(f"\nDATASET: {data_type}\nNo SpectralNFST-Tuned result found.")
            continue

        spectral = spectral_rows.iloc[0]
        spectral_rank = int(spectral["Rank"])
        spectral_mcc = float(spectral["MCC"])
        spectral_feature = int(float(spectral.get("Poly", -1)))
        spectral_q = int(float(spectral.get("Q", 0)))
        summary_rows.append({
            "Dataset": data_type,
            "Rank": f"{spectral_rank}/{len(rows)}",
            "MCC": spectral_mcc,
            "F1 Macro": float(spectral["F1 Macro"]),
            "Gamma": spectral.get("Gamma", ""),
            "Feature": spectral_feature,
            "Q": spectral_q,
            "Best Model": rows.iloc[0]["Comparison Model"],
        })

        display = rows[
            [
                "Rank", "Comparison Model", "MCC", "F1 Macro", "ACC", "FPR",
                "Training time", "Test time",
            ]
        ].copy()
        display["MCC vs Spectral"] = display["MCC"] - spectral_mcc
        display["Comparison Model"] = display["Comparison Model"].map(
            lambda model: (
                f">>> {model} <<<"
                if model == "SpectralNFST-Tuned"
                else model
            )
        )
        display = display.rename(columns={
            "Comparison Model": "Model",
            "Training time": "Train(s)",
            "Test time": "Test(s)",
        })
        display = display[
            [
                "Rank", "Model", "MCC", "MCC vs Spectral", "F1 Macro", "ACC",
                "FPR", "Train(s)", "Test(s)",
            ]
        ]

        print("\n" + "=" * 118)
        print(f"DATASET: {data_type}")
        print(
            "SpectralNFST-Tuned best parameters: "
            f"gamma={spectral.get('Gamma', '')} | "
            f"feature/poly={spectral_feature} | "
            f"Q={spectral_q} | "
            f"kernel={spectral.get('Kernel', '')} | "
            f"scaler={spectral.get('SCALER', '')} | seed={int(spectral['Seed'])}"
        )
        print(
            f"SpectralNFST-Tuned rank: {spectral_rank}/{len(rows)} | "
            f"MCC={spectral_mcc:.6f} | F1 Macro={float(spectral['F1 Macro']):.2f}"
        )
        print("MCC vs Spectral: positive means the competitor is better; negative means worse.")
        print(
            display.to_string(
                index=False,
                formatters={
                    "MCC": lambda value: f"{value:.6f}",
                    "MCC vs Spectral": lambda value: f"{value:+.6f}",
                    "F1 Macro": lambda value: f"{value:.2f}",
                    "ACC": lambda value: f"{value:.2f}",
                    "FPR": lambda value: f"{value:.2f}",
                    "Train(s)": lambda value: f"{value:.3f}",
                    "Test(s)": lambda value: f"{value:.4f}",
                },
            )
        )

    print("\n" + "=" * 118)
    print("SPECTRAL NFST SUMMARY BY DATASET")
    print(
        pd.DataFrame(summary_rows).to_string(
            index=False,
            formatters={
                "MCC": lambda value: f"{value:.6f}",
                "F1 Macro": lambda value: f"{value:.2f}",
            },
        )
    )


def main() -> int:
    comparison_dir = RESULTS_DIR / "knfst_comparison"
    models_dir = comparison_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    tuned_path = models_dir / "SpectralNFST_Tuned.csv"

    configs = _best_configs()
    print("Reused best parameters:")
    print(
        pd.DataFrame(configs)[
            ["data_type", "gamma_label", "feature", "q"]
        ].to_string(index=False)
    )
    spectral = _run_final_spectral(configs, tuned_path)

    competitors, excluded = _competitor_rows(models_dir)
    missing_models = sorted(
        EXPECTED_COMPETITORS - set(competitors["Comparison Model"])
    )
    spectral_compare = spectral.copy()
    spectral_compare["Comparison Model"] = "SpectralNFST-Tuned"
    all_rows = pd.concat(
        [competitors, spectral_compare], ignore_index=True, sort=False
    )
    all_rows_path = comparison_dir / "seed42_spectral_tuned_all_models.csv"
    all_rows.to_csv(all_rows_path, index=False)

    ranking = (
        all_rows.groupby("Comparison Model", as_index=False)
        .agg(
            Datasets=("Data Type", "nunique"),
            MCC_Mean=("MCC", "mean"),
            F1_Mean=("F1 Macro", "mean"),
            ACC_Mean=("ACC", "mean"),
            FPR_Mean=("FPR", "mean"),
            Train_s_Mean=("Training time", "mean"),
            Test_s_Mean=("Test time", "mean"),
        )
        .sort_values(["MCC_Mean", "F1_Mean"], ascending=False)
        .reset_index(drop=True)
    )
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))
    ranking_path = comparison_dir / "seed42_spectral_tuned_ranking.csv"
    ranking.to_csv(ranking_path, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print("\nRanking by mean MCC across 8 datasets (seed 42):")
        print(ranking.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    _print_per_dataset_comparison(all_rows)
    print(f"\nTuned SpectralNFST: {tuned_path}")
    print(f"All comparison rows: {all_rows_path}")
    print(f"Ranking: {ranking_path}")
    if excluded:
        print("Excluded incomplete files:")
        for reason in excluded:
            print(f"  - {reason}")
    if missing_models:
        print(f"Missing competitor CSVs: {', '.join(missing_models)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
