#!/usr/bin/env python3
"""Parallel orchestrator — fan out (dataset x model x kernel x scaler x poly) jobs.

Each job runs in its own process. Results append to one CSV per (dataset, model-group).

Examples:
    # Reproduce paper Tables 2/3 (CPAI-OvA and CPAI-OvR across 8 kernels, 5 datasets):
    python3 run_experiments.py --preset tables_2_3 --workers 8

    # Just run baselines for one dataset:
    python3 run_experiments.py --datasets BoT_IoT --models KNN LDA NuSVC GauNB

    # Full replication (everything):
    python3 run_experiments.py --preset full --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpai.baselines import BASELINE_NAMES
from cpai.datasets import DATASETS
from cpai.kernels import KERNEL_CHOICES
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import SCALERS

from train_evaluate import append_row, run_one

CPAI_MODELS = ("HHH", "HHHv2", "KNFST", "SpectralNFST")
ALL_MODELS = (*CPAI_MODELS, *BASELINE_NAMES)


@dataclass(frozen=True)
class Job:
    dataset: str
    model: str
    kernel: str | None
    scaler: str
    poly: int
    output: Path
    components_per_class: int = 2


def _worker(job: Job) -> tuple[Job, dict | str]:
    try:
        row = run_one(
            job.dataset, job.model, job.kernel, job.scaler, job.poly,
            components_per_class=job.components_per_class,
        )
        append_row(row, job.output)
        return job, row
    except Exception as e:  # pragma: no cover
        return job, f"ERROR: {type(e).__name__}: {e}"


def build_jobs(
    datasets: list[str],
    models: list[str],
    kernels: list[str | None],
    scalers: list[str],
    polys: list[int],
    output_root: Path,
    components_per_class: int = 2,
) -> list[Job]:
    jobs: list[Job] = []
    for ds, model, kernel, scaler, poly in itertools.product(
        datasets, models, kernels, scalers, polys
    ):
        # KNFST always uses a kernel; skip None
        if model in ("KNFST", "SpectralNFST") and (kernel is None or kernel == "none"):
            continue
        # baselines don't use kernel/poly axis — run only once per (dataset, scaler)
        if model in BASELINE_NAMES and (kernel not in (None, "none") or poly != -1):
            continue
        out = output_root / ds / f"{model}.csv"
        jobs.append(Job(ds, model, kernel, scaler, poly, out, components_per_class))
    return jobs


PRESETS = {
    "tables_2_3": dict(
        models=["HHH", "HHHv2"],
        kernels=["none", "rbf", "poly", "linear", "sigmoid", "abel", "laplacian", "sobolev","rff"],
        scalers=["QuantileTransformer", "StandardScaler", "MinMaxScaler"],
        polys=[-1, 0, 2],
    ),
    "tables_4_8": dict(
        models=list(BASELINE_NAMES) + ["KNFST"],
        kernels=["rbf", "rff"],
        scalers=["QuantileTransformer"],
        polys=[-1],
    ),
    "full": dict(
        models=list(ALL_MODELS),
        kernels=list(KERNEL_CHOICES) + "rff",
        scalers=["QuantileTransformer", "StandardScaler", "MinMaxScaler"],
        polys=[-1, 0, 2],
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), help="Preset job grid")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=list(ALL_MODELS))
    parser.add_argument("--kernels", nargs="+", choices=list(KERNEL_CHOICES))
    parser.add_argument("--scalers", nargs="+", choices=list(SCALERS))
    parser.add_argument("--polys", nargs="+", type=int, choices=[-1, 0, 2, 3])
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    parser.add_argument(
        "--components-per-class", type=int, default=2,
        help="SpectralNFST target components Q_j for every class (default: 2).",
    )
    parser.add_argument(
        "--output-root", type=Path, default=RESULTS_DIR / "runs",
        help="Directory to write per-(dataset, model) CSVs into.",
    )
    args = parser.parse_args()

    cfg: dict = {"models": ALL_MODELS, "kernels": ["none", "rbf", "poly", "linear", "sigmoid"],
                 "scalers": ["QuantileTransformer"], "polys": [-1]}
    if args.preset:
        cfg.update(PRESETS[args.preset])
    for key in ("models", "kernels", "scalers", "polys"):
        override = getattr(args, key)
        if override is not None:
            cfg[key] = override

    jobs = build_jobs(
        datasets=args.datasets,
        models=cfg["models"],
        kernels=cfg["kernels"],
        scalers=cfg["scalers"],
        polys=cfg["polys"],
        output_root=args.output_root,
        components_per_class=args.components_per_class,
    )
    print(f"[orchestrator] {len(jobs)} jobs, {args.workers} workers")

    t0 = time.time()
    done = failed = 0
    with Pool(processes=args.workers) as pool:
        for job, result in pool.imap_unordered(_worker, jobs):
            if isinstance(result, str):
                failed += 1
                print(f"  [FAIL] {job.dataset}/{job.model} {job.kernel}/{job.scaler}/poly={job.poly}: {result}")
            else:
                done += 1
                print(f"  [ OK ] {job.dataset}/{job.model} {job.kernel}/{job.scaler}/poly={job.poly} "
                      f"MCC={result['MCC']:.4f}")

    print(f"[orchestrator] done={done} failed={failed} elapsed={time.time() - t0:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
