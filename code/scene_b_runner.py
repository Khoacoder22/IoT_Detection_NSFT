#!/usr/bin/env python3
"""Scene B: full-grid multiseed runner for HHH + HHHv2 on all registered datasets.

Grid dimensions per dataset:
  * models:   HHH, HHHv2
  * kernels:  None, linear, poly, rbf, sigmoid, abel, laplacian, sobolev
  * scalers:  QuantileTransformer, StandardScaler, MinMaxScaler
  * polys:    [-1, 0, 2] if dataset has <30 columns else [-1, 0]
  * seeds:    [42, 43, 44, 45, 46]  (5 seeds)

One JSON per (dataset, model, kernel, scaler, poly, seed) under results/scene_b/runs/.
The presence of a JSON file means the run is done — re-running the script skips it.

Writes are atomic (write to .tmp then rename) so a crash cannot leave half-written files.

Usage:
    python3 code/scene_b_runner.py                  # all datasets
    python3 code/scene_b_runner.py --datasets BoT_IoT IoTID20
    python3 code/scene_b_runner.py --workers 4      # override default (2)
    python3 code/scene_b_runner.py --dry-run        # print what WOULD run
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from cpai import HHH, HHHv2
from cpai.datasets import load_dataset, _DEFAULT_LIMIT
from cpai.metrics import evaluate_extended
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import preprocess


KERNELS = ["None", "linear", "poly", "rbf", "sigmoid", "abel", "laplacian", "sobolev"]
SCALERS = ["QuantileTransformer", "StandardScaler", "MinMaxScaler"]
MODELS = ["HHH", "HHHv2"]
SEEDS = [42, 43, 44, 45, 46]

# Datasets in the current roster (UNSW_NB15 retired from BEST_CONFIGS earlier).
DATASETS = ["BoT_IoT", "IoTID20", "ToN_IoT", "N_BaIoT", "CIC_IoT2023", "Edge_IIoTset", "5G_NIDD"]

SCENE_B_DIR = RESULTS_DIR / "scene_b"
RUNS_DIR = SCENE_B_DIR / "runs"


def _peak_rss_mb() -> float:
    kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_or_bytes / (1024 * 1024) if sys.platform == "darwin" else kb_or_bytes / 1024


def _polys_for(dataset: str) -> list[int]:
    """Notebook rule: if df has <30 columns use [-1, 0, 2], else [-1, 0]."""
    df, _ = load_dataset(dataset, limit=_DEFAULT_LIMIT[dataset])
    return [-1, 0, 2] if df.shape[1] < 30 else [-1, 0]


def _run_id(dataset, model, kernel, scaler, poly, seed) -> str:
    """Stable filename that encodes the full config."""
    return f"{dataset}__{model}__poly{poly}__{kernel}__{scaler}__seed{seed}"


def _out_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _build_model(name: str, kernel: str):
    k = None if kernel == "None" else kernel
    if name == "HHH":
        return HHH(kernel=k)
    if name == "HHHv2":
        return HHHv2(kernel=k)
    raise ValueError(name)


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write to .tmp, fsync, rename. Guarantees no half-written files after crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    serialized = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data.items()}
    with tmp.open("w") as f:
        json.dump(serialized, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def _run_single(cfg: dict) -> tuple[str, dict]:
    """Worker: run one config, save atomic JSON, return summary tuple."""
    run_id = cfg["_run_id"]
    out = _out_path(run_id)
    if out.exists():
        return run_id, {"_status": "skip"}

    try:
        dataset = cfg["dataset"]
        limit = _DEFAULT_LIMIT[dataset]
        seed = cfg["seed"]
        kernel = cfg["kernel"]
        k_arg = None if kernel == "None" else kernel

        t0 = time.time()
        df, _ = load_dataset(dataset, limit=limit)
        X_train, y_train, X_test, y_test, encoder = preprocess(
            df, dataset, poly=cfg["poly"], kernel=k_arg, scaler=cfg["scaler"], seed=seed,
        )
        t_prep = time.time() - t0

        model = _build_model(cfg["model"], kernel)
        t0 = time.time()
        model.fit(X_train, y_train)
        t_fit = time.time() - t0

        t0 = time.time()
        y_scores_raw = model.decision_function(X_test)
        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test, n_classes=len(encoder.classes_))
        except TypeError:
            y_proba = model.predict_proba(X_test)
        t_pred = time.time() - t0

        peak_mb = _peak_rss_mb()

        metrics = evaluate_extended(y_test, y_pred, y_proba, class_names=list(encoder.classes_))
        metrics.update({
            "_run_id": run_id,
            "_dataset": dataset,
            "_model": cfg["model"],
            "_config": {
                "kernel": kernel, "scaler": cfg["scaler"],
                "poly": cfg["poly"], "limit": limit, "seed": seed,
            },
            "_n_train": len(y_train), "_n_test": len(y_test),
            "_n_classes": len(encoder.classes_),
            "_classes": list(encoder.classes_),
            "_t_preprocess": t_prep, "_t_fit": t_fit, "_t_predict": t_pred,
            "_peak_rss_mb": peak_mb,
            "y_score_raw": y_scores_raw,
            "y_true": y_test,
        })
        _atomic_write_json(out, metrics)
        return run_id, {
            "_status": "ok", "MCC": metrics["MCC"], "ACC": metrics["ACC"],
            "t_fit": t_fit, "t_pred": t_pred, "peak_mb": peak_mb,
        }
    except Exception as exc:
        err = {
            "_run_id": run_id, "_status": "error",
            "_error": f"{type(exc).__name__}: {exc}",
            "_trace": traceback.format_exc(),
            "_config": cfg,
        }
        _atomic_write_json(out, err)
        return run_id, err


def enumerate_configs(datasets: list[str]) -> list[dict]:
    configs = []
    for ds in datasets:
        polys = _polys_for(ds)
        for model in MODELS:
            for kernel in KERNELS:
                for scaler in SCALERS:
                    for poly in polys:
                        for seed in SEEDS:
                            cfg = {
                                "dataset": ds, "model": model, "kernel": kernel,
                                "scaler": scaler, "poly": poly, "seed": seed,
                            }
                            cfg["_run_id"] = _run_id(ds, model, kernel, scaler, poly, seed)
                            configs.append(cfg)
    return configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS, help="Datasets to run")
    ap.add_argument("--workers", type=int, default=2, help="Parallel workers")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    configs = enumerate_configs(args.datasets)
    done = [c for c in configs if _out_path(c["_run_id"]).exists()]
    pending = [c for c in configs if not _out_path(c["_run_id"]).exists()]

    print(f"[scene_b] datasets={args.datasets}")
    print(f"[scene_b] total configs: {len(configs)}   done: {len(done)}   pending: {len(pending)}")
    if args.dry_run:
        for c in pending[:10]:
            print(f"    [pending] {c['_run_id']}")
        if len(pending) > 10:
            print(f"    ... and {len(pending)-10} more")
        return 0

    if not pending:
        print("[scene_b] nothing to do")
        return 0

    # Ignore SIGINT in workers (main handles it); lets in-flight jobs finish cleanly.
    original_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        pool = Pool(processes=args.workers)
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    t_wall = time.time()
    ok = err = 0
    try:
        for i, (run_id, summary) in enumerate(pool.imap_unordered(_run_single, pending), 1):
            status = summary.get("_status")
            if status == "ok":
                ok += 1
                print(f"  [OK {i}/{len(pending)}] {run_id}  MCC={summary['MCC']:.4f}  "
                      f"fit={summary['t_fit']:.1f}s  peak={summary['peak_mb']:.0f}MB", flush=True)
            elif status == "error":
                err += 1
                print(f"  [ERR {i}/{len(pending)}] {run_id}  {summary['_error'][:80]}", flush=True)
            # skip = already existed; shouldn't happen given the pending filter
    except KeyboardInterrupt:
        print("\n[scene_b] SIGINT — terminating workers, in-flight JSONs preserved")
        pool.terminate()
        pool.join()
        return 130
    else:
        pool.close()
        pool.join()

    wall = time.time() - t_wall
    print(f"\n[scene_b] completed {ok} OK / {err} ERR in {wall/60:.1f} min "
          f"(avg {wall/max(ok+err,1):.1f}s/run)")
    return 0 if err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
