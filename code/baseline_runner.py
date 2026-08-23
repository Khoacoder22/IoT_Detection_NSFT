#!/usr/bin/env python3
"""Baseline runner: every model × dataset × seed with a single default config.

Config (fixed across all runs):
    * scaler = QuantileTransformer
    * poly   = -1  (no polynomial feature expansion)
    * kernel = None  (baselines do not use CPAI's kernel machinery)

Grid:
    11 baselines  ×  7 datasets  ×  5 seeds  =  385 runs

Serial (no multiprocessing Pool) — TabNet / FTTransformer get segfaults when
forked on macOS so we run one job at a time. Per-run JSONs are atomic, so
killing and restarting the script is always safe (it resumes).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

# Preload torch BEFORE scipy (same macOS libomp workaround as cpai/__init__.py).
try:
    import torch  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from cpai.baselines import build_baseline, BASELINE_NAMES
from cpai.datasets import load_dataset, _DEFAULT_LIMIT
from cpai.metrics import evaluate_extended
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import preprocess


DATASETS = ["BoT_IoT", "IoTID20", "ToN_IoT", "N_BaIoT", "CIC_IoT2023", "Edge_IIoTset", "5G_NIDD"]
SEEDS = [42, 43, 44, 45, 46]

DEFAULT_SCALER = "QuantileTransformer"
DEFAULT_POLY = -1

BASELINES_DIR = RESULTS_DIR / "baselines_default"
RUNS_DIR = BASELINES_DIR / "runs"


def _peak_rss_mb() -> float:
    kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_or_bytes / (1024 * 1024) if sys.platform == "darwin" else kb_or_bytes / 1024


def _run_id(dataset: str, model: str, seed: int) -> str:
    return f"{dataset}__{model}__seed{seed}"


def _out_path(run_id: str) -> Path:
    return RUNS_DIR / f"{run_id}.json"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    serialized = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data.items()}
    with tmp.open("w") as f:
        json.dump(serialized, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def run_single(dataset: str, model_name: str, seed: int) -> dict:
    run_id = _run_id(dataset, model_name, seed)
    out = _out_path(run_id)
    if out.exists():
        return {"_status": "skip"}

    try:
        limit = _DEFAULT_LIMIT[dataset]

        t0 = time.time()
        df, _ = load_dataset(dataset, limit=limit)
        X_train, y_train, X_test, y_test, encoder = preprocess(
            df, dataset, poly=DEFAULT_POLY, kernel=None, scaler=DEFAULT_SCALER, seed=seed,
        )
        t_prep = time.time() - t0

        model = build_baseline(model_name)

        t0 = time.time()
        model.fit(X_train, y_train)
        t_fit = time.time() - t0

        t0 = time.time()
        try:
            y_proba = model.predict_proba(X_test)
        except (AttributeError, NotImplementedError):
            y_proba = None
        y_pred = model.predict(X_test)
        t_pred = time.time() - t0

        peak_mb = _peak_rss_mb()

        metrics = evaluate_extended(y_test, y_pred, y_proba, class_names=list(encoder.classes_))
        metrics.update({
            "_run_id": run_id,
            "_dataset": dataset,
            "_model": model_name,
            "_config": {
                "scaler": DEFAULT_SCALER, "poly": DEFAULT_POLY, "seed": seed, "limit": limit,
            },
            "_n_train": len(y_train), "_n_test": len(y_test),
            "_n_classes": len(encoder.classes_),
            "_classes": list(encoder.classes_),
            "_t_preprocess": t_prep, "_t_fit": t_fit, "_t_predict": t_pred,
            "_peak_rss_mb": peak_mb,
            "y_true": y_test,
        })
        if y_proba is not None:
            # Raw per-class probabilities; same shape as HHHv2 y_score_raw for comparison.
            metrics["y_score_raw"] = y_proba
        _atomic_write_json(out, metrics)
        return {
            "_status": "ok", "MCC": metrics["MCC"], "ACC": metrics["ACC"],
            "F1_macro": metrics["F1 Macro"], "t_fit": t_fit, "t_pred": t_pred,
            "peak_mb": peak_mb,
        }
    except Exception as exc:
        err = {
            "_run_id": run_id, "_status": "error",
            "_error": f"{type(exc).__name__}: {exc}",
            "_trace": traceback.format_exc(),
            "_dataset": dataset, "_model": model_name,
            "_config": {"scaler": DEFAULT_SCALER, "poly": DEFAULT_POLY, "seed": seed},
        }
        _atomic_write_json(out, err)
        return {"_status": "error", "_error": err["_error"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--models", nargs="*", default=list(BASELINE_NAMES))
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(ds, m, s) for ds in args.datasets for m in args.models for s in args.seeds]
    done = [j for j in jobs if _out_path(_run_id(*j)).exists()]
    pending = [j for j in jobs if not _out_path(_run_id(*j)).exists()]

    print(f"[baselines] total={len(jobs)} done={len(done)} pending={len(pending)}")
    if args.dry_run:
        return 0
    if not pending:
        print("[baselines] nothing to do")
        return 0

    t_wall = time.time()
    ok = err = 0
    for i, (ds, m, s) in enumerate(pending, 1):
        t0 = time.time()
        r = run_single(ds, m, s)
        dt = time.time() - t0
        if r["_status"] == "ok":
            ok += 1
            print(f"  [OK  {i}/{len(pending)}] {ds:<14} {m:<14} seed={s}  "
                  f"MCC={r['MCC']:.4f} F1={r['F1_macro']:.2f}  fit={r['t_fit']:.1f}s  peak={r['peak_mb']:.0f}MB  "
                  f"wall={dt:.0f}s", flush=True)
        elif r["_status"] == "error":
            err += 1
            print(f"  [ERR {i}/{len(pending)}] {ds:<14} {m:<14} seed={s}  "
                  f"{r['_error'][:80]}", flush=True)

    wall = time.time() - t_wall
    print(f"\n[baselines] {ok} OK / {err} ERR in {wall/60:.1f} min (avg {wall/max(ok+err,1):.1f}s/run)")
    return 0 if err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
