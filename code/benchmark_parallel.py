#!/usr/bin/env python3
"""Benchmark 14 HHH/HHHv2 best-config jobs at a given worker count.

Run directly as a top-level Python process for each worker count (no subprocess
nesting — avoids macOS multiprocessing.Pool deadlocks when started inside another
Python subprocess):

    python3 code/benchmark_parallel.py --workers 1
    python3 code/benchmark_parallel.py --workers 2
    python3 code/benchmark_parallel.py --workers 4

Prints a single RESULT: line with wall-time, per-worker peak RSS, and n_ok.
Bash-level wrapper aggregates the three runs.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import resource
import sys
import time
import traceback
import warnings
from pathlib import Path

# Preload torch before scipy (macOS libomp)
try:
    import torch  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

BEST_CONFIGS = {
    ("BoT_IoT", "HHH"):     dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("BoT_IoT", "HHHv2"):   dict(kernel="laplacian", poly=-1, scaler="MinMaxScaler",         limit=1000),
    ("IoTID20", "HHH"):     dict(kernel="laplacian", poly=0,  scaler="MinMaxScaler",         limit=2000),
    ("IoTID20", "HHHv2"):   dict(kernel="laplacian", poly=0,  scaler="MinMaxScaler",         limit=2000),
    ("ToN_IoT", "HHH"):     dict(kernel="laplacian", poly=0,  scaler="QuantileTransformer", limit=1000),
    ("ToN_IoT", "HHHv2"):   dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("CIC_IoT2023", "HHH"): dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("CIC_IoT2023", "HHHv2"): dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("N_BaIoT", "HHH"):     dict(kernel="rbf",       poly=-1, scaler="QuantileTransformer", limit=1000),
    ("N_BaIoT", "HHHv2"):   dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("Edge_IIoTset", "HHH"):   dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("Edge_IIoTset", "HHHv2"): dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("5G_NIDD", "HHH"):     dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("5G_NIDD", "HHHv2"):   dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
}


def _run(args):
    (ds, mdl), cfg = args
    try:
        from cpai import HHH, HHHv2
        from cpai.datasets import load_dataset
        from cpai.metrics import calc_index
        from cpai.preprocessing import preprocess
        df, _ = load_dataset(ds, limit=cfg["limit"])
        X_tr, y_tr, X_te, y_te, enc = preprocess(
            df, ds, poly=cfg["poly"], kernel=cfg["kernel"], scaler=cfg["scaler"], seed=42)
        t0 = time.time()
        m = HHH(kernel=cfg["kernel"]) if mdl == "HHH" else HHHv2(kernel=cfg["kernel"])
        m.fit(X_tr, y_tr)
        t_fit = time.time() - t0
        y_pred = m.predict(X_te)
        mcc = calc_index(y_te, y_pred)[0]
        kb_or_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mb = kb_or_b / (1024 * 1024) if sys.platform == "darwin" else kb_or_b / 1024
        return {"ds": ds, "model": mdl, "fit_s": t_fit, "peak_mb": peak_mb, "mcc": mcc}
    except Exception as exc:
        return {"ds": ds, "model": mdl, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    args = ap.parse_args()

    jobs = list(BEST_CONFIGS.items())
    t0 = time.time()
    if args.workers == 1:
        results = [_run(j) for j in jobs]
    else:
        # Explicit spawn context — avoids fork-after-import issues on macOS.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            results = list(pool.imap_unordered(_run, jobs))
    wall = time.time() - t0

    fits = [r["fit_s"] for r in results if "fit_s" in r]
    peaks = [r["peak_mb"] for r in results if "peak_mb" in r]
    errs = [r for r in results if "error" in r]

    kb_or_b = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    parent_peak = kb_or_b / (1024 * 1024) if sys.platform == "darwin" else kb_or_b / 1024

    kb_or_b_c = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    children_peak = kb_or_b_c / (1024 * 1024) if sys.platform == "darwin" else kb_or_b_c / 1024

    summary = {
        "workers": args.workers, "wall_s": wall,
        "sum_fit_s": sum(fits) if fits else 0,
        "max_peak_mb_per_worker": max(peaks) if peaks else 0,
        "mean_peak_mb_per_worker": sum(peaks) / len(peaks) if peaks else 0,
        "parent_peak_mb": parent_peak,
        "children_peak_mb_cumulative": children_peak,
        "n_ok": len(fits), "n_err": len(errs),
    }
    print("RESULT:" + json.dumps(summary))
    if errs:
        print("ERRORS:", errs[:3])
    return 0 if not errs else 2


if __name__ == "__main__":
    sys.exit(main())
