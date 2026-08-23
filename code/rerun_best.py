#!/usr/bin/env python3
"""Rerun HHH and HHHv2 with the best config for each dataset; produce extended metrics
and a side-by-side comparison with the Excel/paper ground truth.

Best configs (kernel, poly, scaler) were extracted from TongHop_res_new.xlsx per-dataset
sheet by picking the row with highest MCC for each (dataset, model).

Datasets: BoT_IoT_1000, IoTID20_2000, ToN_IoT_1000, CIC_IoT2023_1000 (N-BaIoT skipped —
raw input data not in the project).

Metrics computed: MCC, ACC, TPR/PPV/F1 (macro + weighted), FPR, AUROC (OvR macro+weighted,
OvO macro), binary AUC, confusion matrix, per-class report.
"""

from __future__ import annotations

import json
import resource
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from cpai import HHH, HHHv2
from cpai.datasets import load_dataset
from cpai.metrics import evaluate_extended
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import preprocess


# From the Excel per-dataset sheets, row with max MCC per (dataset, model):
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
    # Edge_IIoTset: no Excel reference. Use paper's typical best (laplacian, QT, poly=-1).
    ("Edge_IIoTset", "HHH"):   dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("Edge_IIoTset", "HHHv2"): dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    # 5G-NIDD: replacing UNSW_NB15 slot. No paper reference. Same default as Edge_IIoTset.
    ("5G_NIDD", "HHH"):     dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
    ("5G_NIDD", "HHHv2"):   dict(kernel="laplacian", poly=-1, scaler="QuantileTransformer", limit=1000),
}

PAPER_BEST = {  # per Excel: MCC, ACC, TPR, FPR, F1
    ("BoT_IoT", "HHH"):     (0.9784, 99.49, 84.26, 0.30, 84.04),
    ("BoT_IoT", "HHHv2"):   (0.9916, 99.77, 99.14, 0.14, 99.17),
    ("IoTID20", "HHH"):     (0.9092, 97.57, 77.78, 1.46, 77.46),
    ("IoTID20", "HHHv2"):   (0.9507, 98.42, 96.05, 0.99, 96.04),
    ("ToN_IoT", "HHH"):     (0.9074, 98.23, 81.44, 1.00, 81.32),
    ("ToN_IoT", "HHHv2"):   (0.9647, 99.24, 96.96, 0.43, 96.76),
    ("CIC_IoT2023", "HHH"): (0.7709, 95.41, 69.41, 2.67, 63.82),
    ("CIC_IoT2023", "HHHv2"): (0.9501, 98.83, 94.25, 0.70, 94.14),
    ("N_BaIoT", "HHH"):     (0.9694, 99.40, 97.28, 0.34, 97.27),
    ("N_BaIoT", "HHHv2"):   (0.9981, 99.96, 99.84, 0.02, 99.83),
    # Edge_IIoTset: no paper reference at all.
    ("Edge_IIoTset", "HHH"):   (None, None, None, None, None),
    ("Edge_IIoTset", "HHHv2"): (None, None, None, None, None),
    # 5G-NIDD: no paper reference.
    ("5G_NIDD", "HHH"):   (None, None, None, None, None),
    ("5G_NIDD", "HHHv2"): (None, None, None, None, None),
}


def _build_model(name: str, kernel: str):
    if name == "HHH":
        return HHH(kernel=kernel)
    if name == "HHHv2":
        return HHHv2(kernel=kernel)
    raise ValueError(f"Unknown model {name}")


def _peak_rss_mb() -> float:
    """Peak resident-set size of THIS process, in megabytes.

    macOS reports ru_maxrss in bytes; Linux reports kilobytes. Normalize both to MB.
    """
    kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_or_bytes / (1024 * 1024) if sys.platform == "darwin" else kb_or_bytes / 1024


def run_one(dataset: str, model_name: str, cfg: dict, seed: int = 42) -> dict:
    t0 = time.time()
    df, limit = load_dataset(dataset, limit=cfg["limit"])
    X_train, y_train, X_test, y_test, encoder = preprocess(
        df, dataset, poly=cfg["poly"], kernel=cfg["kernel"], scaler=cfg["scaler"], seed=seed,
    )
    t_prep = time.time() - t0

    model = _build_model(model_name, cfg["kernel"])
    t0 = time.time()
    model.fit(X_train, y_train)
    t_fit = time.time() - t0

    # Core test-time measurements: decision_function + predict + predict_proba.
    # These are the "inference" operations measured without preprocessing.
    t0 = time.time()
    y_scores_raw = model.decision_function(X_test)  # shape (n,) for HHH, (n, C) for HHHv2
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test, n_classes=len(encoder.classes_))
    except TypeError:
        y_proba = model.predict_proba(X_test)
    t_pred = time.time() - t0

    peak_mb = _peak_rss_mb()

    metrics = evaluate_extended(y_test, y_pred, y_proba, class_names=list(encoder.classes_))
    metrics.update({
        "_dataset": dataset, "_model": model_name,
        "_config": cfg,
        "_n_train": len(y_train), "_n_test": len(y_test),
        "_n_classes": len(encoder.classes_),
        "_classes": list(encoder.classes_),
        "_t_preprocess": t_prep, "_t_fit": t_fit, "_t_predict": t_pred,
        "_peak_rss_mb": peak_mb,
        # Raw scores useful for offline ROC/PR curves. HHH: (n_test,); HHHv2: (n_test, n_classes).
        "y_score_raw": y_scores_raw,
        "y_true": y_test,
    })
    return metrics


def _save_job(key, res):
    out_dir = RESULTS_DIR / "rerun_best"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{key[0]}__{key[1]}.json"
    serialized = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}
    fname.write_text(json.dumps(serialized, indent=2, default=str))


def _worker(args):
    key, cfg = args
    dataset, model = key
    try:
        res = run_one(dataset, model, cfg)
        _save_job(key, res)  # save immediately on success
        return key, res
    except Exception as e:
        err = {"_error": f"{type(e).__name__}: {e}", "_trace": traceback.format_exc(),
               "_dataset": dataset, "_model": model, "_config": cfg}
        _save_job(key, err)  # save error too
        return key, err


def _fmt(v, w=6, prec=2):
    if v is None: return "-".rjust(w)
    if isinstance(v, float):
        return f"{v:{w}.{prec}f}"
    return str(v).rjust(w)


def main() -> int:
    # Launch jobs in parallel — 8 jobs, so use workers=4 to be gentle.
    # Skip any job whose per-job JSON file already exists (resumable).
    all_jobs = list(BEST_CONFIGS.items())
    out_dir = RESULTS_DIR / "rerun_best"
    jobs = [j for j in all_jobs if not (out_dir / f"{j[0][0]}__{j[0][1]}.json").exists()]
    skipped = len(all_jobs) - len(jobs)
    if skipped:
        print(f"[rerun] Skipping {skipped} job(s) with existing JSON files. Delete them to force re-run.")
    workers = 2  # keep RAM in check — each job's kernel matrix + LOF can use several GB
    print(f"[rerun] Launching {len(jobs)} experiments ({workers} workers)...")

    with Pool(processes=workers) as pool:
        results = {}
        for key, res in pool.imap_unordered(_worker, jobs):
            if "_error" in res:
                print(f"  [FAIL] {key}: {res['_error']}")
            else:
                print(f"  [ OK ] {key[0]}/{key[1]}  MCC={res['MCC']:.4f}  "
                      f"ACC={res['ACC']:.2f}  fit={res['_t_fit']:.1f}s  pred={res['_t_predict']:.1f}s")
            results[key] = res

    # Save raw results
    out_dir = RESULTS_DIR / "rerun_best"
    out_dir.mkdir(parents=True, exist_ok=True)
    _serializable = {}
    for key, res in results.items():
        name = f"{key[0]}__{key[1]}"
        res_s = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in res.items()}
        _serializable[name] = res_s
    (out_dir / "results.json").write_text(json.dumps(_serializable, indent=2, default=str))
    print(f"\n[rerun] Raw results saved to {out_dir / 'results.json'}")

    # Comparison table vs paper (Excel)
    print("\n\n" + "=" * 110)
    print("Comparison: our rerun vs paper/Excel best (MCC, ACC, TPR, FPR, F1)")
    print("=" * 110)
    print(f"{'Dataset':<12} {'Model':<6}  {'Source':<9}  {'MCC':>7}  {'ACC':>6}  {'TPR':>6}  {'FPR':>5}  {'F1':>6}  {'AUROC':>7}")
    print("-" * 110)
    for key in jobs:
        res = results[key[0]]
        ds, m = key[0]
        paper = PAPER_BEST.get(key[0])
        if "_error" in res:
            print(f"{ds:<12} {m:<6}  ERROR: {res['_error'][:60]}")
            continue
        auroc = res.get("AUROC OvR Macro")
        have_paper = paper is not None and paper[0] is not None
        if have_paper:
            print(f"{ds:<12} {m:<6}  {'paper':<9}  {_fmt(paper[0], 7, 4)}  {_fmt(paper[1])}  {_fmt(paper[2])}  {_fmt(paper[3], 5, 2)}  {_fmt(paper[4])}  {'-':>7}")
        print(f"{ds:<12} {m:<6}  {'ours':<9}  {_fmt(res['MCC'], 7, 4)}  {_fmt(res['ACC'])}  {_fmt(res['TPR Macro'])}  {_fmt(res['FPR'], 5, 2)}  {_fmt(res['F1 Macro'])}  {_fmt(auroc, 7, 2)}")
        if have_paper:
            d_mcc = res['MCC'] - paper[0]
            d_acc = res['ACC'] - paper[1]
            d_f1 = res['F1 Macro'] - paper[4]
            print(f"{'':<12} {'':<6}  {'delta':<9}  {_fmt(d_mcc, 7, 4):>7}  {_fmt(d_acc, 6, 2):>6}  {_fmt(res['TPR Macro']-paper[2], 6, 2):>6}  {_fmt(res['FPR']-paper[3], 5, 2):>5}  {_fmt(d_f1, 6, 2):>6}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
