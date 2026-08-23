#!/usr/bin/env python3
"""Ridge λ ablation study for HHH (CPAI-OvA) and HHHv2 (CPAI-OvR).

For each (dataset, model) we reuse the best `(kernel, scaler, poly)` configuration
from Scene B (picked by mean MCC across the five seeds) and sweep the ridge
λ ∈ {1e-9, 1e-7, 1e-5, 1e-3, 1e-1}. Single seed (42) per user request.

Grid: 7 datasets × 2 models × 5 λ = 70 runs.

Per-run JSONs land in results/ablation_ridge/runs/<run_id>.json; the `summary.csv`
is regenerated at the end.
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

# Preload torch before scipy (macOS libomp)
try:
    import torch  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from cpai import HHH, HHHv2
from cpai.datasets import load_dataset, _DEFAULT_LIMIT
from cpai.metrics import evaluate_extended
from cpai.paths import RESULTS_DIR
from cpai.preprocessing import preprocess


DATASETS = ["BoT_IoT", "IoTID20", "ToN_IoT", "N_BaIoT", "CIC_IoT2023", "Edge_IIoTset", "5G_NIDD"]
MODELS = ["HHH", "HHHv2"]
LAMBDAS = [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]
SEED = 42

ABL_DIR = RESULTS_DIR / "ablation_ridge"
RUNS_DIR = ABL_DIR / "runs"
SUMMARY_CSV = ABL_DIR / "summary.csv"


def _peak_rss_mb() -> float:
    kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_or_bytes / (1024 * 1024) if sys.platform == "darwin" else kb_or_bytes / 1024


def _best_config_from_scene_b(model: str) -> dict[str, dict]:
    """Load Scene B summary and pick best (kernel, scaler, poly) per dataset for `model`."""
    df = pd.read_csv(RESULTS_DIR / "scene_b" / "summary.csv", keep_default_na=False)
    for col in ["MCC"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["poly"] = df["poly"].astype(int)

    out = {}
    for ds in DATASETS:
        sub = df[(df["dataset"] == ds) & (df["model"] == model)]
        agg = sub.groupby(["kernel", "scaler", "poly"])["MCC"].mean().reset_index()
        best = agg.loc[agg["MCC"].idxmax()]
        out[ds] = {
            "kernel": best["kernel"], "scaler": best["scaler"],
            "poly": int(best["poly"]),
            "reference_MCC_5seeds": float(best["MCC"]),
        }
    return out


def _run_id(dataset: str, model: str, ridge: float) -> str:
    return f"{dataset}__{model}__ridge{ridge:g}"


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    serial = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data.items()}
    with tmp.open("w") as f:
        json.dump(serial, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(path)


def run_one(dataset: str, model_name: str, ridge: float, best_cfg: dict) -> dict:
    run_id = _run_id(dataset, model_name, ridge)
    out = RUNS_DIR / f"{run_id}.json"
    if out.exists():
        return {"_status": "skip"}
    try:
        kernel_str = best_cfg["kernel"]
        k_arg = None if kernel_str == "None" else kernel_str

        t0 = time.time()
        df, limit = load_dataset(dataset, limit=_DEFAULT_LIMIT[dataset])
        X_tr, y_tr, X_te, y_te, enc = preprocess(
            df, dataset, poly=best_cfg["poly"], kernel=k_arg, scaler=best_cfg["scaler"], seed=SEED,
        )
        t_prep = time.time() - t0

        if model_name == "HHH":
            model = HHH(kernel=k_arg, ridge=ridge)
        else:
            model = HHHv2(kernel=k_arg, ridge=ridge)

        t0 = time.time()
        model.fit(X_tr, y_tr)
        t_fit = time.time() - t0

        t0 = time.time()
        y_scores = model.decision_function(X_te)
        y_pred = model.predict(X_te)
        try:
            y_proba = model.predict_proba(X_te, n_classes=len(enc.classes_))
        except TypeError:
            y_proba = model.predict_proba(X_te)
        t_pred = time.time() - t0

        peak_mb = _peak_rss_mb()
        metrics = evaluate_extended(y_te, y_pred, y_proba, class_names=list(enc.classes_))
        metrics.update({
            "_run_id": run_id, "_dataset": dataset, "_model": model_name,
            "_config": {"kernel": kernel_str, "scaler": best_cfg["scaler"],
                        "poly": best_cfg["poly"], "ridge": ridge, "seed": SEED,
                        "limit": limit},
            "_n_train": len(y_tr), "_n_test": len(y_te), "_n_classes": len(enc.classes_),
            "_classes": list(enc.classes_),
            "_t_preprocess": t_prep, "_t_fit": t_fit, "_t_predict": t_pred,
            "_peak_rss_mb": peak_mb,
            "y_score_raw": y_scores, "y_true": y_te,
        })
        _atomic_write_json(out, metrics)
        return {"_status": "ok", "MCC": metrics["MCC"], "ACC": metrics["ACC"],
                "F1": metrics["F1 Macro"], "t_fit": t_fit, "peak_mb": peak_mb}
    except Exception as e:
        err = {"_run_id": run_id, "_status": "error",
               "_error": f"{type(e).__name__}: {e}",
               "_trace": traceback.format_exc(),
               "_dataset": dataset, "_model": model_name, "_config": best_cfg}
        _atomic_write_json(out, err)
        return {"_status": "error", "_error": err["_error"]}


def write_summary() -> None:
    """Flatten the per-run JSONs into a compact CSV."""
    rows = []
    for jf in sorted(RUNS_DIR.glob("*.json")):
        d = json.loads(jf.read_text())
        c = d.get("_config") or {}
        if d.get("_status") == "error" or "_error" in d:
            rows.append({"dataset": d.get("_dataset", ""), "model": d.get("_model", ""),
                         "ridge": c.get("ridge", ""), "status": "error"})
            continue
        rows.append({
            "dataset": d["_dataset"], "model": d["_model"],
            "kernel": c.get("kernel", ""), "scaler": c.get("scaler", ""),
            "poly": c.get("poly", ""), "ridge": c.get("ridge", ""),
            "n_train": d.get("_n_train"), "n_test": d.get("_n_test"),
            "n_classes": d.get("_n_classes"),
            "MCC": d.get("MCC"), "ACC": d.get("ACC"),
            "TPR_macro": d.get("TPR Macro"), "FPR": d.get("FPR"),
            "PPV_macro": d.get("PPV Macro"), "F1_macro": d.get("F1 Macro"),
            "AUROC_OvR_macro": d.get("AUROC OvR Macro"),
            "AUROC_binary": d.get("AUROC Binary"),
            "t_preprocess_s": d.get("_t_preprocess"), "t_fit_s": d.get("_t_fit"),
            "t_predict_s": d.get("_t_predict"),
            "peak_rss_mb": d.get("_peak_rss_mb"),
            "status": "ok",
        })
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"[ablation] wrote {SUMMARY_CSV} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--lambdas", nargs="*", type=float, default=LAMBDAS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    best_by_model = {m: _best_config_from_scene_b(m) for m in args.models}

    jobs = [(ds, m, r) for ds in args.datasets for m in args.models for r in args.lambdas]
    print(f"[ablation] {len(jobs)} jobs (7 datasets × 2 models × 5 λ)")

    if args.dry_run:
        for ds, m, r in jobs[:10]:
            cfg = best_by_model[m][ds]
            print(f"  [plan] {ds:<14} {m:<6} ridge={r:g}  using ({cfg['kernel']}, {cfg['scaler']}, "
                  f"poly={cfg['poly']}) ref-MCC={cfg['reference_MCC_5seeds']:.4f}")
        return 0

    t_wall = time.time()
    ok = err = skip = 0
    for i, (ds, m, r) in enumerate(jobs, 1):
        cfg = best_by_model[m][ds]
        out = run_one(ds, m, r, cfg)
        if out["_status"] == "ok":
            ok += 1
            print(f"  [OK  {i:>3}/{len(jobs)}] {ds:<14} {m:<6} λ={r:<7g} "
                  f"MCC={out['MCC']:.4f}  F1={out['F1']:.2f}  fit={out['t_fit']:.1f}s  "
                  f"peak={out['peak_mb']:.0f}MB", flush=True)
        elif out["_status"] == "skip":
            skip += 1
        else:
            err += 1
            print(f"  [ERR {i:>3}/{len(jobs)}] {ds:<14} {m:<6} λ={r:<7g}  "
                  f"{out['_error'][:80]}", flush=True)

    write_summary()
    print(f"\n[ablation] {ok} OK / {skip} skip / {err} ERR in {(time.time()-t_wall)/60:.1f} min")
    return 0 if err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
