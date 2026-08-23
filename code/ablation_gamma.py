#!/usr/bin/env python3
"""γ-heuristic ablation study — applies to rbf / poly / sigmoid kernels only.

Compare sklearn's default `γ = 1/n_features` against the paper's Section 3.4
heuristic `γ = 1/(2σ²)`, σ = median over samples of the mean k-NN distance (k=5).

For each (dataset, model, kernel) triple we fix the best (scaler, poly) from Scene
B and vary only `gamma ∈ {default, heuristic}`. Single seed (42). 84 runs total
(7 datasets × 2 models × 3 kernels × 2 settings).
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
KERNELS = ["rbf", "poly", "sigmoid"]
GAMMA_SETTINGS = [("default", None), ("heuristic", "heuristic")]
SEED = 42

ABL_DIR = RESULTS_DIR / "ablation_gamma"
RUNS_DIR = ABL_DIR / "runs"
SUMMARY_CSV = ABL_DIR / "summary.csv"


def _peak_rss_mb() -> float:
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / (1024 * 1024) if sys.platform == "darwin" else kb / 1024


def _best_scaler_poly(model: str) -> dict[tuple[str, str], tuple[str, int]]:
    """For each (dataset, kernel) with this model, pick (scaler, poly) maximizing mean MCC."""
    df = pd.read_csv(RESULTS_DIR / "scene_b" / "summary.csv", keep_default_na=False)
    df["poly"] = df["poly"].astype(int)
    df["MCC"] = pd.to_numeric(df["MCC"], errors="coerce")
    out = {}
    for ds in DATASETS:
        for k in KERNELS:
            sub = df[(df.dataset == ds) & (df.model == model) & (df.kernel == k)]
            agg = sub.groupby(["scaler", "poly"])["MCC"].mean().reset_index()
            best = agg.loc[agg["MCC"].idxmax()]
            out[(ds, k)] = (best["scaler"], int(best["poly"]))
    return out


def _run_id(dataset: str, model: str, kernel: str, gamma_label: str) -> str:
    return f"{dataset}__{model}__{kernel}__gamma_{gamma_label}"


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    serial = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in data.items()}
    with tmp.open("w") as f:
        json.dump(serial, f, indent=2, default=str)
        f.flush(); os.fsync(f.fileno())
    tmp.rename(path)


def run_one(dataset: str, model_name: str, kernel: str, gamma_label: str,
            gamma_val, scaler: str, poly: int) -> dict:
    run_id = _run_id(dataset, model_name, kernel, gamma_label)
    out = RUNS_DIR / f"{run_id}.json"
    if out.exists():
        return {"_status": "skip"}
    try:
        t0 = time.time()
        df, limit = load_dataset(dataset, limit=_DEFAULT_LIMIT[dataset])
        X_tr, y_tr, X_te, y_te, enc = preprocess(
            df, dataset, poly=poly, kernel=kernel, scaler=scaler, seed=SEED)
        t_prep = time.time() - t0

        cls = HHH if model_name == "HHH" else HHHv2
        model = cls(kernel=kernel, gamma=gamma_val)

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
            "_config": {"kernel": kernel, "scaler": scaler, "poly": poly,
                        "gamma_setting": gamma_label,
                        "gamma_used": model._gamma_used,
                        "seed": SEED, "limit": limit},
            "_n_train": len(y_tr), "_n_test": len(y_te),
            "_n_classes": len(enc.classes_), "_classes": list(enc.classes_),
            "_t_preprocess": t_prep, "_t_fit": t_fit, "_t_predict": t_pred,
            "_peak_rss_mb": peak_mb,
            "y_score_raw": y_scores, "y_true": y_te,
        })
        _atomic_write(out, metrics)
        return {"_status": "ok", "MCC": metrics["MCC"], "ACC": metrics["ACC"],
                "F1": metrics["F1 Macro"], "gamma_used": model._gamma_used,
                "t_fit": t_fit, "peak_mb": peak_mb}
    except Exception as exc:
        err = {"_run_id": run_id, "_status": "error",
               "_error": f"{type(exc).__name__}: {exc}",
               "_trace": traceback.format_exc(),
               "_dataset": dataset, "_model": model_name,
               "_config": {"kernel": kernel, "scaler": scaler, "poly": poly,
                           "gamma_setting": gamma_label, "seed": SEED}}
        _atomic_write(out, err)
        return {"_status": "error", "_error": err["_error"]}


def write_summary() -> None:
    rows = []
    for jf in sorted(RUNS_DIR.glob("*.json")):
        d = json.loads(jf.read_text())
        c = d.get("_config") or {}
        if d.get("_status") == "error" or "_error" in d:
            rows.append({"dataset": d.get("_dataset", ""), "model": d.get("_model", ""),
                         "kernel": c.get("kernel", ""), "scaler": c.get("scaler", ""),
                         "poly": c.get("poly", ""), "gamma_setting": c.get("gamma_setting", ""),
                         "status": "error"})
            continue
        rows.append({
            "dataset": d["_dataset"], "model": d["_model"],
            "kernel": c.get("kernel"), "scaler": c.get("scaler"),
            "poly": c.get("poly"),
            "gamma_setting": c.get("gamma_setting"),
            "gamma_used": c.get("gamma_used"),
            "n_train": d.get("_n_train"), "n_classes": d.get("_n_classes"),
            "MCC": d.get("MCC"), "ACC": d.get("ACC"),
            "TPR_macro": d.get("TPR Macro"), "FPR": d.get("FPR"),
            "PPV_macro": d.get("PPV Macro"), "F1_macro": d.get("F1 Macro"),
            "AUROC_OvR_macro": d.get("AUROC OvR Macro"),
            "AUROC_binary": d.get("AUROC Binary"),
            "t_fit_s": d.get("_t_fit"), "t_predict_s": d.get("_t_predict"),
            "peak_rss_mb": d.get("_peak_rss_mb"), "status": "ok",
        })
    pd.DataFrame(rows).to_csv(SUMMARY_CSV, index=False)
    print(f"[ablation γ] wrote {SUMMARY_CSV} ({len(rows)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    scaler_poly = {m: _best_scaler_poly(m) for m in args.models}

    jobs = []
    for ds in args.datasets:
        for m in args.models:
            for k in KERNELS:
                s, p = scaler_poly[m][(ds, k)]
                for label, val in GAMMA_SETTINGS:
                    jobs.append((ds, m, k, s, p, label, val))

    print(f"[ablation γ] {len(jobs)} runs (7 ds × 2 models × 3 kernels × 2 γ)")
    if args.dry_run:
        for j in jobs[:12]:
            print(f"  [plan] {j[0]:<14} {j[1]:<6} {j[2]:<8} (sc={j[3]}, poly={j[4]}, γ={j[5]})")
        return 0

    t_wall = time.time()
    ok = err = 0
    for i, (ds, m, k, s, p, lbl, val) in enumerate(jobs, 1):
        r = run_one(ds, m, k, lbl, val, s, p)
        if r["_status"] == "ok":
            ok += 1
            print(f"  [OK  {i:>3}/{len(jobs)}] {ds:<14} {m:<6} {k:<8} γ={lbl:<9}  "
                  f"MCC={r['MCC']:.4f} F1={r['F1']:.2f}  "
                  f"γ_used={r['gamma_used']}  fit={r['t_fit']:.1f}s", flush=True)
        elif r["_status"] == "error":
            err += 1
            print(f"  [ERR {i:>3}/{len(jobs)}] {ds:<14} {m:<6} {k:<8} γ={lbl:<9}  "
                  f"{r['_error'][:80]}", flush=True)
    write_summary()
    print(f"\n[ablation γ] {ok} OK / {err} ERR in {(time.time()-t_wall)/60:.1f} min")
    return 0 if err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
