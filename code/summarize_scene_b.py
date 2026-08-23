#!/usr/bin/env python3
"""Aggregate Scene B per-run JSONs into a single CSV for easy analysis.

Reads results/scene_b/runs/*.json, writes results/scene_b/summary.csv with one row per run.
Core metric columns + timings + peak RSS. Does NOT include y_score arrays or the confusion
matrix — those stay in the per-run JSON.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpai.paths import RESULTS_DIR

RUNS_DIR = RESULTS_DIR / "scene_b" / "runs"
OUT_CSV = RESULTS_DIR / "scene_b" / "summary.csv"

COLUMNS = [
    "dataset", "model", "kernel", "scaler", "poly", "seed",
    "n_train", "n_test", "n_classes",
    "MCC", "ACC",
    "TPR_macro", "TPR_weighted",
    "FPR",
    "PPV_macro", "PPV_weighted",
    "F1_macro", "F1_weighted",
    "AUROC_OvR_macro", "AUROC_OvR_weighted", "AUROC_OvO_macro", "AUROC_binary",
    "t_preprocess_s", "t_fit_s", "t_predict_s",
    "peak_rss_mb",
    "status",
]


def main() -> int:
    rows = []
    for p in sorted(RUNS_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        c = d.get("_config") or {}
        if d.get("_status") == "error" or "_error" in d:
            rows.append({
                "dataset": d.get("_dataset") or c.get("dataset", ""),
                "model":   d.get("_model") or c.get("model", ""),
                "kernel":  c.get("kernel", ""), "scaler": c.get("scaler", ""),
                "poly":    c.get("poly", ""),   "seed":   c.get("seed", ""),
                "status":  "error",
            })
            continue
        rows.append({
            "dataset": d["_dataset"], "model": d["_model"],
            "kernel":  c.get("kernel", ""), "scaler": c.get("scaler", ""),
            "poly":    c.get("poly", ""),   "seed":   c.get("seed", ""),
            "n_train": d.get("_n_train"), "n_test": d.get("_n_test"),
            "n_classes": d.get("_n_classes"),
            "MCC": d.get("MCC"), "ACC": d.get("ACC"),
            "TPR_macro": d.get("TPR Macro"), "TPR_weighted": d.get("TPR Weighted"),
            "FPR": d.get("FPR"),
            "PPV_macro": d.get("PPV Macro"), "PPV_weighted": d.get("PPV Weighted"),
            "F1_macro": d.get("F1 Macro"), "F1_weighted": d.get("F1 Weighted"),
            "AUROC_OvR_macro":    d.get("AUROC OvR Macro"),
            "AUROC_OvR_weighted": d.get("AUROC OvR Weighted"),
            "AUROC_OvO_macro":    d.get("AUROC OvO Macro"),
            "AUROC_binary":       d.get("AUROC Binary"),
            "t_preprocess_s": d.get("_t_preprocess"),
            "t_fit_s":        d.get("_t_fit"),
            "t_predict_s":    d.get("_t_predict"),
            "peak_rss_mb":    d.get("_peak_rss_mb"),
            "status":         "ok",
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in COLUMNS})

    print(f"[summary] wrote {len(rows)} rows → {OUT_CSV}")
    errs = sum(1 for r in rows if r.get("status") == "error")
    print(f"[summary] ok={len(rows)-errs}  errors={errs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
