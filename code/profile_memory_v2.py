#!/usr/bin/env python3
"""Measure peak RSS of a real HHHv2.fit / HHH.fit on Edge_IIoTset and N_BaIoT after the
in-place kernel + solver optimizations. Launched as a fresh subprocess per run so the
high-water-mark is clean (not cumulative across datasets).
"""

from __future__ import annotations

import resource
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _peak_of_child(script: str) -> tuple[int, float, str]:
    """Run `script` as a fresh python subprocess; return (rc, peak_rss_MB, stdout)."""
    proc = subprocess.run(
        [sys.executable, "-u", "-c", script],
        capture_output=True, text=True,
        env={"VECLIB_MAXIMUM_THREADS": "6", "OMP_NUM_THREADS": "6", "PATH": "/usr/bin:/bin"},
        cwd=HERE.parent,
    )
    # ru_maxrss of the CHILD — rusage(RUSAGE_CHILDREN) is cumulative across reaped children.
    peak_kb_or_b = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_mb = peak_kb_or_b / (1024 * 1024) if sys.platform == "darwin" else peak_kb_or_b / 1024
    return proc.returncode, peak_mb, proc.stdout + proc.stderr


TEMPLATE = textwrap.dedent("""
    import sys, resource, time, warnings
    sys.path.insert(0, 'code')
    warnings.filterwarnings('ignore')
    import torch  # preload
    from cpai import {model}
    from cpai.datasets import load_dataset
    from cpai.preprocessing import preprocess

    df, _ = load_dataset('{dataset}')
    X_tr, y_tr, X_te, y_te, enc = preprocess(df, '{dataset}',
                                              poly={poly}, kernel='{kernel}',
                                              scaler='{scaler}', seed=42)
    t0 = time.time()
    m = {model}(kernel='{kernel}')
    m.fit(X_tr, y_tr)
    t_fit = time.time()-t0
    y_pred = m.predict(X_te)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024*1024)
    print(f'{dataset},{model},peak_mb={{peak:.0f}},n_train={{len(y_tr)}},t_fit={{t_fit:.1f}}')
""")


def run_case(dataset: str, model: str, kernel: str, scaler: str, poly: int) -> None:
    script = TEMPLATE.format(dataset=dataset, model=model, kernel=kernel, scaler=scaler, poly=poly)
    prev_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024 if sys.platform == "darwin" else 1024)
    rc, peak, out = _peak_of_child(script)
    # rusage(CHILDREN) returns the MAX peak among all reaped children, but it's cumulative.
    # A cleaner reading: parse the child's own printed peak.
    for line in out.splitlines():
        if "peak_mb=" in line:
            print(f"  {line}")
            return
    print(f"  [FAIL {rc}] {dataset}/{model}: {out[-300:]}")


def main() -> int:
    cases = [
        # dataset, model, kernel, scaler, poly — the best-per-dataset configs
        ("Edge_IIoTset", "HHHv2", "laplacian", "QuantileTransformer", -1),
        ("Edge_IIoTset", "HHH",   "abel",      "QuantileTransformer", 0),
        ("N_BaIoT",      "HHHv2", "laplacian", "MinMaxScaler",        -1),
        ("N_BaIoT",      "HHH",   "laplacian", "MinMaxScaler",        -1),
        ("BoT_IoT",      "HHHv2", "laplacian", "MinMaxScaler",        -1),
        ("5G_NIDD",      "HHHv2", "laplacian", "MinMaxScaler",        -1),
    ]
    for case in cases:
        run_case(*case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
