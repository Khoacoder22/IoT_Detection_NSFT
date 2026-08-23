#!/usr/bin/env python3
"""Trace peak RSS through every stage of HHHv2.fit on the largest dataset (Edge_IIoTset)
to identify the memory bottleneck.

Uses both resource.getrusage (OS-level peak RSS, cumulative / high-water-mark) and
psutil.Process().memory_info().rss (instantaneous) to distinguish cumulative peaks
from current footprint.
"""

from __future__ import annotations

import gc
import resource
import sys
import time
from pathlib import Path

# Preload torch before scipy (macOS libomp)
try:
    import torch  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import psutil
from scipy.linalg import solve as _la_solve

from cpai.datasets import load_dataset
from cpai.kernels import compute_kernel
from cpai.preprocessing import preprocess

# ------------------------------------------------------------ memory probes

_proc = psutil.Process()


def _rss_mb() -> float:
    """Instantaneous RSS of THIS process (MB)."""
    return _proc.memory_info().rss / (1024 ** 2)


def _peak_mb() -> float:
    """OS high-water-mark since start (MB). Only increases."""
    kb_or_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb_or_bytes / (1024 * 1024) if sys.platform == "darwin" else kb_or_bytes / 1024


def _mark(label: str, notes: str = "") -> None:
    print(f"  [{label:<34}] RSS={_rss_mb():7.0f} MB   peak={_peak_mb():7.0f} MB  {notes}", flush=True)


# ------------------------------------------------------------ manual HHHv2 trace

def trace_hhhv2(dataset: str, kernel: str, scaler: str, poly: int, seed: int = 42) -> None:
    print(f"\n=== Tracing HHHv2 on {dataset} (kernel={kernel}, scaler={scaler}, poly={poly}) ===")
    _mark("start (post-imports)")

    # --- 1. Data load
    df, limit = load_dataset(dataset)
    _mark("after load_dataset", f"df shape={df.shape}")

    # --- 2. Full preprocessing
    X_train, y_train, X_test, y_test, enc = preprocess(
        df, dataset, poly=poly, kernel=kernel, scaler=scaler, seed=seed
    )
    n_train, n_feat = X_train.shape
    n_test = X_test.shape[0]
    n_classes = len(enc.classes_)
    _mark("after preprocess",
          f"X_train={X_train.shape}  X_test={X_test.shape}  n_classes={n_classes}")

    del df  # free the raw dataframe
    gc.collect()
    _mark("after drop raw df + gc", "")

    # --- 3. Build kernel matrix K = k(X_train, X_train)
    t0 = time.time()
    K = compute_kernel(X_train, kernel=kernel)
    K_MB_theory = (n_train ** 2 * 8) / 1024 ** 2
    _mark("after compute_kernel(X_train)",
          f"K shape={K.shape}  expected~{K_MB_theory:.0f} MB  t={time.time()-t0:.1f}s")

    # --- 4. One-hot encode y_train (HHHv2 targets)
    from sklearn.preprocessing import OneHotEncoder
    y_oh = OneHotEncoder(sparse_output=False).fit_transform(y_train.reshape(-1, 1))
    _mark("after y_onehot", f"y_oh shape={y_oh.shape}")

    # --- 5. A_reg = K + ridge * I — this is the main extra-copy cost
    t0 = time.time()
    A_reg = K + np.eye(n_train) * 1e-5
    _mark("after A + eye*ridge",
          f"A_reg shape={A_reg.shape}  (K and A_reg both live; t={time.time()-t0:.1f}s)")

    # --- 6. scipy.linalg.solve (Cholesky) — LAPACK may alloc workspace
    t0 = time.time()
    alpha = _la_solve(A_reg, y_oh, assume_a="pos")
    _mark("after scipy.linalg.solve(pos)",
          f"alpha shape={alpha.shape}  t={time.time()-t0:.1f}s")

    # --- 7. Free intermediates one at a time to see the drop
    del A_reg
    gc.collect()
    _mark("after del A_reg + gc", "")

    del K
    gc.collect()
    _mark("after del K + gc", "")

    # --- 8. Inference — kernel K(X_test, X_train) @ alpha
    t0 = time.time()
    K_test = compute_kernel(X_test, X_train, kernel=kernel)
    _mark("after compute_kernel(X_test, X_train)",
          f"K_test shape={K_test.shape}  t={time.time()-t0:.1f}s")

    scores = K_test @ alpha
    _mark("after K_test @ alpha",
          f"scores shape={scores.shape}")

    del K_test
    gc.collect()
    _mark("after del K_test + gc", "")

    print(f"\n  >> Summary: n_train={n_train}, n_feat={n_feat}, n_test={n_test}, n_classes={n_classes}")
    print(f"     Theoretical single n×n float64 matrix = {K_MB_theory:.0f} MB")
    print(f"     Final peak RSS (OS high-water-mark)   = {_peak_mb():.0f} MB")


def main() -> int:
    # Run on the two worst cases from the report
    for dataset, kernel, scaler, poly in [
        ("Edge_IIoTset", "laplacian", "QuantileTransformer", -1),   # 9.3 GB peak
        ("N_BaIoT", "laplacian", "MinMaxScaler", -1),                # 4.9 GB peak
    ]:
        trace_hhhv2(dataset, kernel, scaler, poly)
    return 0


if __name__ == "__main__":
    sys.exit(main())
