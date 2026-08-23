"""Row-selecting Gaussian elimination used to pick linearly-independent samples per class."""

from __future__ import annotations

import numpy as np

from .kernels import compute_kernel


def gaussian_elimination(vectors: np.ndarray, threshold: float = 1e-10):
    """Pick linearly-independent rows of `vectors` via partial-pivot Gaussian elimination.

    Returns (independent_rows, original_indices).
    """
    matrix = vectors.copy().astype(np.float64)
    num_rows, num_cols = matrix.shape
    max_rank = min(num_rows, num_cols)
    independent_indices: list[int] = []

    for col in range(max_rank):
        pivot_row = int(np.argmax(np.abs(matrix[col:, col])) + col)
        if abs(matrix[pivot_row, col]) < threshold:
            continue
        if pivot_row != col:
            matrix[[col, pivot_row]] = matrix[[pivot_row, col]]
        matrix[col] /= matrix[col, col]
        matrix[col + 1:] -= np.outer(matrix[col + 1:, col], matrix[col])
        independent_indices.append(pivot_row)

    return vectors[independent_indices], independent_indices


def form_independent(
    X: np.ndarray, y: np.ndarray, kernel: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """For each class, keep only rows that contribute new information.

    If `kernel` is given, independence is measured in feature space (kernel matrix);
    otherwise on the raw vectors.
    """
    kept_X: list[np.ndarray] = []
    kept_y: list = []
    for label in np.unique(y):
        mask = y == label
        data = X[mask]
        if kernel is not None and kernel != "none":
            data = compute_kernel(data, kernel=kernel)
        _, idx = gaussian_elimination(data)
        kept_X.append(X[mask][idx])
        kept_y.extend([label] * len(idx))
    return np.vstack(kept_X), np.array(kept_y)
