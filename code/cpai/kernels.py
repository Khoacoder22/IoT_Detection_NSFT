"""Kernel functions matching the CPAI notebook (NewIdea_AnhHoi.ipynb).

Standard sklearn kernels: linear, poly, rbf, sigmoid. Custom kernels:
- abel:      K(x,y) = exp(-alpha * ||x-y||_2)         with alpha=0.1
- laplacian: K(x,y) = exp(-alpha * ||x-y||_1)         with alpha=0.1
- sobolev:   K(x,y) = r^(k-d/2) * K_{k-d/2}(r)        with k=0.5, d=1, r=||x-y||_2

These match the notebook exactly — alpha and Bessel-order hyperparameters are
hardcoded there (not sklearn defaults).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import kv
from sklearn.metrics.pairwise import pairwise_kernels as sk_pairwise

SKLEARN_KERNELS = ("linear", "poly", "rbf", "sigmoid")
CUSTOM_KERNELS = ("abel", "laplacian", "sobolev")
KERNEL_CHOICES = ("none", *SKLEARN_KERNELS, *CUSTOM_KERNELS)

ABEL_ALPHA = 0.1
LAPLACIAN_ALPHA = 0.1
SOBOLEV_K = 0.5
SOBOLEV_D = 1


def abel_kernel(X: np.ndarray, Y: np.ndarray | None = None, alpha: float = ABEL_ALPHA) -> np.ndarray:
    """Compute in-place: np.exp(-alpha * cdist(X, Y, 'euclidean'), out=D) reuses one buffer."""
    if Y is None:
        Y = X
    D = cdist(X, Y, metric="euclidean")
    np.multiply(D, -alpha, out=D)
    np.exp(D, out=D)
    return D


def laplacian_kernel(X: np.ndarray, Y: np.ndarray | None = None, alpha: float = LAPLACIAN_ALPHA) -> np.ndarray:
    """In-place exp over the L1 distance buffer — avoids a second n×n float64 allocation."""
    if Y is None:
        Y = X
    D = cdist(X, Y, metric="cityblock")
    np.multiply(D, -alpha, out=D)
    np.exp(D, out=D)
    return D


def sobolev_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    k: float = SOBOLEV_K,
    d: int = SOBOLEV_D,
) -> np.ndarray:
    if Y is None:
        Y = X
    r = cdist(X, Y, metric="euclidean")
    r += 1e-10  # in-place
    order = k - d / 2.0
    # r^order * K_{order}(r) — still needs two full n×n arrays (Bessel allocates its own output).
    # We can at least reuse r for the power: ** is not in-place, but np.power with out= is.
    return np.power(r, order) * kv(order, r)


def gamma_heuristic(X: np.ndarray, k: int = 5) -> float:
    """Data-driven bandwidth γ per Section 3.4 of the revised CPAI paper.

        σ = median over samples of the mean distance to its k nearest neighbors
        γ = 1 / (2 σ²)

    The k-NN search excludes the sample itself. Returns a scalar γ suitable for
    sklearn's `pairwise_kernels(..., gamma=γ)` on rbf / poly / sigmoid.
    """
    from sklearn.neighbors import NearestNeighbors
    k_eff = min(k + 1, X.shape[0])  # +1 for self
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X)
    dists, _ = nn.kneighbors(X)
    # drop self (column 0 is distance 0)
    mean_dist_per_sample = dists[:, 1:].mean(axis=1)
    sigma = float(np.median(mean_dist_per_sample))
    # numeric floor so γ stays finite if points collapse
    sigma = max(sigma, 1e-12)
    return 1.0 / (2.0 * sigma * sigma)


def compute_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    kernel: str | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    """Compute pairwise kernel matrix. `kernel=None` means linear-equivalent (Q^T Q).

    `gamma`: if given, passed through to sklearn for rbf/poly/sigmoid. `None` falls back
    to sklearn's default (1/n_features). Ignored for linear and our custom kernels.
    """
    if kernel is None or kernel == "none":
        return X @ (Y.T if Y is not None else X.T)
    key = kernel.lower()
    if key in SKLEARN_KERNELS:
        # linear_kernel has no gamma parameter; the other sklearn kernels treat
        # gamma=None as their default (1/n_features).
        if key == "linear":
            return sk_pairwise(X, Y, metric=key)
        return sk_pairwise(X, Y, metric=key, gamma=gamma)
    if key == "laplacian":
        return laplacian_kernel(X, Y)
    if key == "abel":
        return abel_kernel(X, Y)
    if key == "sobolev":
        return sobolev_kernel(X, Y)
    raise ValueError(f"Unknown kernel '{kernel}'. Valid: {KERNEL_CHOICES}")
