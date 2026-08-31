"""Kernel functions matching the CPAI notebook (NewIdea_AnhHoi.ipynb).

Standard sklearn kernels: linear, poly, rbf, sigmoid. Custom kernels:
- abel:      K(x,y) = exp(-alpha * ||x-y||_2)         with alpha=0.1
- laplacian: K(x,y) = exp(-alpha * ||x-y||_1)         with alpha=0.1
- sobolev:   K(x,y) = r^(k-d/2) * K_{k-d/2}(r)        with k=0.5, d=1, r=||x-y||_2
- rff:       Random Fourier Features approximation for RBF kernel
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import kv
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics.pairwise import cosine_similarity, chi2_kernel # chi-kernel 
from sklearn.metrics.pairwise import pairwise_kernels as sk_pairwise

SKLEARN_KERNELS = ("linear", "poly", "rbf", "sigmoid")
CUSTOM_KERNELS = ("abel", "laplacian", "sobolev", "rff", "chi2", "fractional")
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
    return np.power(r, order) * kv(order, r)

# RFF
def rff_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    n_components: int = 200,
    gamma: float | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Approximate RBF Kernel matrix using Random Fourier Features."""
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    rff = RBFSampler(n_components=n_components, gamma=gamma, random_state=random_state)
    X_trans = rff.fit_transform(X)
    Y_trans = rff.transform(Y) if Y is not None else X_trans
    return X_trans @ Y_trans.T


def gamma_heuristic(X: np.ndarray, k: int = 5) -> float:
    """Data-driven bandwidth γ per Section 3.4 of the revised CPAI paper."""
    from sklearn.neighbors import NearestNeighbors
    k_eff = min(k + 1, X.shape[0])  # +1 for self
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X)
    dists, _ = nn.kneighbors(X)
    mean_dist_per_sample = dists[:, 1:].mean(axis=1)
    sigma = float(np.median(mean_dist_per_sample))
    sigma = max(sigma, 1e-12)
    return 1.0 / (2.0 * sigma * sigma)

def fractional_kernel(X, Y=None, gamma=None):
    """Tính Fractional (L_0.5) Kernel tối ưu bộ nhớ 2D."""
    X_arr = np.asarray(X, dtype=np.float64)
    Y_arr = X_arr if Y is None else np.asarray(Y, dtype=np.float64)

    n_samples_X, n_features = X_arr.shape
    n_samples_Y = Y_arr.shape[0]

    # Cộng dồn 2D theo từng cột thuộc tính để tối ưu bộ nhớ
    dist_l05 = np.zeros((n_samples_X, n_samples_Y), dtype=np.float64)
    for j in range(n_features):
        diff_j = np.abs(X_arr[:, j : j + 1] - Y_arr[:, j : j + 1].T)
        dist_l05 += np.sqrt(diff_j)

    alpha = float(gamma) if gamma is not None else 0.1
    return np.exp(-alpha * dist_l05)

def compute_kernel(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    kernel: str | None = None,
    gamma: float | None = None,
) -> np.ndarray:
    """Compute pairwise kernel matrix."""
    if kernel is None or kernel == "none":
        return X @ (Y.T if Y is not None else X.T)
    key = kernel.lower()
    if key in SKLEARN_KERNELS:
        if key == "linear":
            return sk_pairwise(X, Y, metric=key)
        return sk_pairwise(X, Y, metric=key, gamma=gamma)
    if key == "laplacian":
        return laplacian_kernel(X, Y)
    if key == "abel":
        return abel_kernel(X, Y)
    if key == "sobolev":
        return sobolev_kernel(X, Y)
    if key == "rff":
        return rff_kernel(X, Y, gamma=gamma)
    if key == "chi2":
        Y_arr = X if Y is None else Y
        # Ép kiểu float64 trực tiếp để xử lý triệt để lỗi dtype('O')
        X_pos = np.asarray(X, dtype=np.float64)
        Y_pos = np.asarray(Y_arr, dtype=np.float64)

        min_val = min(float(np.min(X_pos)), float(np.min(Y_pos)))
        if min_val < 0:
            X_pos = X_pos - min_val
            Y_pos = Y_pos - min_val

        gamma_val = float(gamma) if gamma is not None else (1.0 / X_pos.shape[1])
        return chi2_kernel(X_pos, Y_pos, gamma=gamma_val)
    if key == "fractional":
        return fractional_kernel(X, Y, gamma)
    raise ValueError(f"Unknown kernel '{kernel}'. Valid: {KERNEL_CHOICES}")