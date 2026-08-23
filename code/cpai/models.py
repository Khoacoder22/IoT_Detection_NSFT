"""HHH, HHHv2, KNFST, and Spectral NFST implementations."""

from __future__ import annotations

import numpy as np
from scipy.linalg import LinAlgError, eigh, solve as _la_solve, svd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.preprocessing import KernelCenterer, OneHotEncoder

from .kernels import compute_kernel, gamma_heuristic


def _regularized_solve(A: np.ndarray, b: np.ndarray, ridge: float = 1e-5) -> np.ndarray:
    """Solve (A + ridge*I) alpha = b for alpha.

    MEMORY CONTRACT: A is MUTATED IN PLACE (diagonal updated with ridge, then LAPACK
    factors the matrix destructively during Cholesky). The caller must not use A
    again after this call.

    Implementation:
    - In-place diagonal add (saves one n×n float64 temporary vs `A + np.eye(n) * ridge`)
    - `overwrite_a=True` on scipy.linalg.solve so LAPACK factors in place
      (saves one more n×n copy that scipy would otherwise make internally)

    Strategy: Cholesky first (SPD assumption, fastest). If that fails, the matrix
    is left in an undefined state — the caller (HHH/HHHv2.fit) is responsible for
    rebuilding K and retrying with a boosted ridge.
    """
    n = A.shape[0]
    A.reshape(-1)[::n + 1] += ridge  # in-place diagonal ridge
    return _la_solve(A, b, assume_a="pos", overwrite_a=True, overwrite_b=False)


# ------------------------------------------------------------------ HHH (CPAI-OvA)

class HHH:
    """CPAI-OvA: single discriminant direction mapping class j to scalar target b_j = j."""

    def __init__(self, kernel: str | None = None, ridge: float | None = None,
                 gamma: float | str | None = None):
        """
        ridge: overrides both the linear (1e-5) and kernel (1e-9) defaults when set.
        gamma: bandwidth for rbf/poly/sigmoid. Options:
            - None             → sklearn default (γ = 1/n_features)
            - float            → explicit value
            - "heuristic"      → γ = 1/(2σ²) with σ = median kNN mean distance
                                 (paper Section 3.4, k=5). Computed from X_train at fit().
        """
        self.kernel = kernel
        self.ridge = ridge
        self.gamma = gamma
        self._gamma_used: float | None = None
        self.X_train: np.ndarray | None = None
        self.alpha: np.ndarray | None = None
        self._y_min: int = 0
        self._y_max: int = 0

    def _resolve_gamma(self, X: np.ndarray) -> float | None:
        if self.gamma == "heuristic":
            g = gamma_heuristic(X)
            self._gamma_used = g
            return g
        if isinstance(self.gamma, (int, float)):
            self._gamma_used = float(self.gamma)
            return float(self.gamma)
        self._gamma_used = None
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HHH":
        self.X_train = X
        self._y_min = int(y.min())
        self._y_max = int(y.max())
        y_f = y.astype(np.float64)
        g = self._resolve_gamma(X)
        if self.kernel is None or self.kernel == "none":
            r = self.ridge if self.ridge is not None else 1e-5
            Q = X.T
            A = Q.T @ Q
            try:
                self.alpha = _regularized_solve(A, y_f, ridge=r)
            except LinAlgError:
                A = Q.T @ Q
                self.alpha = _regularized_solve(A, y_f, ridge=max(r, 1e-9) * 1e6)
            self._theta = Q @ self.alpha
        else:
            r = self.ridge if self.ridge is not None else 1e-9
            K = compute_kernel(X, kernel=self.kernel, gamma=g)
            try:
                self.alpha = _regularized_solve(K, y_f, ridge=r)
            except LinAlgError:
                # Second-chance: rebuild K (destroyed by partial LAPACK factor),
                # try a 10^6× boosted ridge, then fall back to pinv on third failure.
                K = compute_kernel(X, kernel=self.kernel, gamma=g)
                try:
                    self.alpha = _regularized_solve(K, y_f, ridge=max(r, 1e-9) * 1e6)
                except LinAlgError:
                    K = compute_kernel(X, kernel=self.kernel, gamma=g)
                    K.reshape(-1)[::K.shape[0] + 1] += 1.0
                    self.alpha = np.linalg.pinv(K) @ y_f
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.kernel is None or self.kernel == "none":
            return X @ self._theta
        return compute_kernel(X, self.X_train, kernel=self.kernel,
                              gamma=self._gamma_used) @ self.alpha

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        rounded = np.round(scores).astype(int)
        out_of_range = (rounded < self._y_min) | (rounded > self._y_max)
        rounded[out_of_range] = -1
        return rounded

    def predict_proba(self, X: np.ndarray, n_classes: int) -> np.ndarray:
        """Pseudo-probabilities from scalar score: softmax(-|score - j|) over class j."""
        scores = self.decision_function(X)
        targets = np.arange(n_classes, dtype=np.float64)
        dists = np.abs(scores[:, None] - targets[None, :])
        return _softmax(-dists, axis=1)


# ------------------------------------------------------------------ HHHv2 (CPAI-OvR)

class HHHv2:
    """CPAI-OvR: one discriminant per class, predict by argmin |f_j(x) - 1|."""

    def __init__(self, kernel: str | None = None, ridge: float | None = None,
                 gamma: float | str | None = None):
        """See HHH.__init__ for `gamma` semantics."""
        self.kernel = kernel
        self.ridge = ridge
        self.gamma = gamma
        self._gamma_used: float | None = None
        self.X_train: np.ndarray | None = None
        self.alpha: np.ndarray | None = None

    def _resolve_gamma(self, X: np.ndarray) -> float | None:
        if self.gamma == "heuristic":
            g = gamma_heuristic(X)
            self._gamma_used = g
            return g
        if isinstance(self.gamma, (int, float)):
            self._gamma_used = float(self.gamma)
            return float(self.gamma)
        self._gamma_used = None
        return None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HHHv2":
        self.X_train = X
        y_onehot = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
        r = self.ridge if self.ridge is not None else 1e-5
        g = self._resolve_gamma(X)
        if self.kernel is None or self.kernel == "none":
            Q = X.T
            A = Q.T @ Q
            try:
                self.alpha = _regularized_solve(A, y_onehot, ridge=r)
            except LinAlgError:
                A = Q.T @ Q
                self.alpha = _regularized_solve(A, y_onehot, ridge=max(r, 1e-9) * 1e6)
            self._theta = Q @ self.alpha
        else:
            K = compute_kernel(X, kernel=self.kernel, gamma=g)
            try:
                self.alpha = _regularized_solve(K, y_onehot, ridge=r)
            except LinAlgError:
                K = compute_kernel(X, kernel=self.kernel, gamma=g)
                try:
                    self.alpha = _regularized_solve(K, y_onehot, ridge=max(r, 1e-9) * 1e6)
                except LinAlgError:
                    K = compute_kernel(X, kernel=self.kernel, gamma=g)
                    K.reshape(-1)[::K.shape[0] + 1] += 1.0
                    self.alpha = np.linalg.pinv(K) @ y_onehot
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.kernel is None or self.kernel == "none":
            return X @ self._theta
        return compute_kernel(X, self.X_train, kernel=self.kernel,
                              gamma=self._gamma_used) @ self.alpha

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return np.argmin(np.abs(scores - 1.0), axis=1)

    def predict_proba(self, X: np.ndarray, n_classes: int | None = None) -> np.ndarray:
        """Pseudo-probabilities: softmax over -|score_j - 1| (closer to 1 = higher prob)."""
        scores = self.decision_function(X)
        return _softmax(-np.abs(scores - 1.0), axis=1)


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


# ------------------------------------------------------------------ KNFST

def _sparse_L(y: np.ndarray) -> csr_matrix:
    classes, counts = np.unique(y, return_counts=True)
    inv = dict(zip(classes, 1.0 / counts))
    n = len(y)
    inv_vec = np.vectorize(inv.get)(y)
    rows, cols = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    mask = y[rows] == y[cols]
    rows, cols = rows[mask], cols[mask]
    return csr_matrix((inv_vec[rows], (rows, cols)), shape=(n, n))


def _nullspace(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    _, s, vh = svd(A)
    null = s <= eps
    return vh[null].T


def _learn_knfst(K: np.ndarray, y: np.ndarray):
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("KNFST requires 2+ classes")
    n, m = K.shape
    centered = KernelCenterer().fit_transform(K)
    vals, vecs = np.linalg.eig(centered)
    keep = vals > 1e-12
    vecs = vecs[:, keep]
    vals = vals[keep]
    basis = vecs @ np.diag(1.0 / np.sqrt(vals))

    L = _sparse_L(y)
    M = np.ones((m, m)) / m
    H = (((np.eye(m) - M) @ basis).T) @ K @ (np.eye(n, m) - L)
    t_sw = H @ H.T
    eigenvecs = _nullspace(t_sw)

    if eigenvecs.shape[1] < 1:
        eigenvals_, eigenvecs_ = np.linalg.eigh(t_sw)
        eigenvecs = eigenvecs_[:, eigenvals_.argsort()[0:1]]

    proj = (np.eye(m) - M) @ basis @ eigenvecs

    # class centroids in null space
    centroids = []
    for c in classes:
        ks_c = K[:, y == c]
        centroids.append(np.mean(ks_c.T @ proj, axis=0))
    return proj, np.array(centroids).real


class KNFST:
    """Kernel Null Foley-Sammon Transform — distance-to-class-centroid in the null space."""

    def __init__(self, kernel: str = "rbf"):
        self.kernel = kernel
        self.X_train: np.ndarray | None = None
        self.proj: np.ndarray | None = None
        self.centroids: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNFST":
        self.X_train = X
        K = compute_kernel(X, kernel=self.kernel)
        self.proj, self.centroids = _learn_knfst(K, y)
        return self

    def _distances(self, X: np.ndarray) -> np.ndarray:
        ks = compute_kernel(self.X_train, X, kernel=self.kernel)
        projected = (ks.T @ self.proj).real
        return np.linalg.norm(projected[:, None, :] - self.centroids[None, :, :], axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmin(self._distances(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Pseudo-probabilities: softmax over -distance (closer centroid = higher prob)."""
        d = self._distances(X)
        return _softmax(-d, axis=1)


# ------------------------------------------------------------- Spectral NFST

def _component_counts(n_components, classes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Resolve scalar, sequence, or class-keyed component counts to class order."""
    if np.isscalar(n_components):
        counts = np.full(len(classes), n_components)
    elif hasattr(n_components, "keys"):
        missing = [label for label in classes if label not in n_components]
        if missing:
            raise ValueError(f"n_components is missing class labels: {missing}")
        counts = np.asarray([n_components[label] for label in classes])
    else:
        counts = np.asarray(n_components)
        if counts.ndim != 1 or len(counts) != len(classes):
            raise ValueError(
                "n_components must be a scalar, a mapping keyed by class label, "
                f"or a sequence of length {len(classes)}"
            )

    try:
        counts_float = counts.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_components values must be positive integers") from exc
    if not np.all(np.isfinite(counts_float)) or not np.all(counts_float == np.floor(counts_float)):
        raise ValueError("n_components values must be positive integers")
    counts = counts_float.astype(int)
    class_sizes = np.asarray([np.count_nonzero(y == label) for label in classes])
    if np.any(counts < 1) or np.any(counts > class_sizes):
        details = {str(c): (int(q), int(n)) for c, q, n in zip(classes, counts, class_sizes)}
        raise ValueError(
            "Each component count must be between 1 and its class sample count; "
            f"got class: (components, samples) = {details}"
        )
    return counts


def _maxst_components(affinity: np.ndarray, n_components: int) -> np.ndarray:
    """Split a complete affinity graph by cutting the weakest MaxST edges.

    Component IDs are made deterministic by ordering components according to their
    smallest local sample index.
    """
    n = affinity.shape[0]
    if n_components == 1:
        return np.zeros(n, dtype=int)

    affinity = np.asarray(affinity, dtype=np.float64)
    affinity = (affinity + affinity.T) * 0.5
    if not np.all(np.isfinite(affinity)):
        raise ValueError("Kernel affinity graph contains NaN or infinite values")

    # A maximum spanning tree of affinities is a minimum spanning tree of these
    # shifted distances. A positive floor keeps maximum-affinity off-diagonal
    # edges from being interpreted as absent sparse-graph entries.
    off_diagonal = ~np.eye(n, dtype=bool)
    max_affinity = float(np.max(affinity[off_diagonal]))
    scale = max(1.0, abs(max_affinity), float(np.max(np.abs(affinity[off_diagonal]))))
    distances = max_affinity - affinity
    distances[off_diagonal] += np.finfo(np.float64).eps * scale
    np.fill_diagonal(distances, 0.0)

    tree = minimum_spanning_tree(csr_matrix(distances)).tocoo()
    if tree.nnz != n - 1:
        raise RuntimeError("Could not construct a spanning tree for a class affinity graph")

    edges = [
        (int(i), int(j), float(affinity[int(i), int(j)]))
        for i, j in zip(tree.row, tree.col)
    ]
    # The first Q-1 entries are the weakest similarity links to remove. Endpoint
    # keys make ties reproducible.
    edges.sort(key=lambda edge: (edge[2], min(edge[0], edge[1]), max(edge[0], edge[1])))
    kept = edges[n_components - 1:]
    if kept:
        rows = np.fromiter((edge[0] for edge in kept), dtype=int)
        cols = np.fromiter((edge[1] for edge in kept), dtype=int)
        graph = csr_matrix(
            (np.ones(2 * len(kept)), (np.r_[rows, cols], np.r_[cols, rows])),
            shape=(n, n),
        )
    else:
        graph = csr_matrix((n, n))

    found, labels = connected_components(graph, directed=False)
    if found != n_components:
        raise RuntimeError(f"Expected {n_components} MaxST components, found {found}")
    order = sorted(range(found), key=lambda component: int(np.flatnonzero(labels == component)[0]))
    remap = np.empty(found, dtype=int)
    remap[order] = np.arange(found)
    return remap[labels]


class SpectralNFST:
    """Spectral NFST from ``AD_with_Khoa_MND.pdf``.

    The estimator decomposes each class affinity graph into ``n_components``
    sub-manifolds with a maximum spanning tree, learns the paper's global
    indicator-matrix null projection, and classifies by the nearest projected
    component centroid.

    ``n_components`` may be one integer applied to every class, a sequence in
    sorted ``classes_`` order, or a mapping from original class labels to counts.
    Set a count greater than one to activate the proposed manifold decomposition.
    """

    def __init__(
        self,
        n_components=2,
        kernel: str = "rbf",
        gamma: float | str | None = None,
        novelty_threshold: float | None = None,
        novelty_label=-1,
        rank_tol: float | None = None,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.novelty_threshold = novelty_threshold
        self.novelty_label = novelty_label
        self.rank_tol = rank_tol
        self._gamma_used: float | None = None

    def _resolve_gamma(self, X: np.ndarray) -> float | None:
        if self.gamma == "heuristic":
            self._gamma_used = gamma_heuristic(X)
        elif isinstance(self.gamma, (int, float)):
            self._gamma_used = float(self.gamma)
        elif self.gamma is None:
            self._gamma_used = None
        else:
            raise ValueError("gamma must be None, a numeric value, or 'heuristic'")
        return self._gamma_used

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SpectralNFST":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("X must be 2-D and y must be 1-D with matching sample counts")
        if X.shape[0] < 2 or not np.all(np.isfinite(X)):
            raise ValueError("X must contain at least two finite samples")
        if self.kernel is None or str(self.kernel).lower() == "none":
            raise ValueError("SpectralNFST requires a strictly positive-definite kernel")
        if self.novelty_threshold is not None and self.novelty_threshold < 0:
            raise ValueError("novelty_threshold must be non-negative")

        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("SpectralNFST requires at least two classes")
        counts = _component_counts(self.n_components, self.classes_, y)
        total_components = int(counts.sum())
        if total_components - 1 >= X.shape[0]:
            raise ValueError("The total number of components must not exceed the sample count")

        self.X_train_ = X.copy()
        gamma = self._resolve_gamma(X)
        K = compute_kernel(X, kernel=self.kernel, gamma=gamma)
        K = np.asarray((K + K.T) * 0.5, dtype=np.float64)

        component_indices: list[np.ndarray] = []
        component_labels: list = []
        component_ids = np.empty(len(y), dtype=int)
        next_component = 0
        for class_label, target_count in zip(self.classes_, counts):
            class_indices = np.flatnonzero(y == class_label)
            local_affinity = K[np.ix_(class_indices, class_indices)]
            local_ids = _maxst_components(local_affinity, int(target_count))
            for local_component in range(int(target_count)):
                members = class_indices[local_ids == local_component]
                component_indices.append(members)
                component_labels.append(class_label)
                component_ids[members] = next_component
                next_component += 1

        self.component_indices_ = tuple(component_indices)
        self.component_labels_ = np.asarray(component_labels, dtype=y.dtype)
        self.component_ids_ = component_ids
        self.component_counts_ = counts

        H = np.zeros((len(y), total_components), dtype=np.float64)
        for component, members in enumerate(component_indices):
            H[members, component] = 1.0 / np.sqrt(len(members))

        # V is an orthonormal basis for the range of the centered similarity
        # profiles z_i - mean(z), i.e. the centered columns of K.
        centered_K = K - K.mean(axis=1, keepdims=True)
        U, singular_values, _ = svd(centered_K, full_matrices=False)
        default_tol = (
            max(centered_K.shape) * np.finfo(np.float64).eps * singular_values[0]
            if singular_values.size else 0.0
        )
        tolerance = default_tol if self.rank_tol is None else float(self.rank_tol)
        if tolerance < 0:
            raise ValueError("rank_tol must be non-negative")
        rank = int(np.count_nonzero(singular_values > tolerance))
        projection_dimension = total_components - 1
        if rank < projection_dimension:
            raise ValueError(
                f"Centered kernel rank {rank} is smaller than required projection "
                f"dimension {projection_dimension}"
            )
        V = U[:, :rank]

        # V.T Sw V with Sw = K(I-HH.T)K. The residual factorization is both
        # more stable and avoids materializing either n-by-n scatter matrix.
        KV = K @ V
        residual = KV - H @ (H.T @ KV)
        projected_sw = residual.T @ residual
        eigenvalues, alpha = eigh(
            projected_sw,
            subset_by_index=(0, projection_dimension - 1),
            check_finite=False,
        )
        self.proj_ = np.asarray(V @ alpha, dtype=np.float64)
        self.within_eigenvalues_ = np.asarray(eigenvalues, dtype=np.float64)

        # Each z_i is column i of K. Project the mean similarity profile of
        # every discovered component, exactly as Algorithm 1 line 14.
        similarity_means = np.column_stack(
            [K[:, members].mean(axis=1) for members in component_indices]
        )
        self.centroids_ = np.asarray(similarity_means.T @ self.proj_, dtype=np.float64)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "proj_"):
            raise RuntimeError("SpectralNFST must be fitted before prediction")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Embed samples into the learned Q-1 dimensional null space."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(f"X must have shape (n_samples, {self.X_train_.shape[1]})")
        similarities = compute_kernel(
            self.X_train_, X, kernel=self.kernel, gamma=self._gamma_used
        )
        return np.asarray(similarities.T @ self.proj_, dtype=np.float64)

    def component_distances(self, X: np.ndarray) -> np.ndarray:
        """Euclidean distances to all projected sub-group centroids."""
        projected = self.transform(X)
        return np.linalg.norm(
            projected[:, None, :] - self.centroids_[None, :, :], axis=2
        )

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Paper anomaly score A: distance to the nearest component centroid."""
        return np.min(self.component_distances(X), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Closed-set Algorithm 2 prediction using the nearest component."""
        nearest = np.argmin(self.component_distances(X), axis=1)
        return self.component_labels_[nearest]

    def predict_with_novelty(
        self, X: np.ndarray, threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Algorithm 3 labels and anomaly scores for threshold ``tau``."""
        tau = self.novelty_threshold if threshold is None else threshold
        if tau is None:
            raise ValueError("A novelty threshold must be passed or set at construction")
        if tau < 0:
            raise ValueError("threshold must be non-negative")
        distances = self.component_distances(X)
        scores = np.min(distances, axis=1)
        labels = self.component_labels_[np.argmin(distances, axis=1)].astype(object)
        labels[scores > tau] = self.novelty_label
        return labels, scores

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Per-class similarity scores (negative nearest-component distance)."""
        component_distances = self.component_distances(X)
        class_distances = np.empty((len(component_distances), len(self.classes_)))
        for column, class_label in enumerate(self.classes_):
            class_distances[:, column] = np.min(
                component_distances[:, self.component_labels_ == class_label], axis=1
            )
        return -class_distances

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Pseudo-probabilities from softmax of negative per-class distance."""
        return _softmax(self.decision_function(X), axis=1)
