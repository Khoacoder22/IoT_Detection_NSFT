import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from cpai.models import SpectralNFST, _maxst_components
from cpai.kernels import compute_kernel


def _two_mode_classes():
    return np.array([
        [-3.1, 0.0], [-3.0, 0.1], [-2.9, -0.1],
        [-1.1, 0.0], [-1.0, 0.1], [-0.9, -0.1],
        [0.9, 0.0], [1.0, 0.1], [1.1, -0.1],
        [2.9, 0.0], [3.0, 0.1], [3.1, -0.1],
    ]), np.repeat(np.array(["left", "right"]), 6)


def test_maxst_cuts_the_weakest_link_deterministically():
    affinity = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.2, 0.1],
        [0.1, 0.2, 1.0, 0.8],
        [0.1, 0.1, 0.8, 1.0],
    ])
    assert _maxst_components(affinity, 2).tolist() == [0, 0, 1, 1]


def test_spectral_nfst_training_embedding_and_classification():
    X, y = _two_mode_classes()
    model = SpectralNFST(n_components={"left": 2, "right": 2}, gamma=2.0)
    model.fit(X, y)

    assert model.proj_.shape == (len(X), 3)
    assert model.centroids_.shape == (4, 3)
    assert model.component_counts_.tolist() == [2, 2]
    assert sorted(np.bincount(model.component_ids_).tolist()) == [3, 3, 3, 3]
    assert np.max(np.abs(model.within_eigenvalues_)) < 1e-8
    assert np.mean(model.predict(X) == y) >= 0.9
    assert model.predict_proba(X).shape == (len(X), 2)
    assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)


def test_novelty_threshold_returns_paper_score_and_novelty_label():
    X, y = _two_mode_classes()
    model = SpectralNFST(n_components=2, gamma=2.0).fit(X, y)
    novelty = np.array([[100.0, 100.0]])
    scores = model.anomaly_score(novelty)
    labels, returned_scores = model.predict_with_novelty(novelty, threshold=0.0)

    assert np.array_equal(scores, returned_scores)
    assert labels.tolist() == [-1]


def test_component_count_validation():
    X, y = _two_mode_classes()
    with pytest.raises(ValueError, match="component count"):
        SpectralNFST(n_components=7).fit(X, y)
    with pytest.raises(ValueError, match="missing class labels"):
        SpectralNFST(n_components={"left": 2}).fit(X, y)


def test_linear_kernel_does_not_receive_an_invalid_gamma_argument():
    X, _ = _two_mode_classes()
    K = compute_kernel(X, kernel="linear", gamma=1.0)
    assert np.allclose(K, X @ X.T)
