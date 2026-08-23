import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from spectral_nfst_experiments import (
    _novelty_metrics,
    _prepare_multinovelty_data,
    _sample_per_class,
)


def _novelty_frame():
    rng = np.random.default_rng(4)
    rows = []
    for label, center in [("known_a", -2.0), ("known_b", 0.0), ("unknown", 2.0)]:
        X = rng.normal(center, 0.2, size=(30, 3))
        rows.extend([(*sample, label) for sample in X])
    return pd.DataFrame(rows, columns=["f1", "f2", "f3", "Label"])


def test_class_sampling_keeps_labels_and_caps_each_class():
    sampled = _sample_per_class(_novelty_frame(), 7, seed=42)
    assert sampled.groupby("Label").size().to_dict() == {
        "known_a": 7, "known_b": 7, "unknown": 7,
    }


def test_multinovelty_preparation_removes_unknown_from_training_only():
    X_train, y_train, X_test, y_test, encoder = _prepare_multinovelty_data(
        _novelty_frame(), "synthetic", ["unknown"], -1, "rbf",
        "StandardScaler", 42,
    )
    assert X_train.shape[0] == y_train.shape[0]
    assert encoder.classes_.tolist() == ["known_a", "known_b"]
    assert set(np.unique(y_train)) == {0, 1}
    assert -1 in y_test
    assert set(np.unique(y_test)) == {-1, 0, 1}


def test_novelty_metrics_use_continuous_scores():
    y_true = np.array([0, 1, -1, -1])
    y_pred = np.array([0, 1, -1, 0])
    scores = np.array([0.05, 0.10, 0.90, 0.80])
    metrics = _novelty_metrics(y_true, y_pred, scores)
    assert metrics["NDR"] == 50.0
    assert metrics["FPR_N"] == 0.0
    assert metrics["AUC_N"] == 100.0
    assert metrics["AUPRC_N"] == 100.0
