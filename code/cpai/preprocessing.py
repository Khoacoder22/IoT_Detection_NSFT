"""End-to-end preprocessing: scale, polynomial expand, split, (optionally) remove outliers, form independent."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    Normalizer,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from .gaussian_elim import form_independent

SCALERS = ("QuantileTransformer", "StandardScaler", "MinMaxScaler", "RobustScaler", "Normalizer")


def _scaler(name: str, random_state: int = 42):
    mapping = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
        "Normalizer": Normalizer(),
        "QuantileTransformer": QuantileTransformer(output_distribution="normal", random_state=random_state),
    }
    if name not in mapping:
        raise ValueError(f"Unknown scaler '{name}'. Valid: {list(mapping)}")
    return mapping[name]


def _remove_outliers_lof(X: np.ndarray, y: np.ndarray, contamination: float = 0.05):
    """Drop LOF-flagged rows per class, except class 0 (benign)."""
    keep_X, keep_y = [], []
    for label in np.unique(y):
        mask = y == label
        X_cls, y_cls = X[mask], y[mask]
        if label == 0:
            keep_X.append(X_cls)
            keep_y.append(y_cls)
            continue
        outliers = LocalOutlierFactor(contamination=contamination).fit_predict(X_cls) == -1
        keep_X.append(X_cls[~outliers])
        keep_y.append(y_cls[~outliers])
    return np.vstack(keep_X), np.concatenate(keep_y)


def preprocess(
    df,
    dataset_name: str,
    poly: int,
    kernel: str | None,
    scaler: str,
    seed: int = 42,
):
    """Full preprocessing pipeline matching the legacy `preprocess_data_iv`.

    Steps: imputation (IoTID20 only) -> polynomial expand -> stratified 80/20 split
    -> scale -> nan-to-num -> per-class LOF outlier removal -> form_independent
    (only when poly != -1).

    Returns (X_train, y_train, X_test, y_test, label_encoder).
    """
    data = df.to_numpy()
    X, y_raw = data[:, :-1].astype(np.float64), data[:, -1]

    if dataset_name == "IoTID20":
        X[np.isinf(X)] = np.nan
        X = SimpleImputer(strategy="mean").fit_transform(X)

    if poly > 1:
        X = PolynomialFeatures(poly, interaction_only=True).fit_transform(X)

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y_raw, test_size=0.2, stratify=y_raw, random_state=seed
    )

    idx = y_train_raw.argsort()
    X_train, y_train_raw = X_train[idx], y_train_raw[idx]

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_raw)
    y_test = encoder.transform(y_test_raw)

    sc = _scaler(scaler, random_state=seed)
    sc.fit(X_train)
    X_train = np.nan_to_num(sc.transform(X_train), nan=0.0)
    X_test = np.nan_to_num(sc.transform(X_test), nan=0.0)

    X_train, y_train = _remove_outliers_lof(X_train, y_train)

    if poly != -1:
        X_train, y_train = form_independent(X_train, y_train, kernel)

    return X_train, y_train, X_test, y_test, encoder
