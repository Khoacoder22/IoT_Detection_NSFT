"""Wrappers for the baseline classifiers used in paper Tables 4-8.

Classical baselines: KNN, LDA, NuSVC, SGD, GauNB, NC, RNC (from notebook cell 27).
Modern / lightweight additions:
    - LGBM           (gradient boosting, notebook __MODEL_SUP)
    - MLP            (sklearn feedforward NN, notebook __MODEL_SUP)
    - TabNet         (attention-based tabular DL, pytorch-tabnet)
    - FTTransformer  (feature-tokenizer transformer, tab-transformer-pytorch)
                     — used in lieu of TabTransformer since our preprocessed data is
                     continuous-only (all categoricals label-encoded into ints, then
                     scaled). FT-Transformer is designed for continuous features.

Params track the notebook (cell 27 __Prameter_profile) where they exist.
TabNet uses Arik & Pfister (2019) defaults; FT-Transformer uses Gorishniy et al.
(2021) small-model defaults.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    NearestCentroid,
    RadiusNeighborsClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import NuSVC

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    LGBMClassifier = None  # type: ignore[assignment]
    _HAS_LGBM = False

try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetRaw
    _HAS_TABNET = True
except ImportError:
    _TabNetRaw = None  # type: ignore[assignment]
    _HAS_TABNET = False

try:
    import torch  # noqa: F811  (re-import guard; torch already imported above if available)
    from tab_transformer_pytorch import FTTransformer as _FTTrans
    _HAS_FTT = True
except ImportError:
    _FTTrans = None  # type: ignore[assignment]
    _HAS_FTT = False

try:
    import torch  # noqa: F811
    import torch.nn as _nn
    _HAS_SAINT = True
except ImportError:
    _HAS_SAINT = False

# KNFST lives in cpai.models (it shares CPAI's kernel / ridge infrastructure) but is
# a comparison BASELINE in the paper — re-export from here so build_baseline("KNFST")
# works the same as the other baselines.
from .models import KNFST as _KNFSTImpl


class KNFSTClassifier(_KNFSTImpl):
    """Thin alias of cpai.models.KNFST with a sklearn-style __init__ signature for the
    baseline runner. Default kernel = rbf (paper's original KNFST reference)."""
    def __init__(self, kernel: str = "rbf"):
        super().__init__(kernel=kernel)


class TabNetClassifier:
    """sklearn-style wrapper around pytorch_tabnet TabNetClassifier.

    - Forces `num_workers=0` during fit to avoid DataLoader fork-on-macOS segfault.
    - Always pins to CPU (our runs are CPU-bound with multi-worker Pool — using MPS
      from inside a forked subprocess can also segfault).
    """

    def __init__(self, n_d=16, n_a=16, n_steps=4, max_epochs=50, patience=10,
                 batch_size=256, virtual_batch_size=128, seed=42):
        if not _HAS_TABNET:
            raise RuntimeError("pytorch-tabnet not installed. Run: pip install pytorch-tabnet")
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.seed = seed
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabNetClassifier":
        self._model = _TabNetRaw(
            n_d=self.n_d, n_a=self.n_a, n_steps=self.n_steps,
            seed=self.seed, verbose=0, device_name="cpu",
        )
        self._model.fit(
            np.ascontiguousarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.int64),
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
            num_workers=0,
            drop_last=False,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(np.ascontiguousarray(X, dtype=np.float32))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(np.ascontiguousarray(X, dtype=np.float32))


class FTTransformerClassifier:
    """Minimal training loop for the lucidrains tab-transformer-pytorch FTTransformer.

    No categorical columns — our preprocessor encodes categoricals to ints then
    passes everything through a scaler, so downstream all features are continuous.
    We thus pass `categories=()` and route all columns through the continuous path.
    """

    def __init__(self, dim=32, depth=3, heads=4, dim_head=16,
                 attn_dropout=0.1, ff_dropout=0.1,
                 max_epochs=30, batch_size=256, lr=1e-3, seed=42):
        if not _HAS_FTT:
            raise RuntimeError("tab-transformer-pytorch not installed.")
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        self._model = None
        self._n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FTTransformerClassifier":
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = torch.device("cpu")

        self._n_classes = int(np.asarray(y).max()) + 1
        n_cont = X.shape[1]

        self._model = _FTTrans(
            categories=(),
            num_continuous=n_cont,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            dim_head=self.dim_head,
            dim_out=self._n_classes,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
        ).to(device)

        X_t = torch.tensor(np.ascontiguousarray(X, dtype=np.float32))
        y_t = torch.tensor(np.asarray(y, dtype=np.int64))
        ds = TensorDataset(X_t, y_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            num_workers=0, drop_last=False)

        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self._model.train()
        empty_categ = torch.empty((self.batch_size, 0), dtype=torch.long, device=device)
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                bs = xb.shape[0]
                ec = empty_categ[:bs] if bs != self.batch_size else empty_categ
                opt.zero_grad()
                logits = self._model(ec, xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        return self

    def _infer(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        X_t = torch.tensor(np.ascontiguousarray(X, dtype=np.float32))
        with torch.no_grad():
            ec = torch.empty((X_t.shape[0], 0), dtype=torch.long)
            logits = self._model(ec, X_t)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._infer(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._infer(X)


# ------------------------------------------------------------------------- SAINT

class _SAINTBlock(_nn.Module if _HAS_SAINT else object):
    """One SAINT layer: feature self-attention → intersample attention → FFN.

    Feature self-attention: (batch, n_tokens, dim) — attends across feature axis.
    Intersample attention:  (n_tokens, batch, dim) — attends across batch axis, one
    attention pool per feature token (the SAINT novelty — samples attend to samples).
    """

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.feat_attn = _nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.sample_attn = _nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ff = _nn.Sequential(
            _nn.Linear(dim, 4 * dim), _nn.GELU(), _nn.Dropout(dropout),
            _nn.Linear(4 * dim, dim), _nn.Dropout(dropout),
        )
        self.ln1 = _nn.LayerNorm(dim)
        self.ln2 = _nn.LayerNorm(dim)
        self.ln3 = _nn.LayerNorm(dim)
        self.dropout = _nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, n_tokens, dim)
        # 1. Feature self-attention (within each sample)
        h = self.ln1(x)
        a, _ = self.feat_attn(h, h, h, need_weights=False)
        x = x + self.dropout(a)

        # 2. Intersample attention (across batch, for each token position)
        #    transpose to (n_tokens, batch, dim) so attention operates across batch.
        h = self.ln2(x).transpose(0, 1)
        a, _ = self.sample_attn(h, h, h, need_weights=False)
        x = x + self.dropout(a.transpose(0, 1))

        # 3. FFN
        x = x + self.ff(self.ln3(x))
        return x


class _SAINTNet(_nn.Module if _HAS_SAINT else object):
    """Minimal SAINT: feature tokenization + [CLS] + L dual-attention blocks + head."""

    def __init__(self, num_features: int, n_classes: int,
                 dim: int = 32, depth: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        # One learnable linear embedding per feature (like FT-Transformer's numerical tokenizer)
        self.feat_weight = _nn.Parameter(torch.randn(num_features, dim) * 0.02)
        self.feat_bias = _nn.Parameter(torch.zeros(num_features, dim))
        self.cls_token = _nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.blocks = _nn.ModuleList([_SAINTBlock(dim, heads, dropout) for _ in range(depth)])
        self.head = _nn.Sequential(_nn.LayerNorm(dim), _nn.Linear(dim, n_classes))

    def forward(self, x):
        # x: (batch, num_features)
        b = x.shape[0]
        # Feature tokenization: token_j = x[:, j] * feat_weight[j] + feat_bias[j]
        tokens = x.unsqueeze(-1) * self.feat_weight + self.feat_bias  # (batch, num_features, dim)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (batch, num_features+1, dim)
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.head(tokens[:, 0])  # classify from [CLS]


class SAINTClassifier:
    """sklearn-style wrapper for the minimal SAINT implementation above.

    Trains with Adam + cross-entropy, 30 epochs default. Intersample attention means
    the forward pass depends on the batch composition at test time too — we use a
    deterministic batched inference (no shuffle) so predictions are reproducible.
    """

    def __init__(self, dim: int = 32, depth: int = 3, heads: int = 4,
                 dropout: float = 0.1, max_epochs: int = 30,
                 batch_size: int = 256, lr: float = 1e-3, seed: int = 42):
        if not _HAS_SAINT:
            raise RuntimeError("torch not installed.")
        self.dim = dim; self.depth = depth; self.heads = heads; self.dropout = dropout
        self.max_epochs = max_epochs; self.batch_size = batch_size
        self.lr = lr; self.seed = seed
        self._model = None
        self._n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SAINTClassifier":
        from torch.utils.data import TensorDataset, DataLoader

        torch.manual_seed(self.seed); np.random.seed(self.seed)
        device = torch.device("cpu")
        self._n_classes = int(np.asarray(y).max()) + 1

        self._model = _SAINTNet(
            num_features=X.shape[1], n_classes=self._n_classes,
            dim=self.dim, depth=self.depth, heads=self.heads, dropout=self.dropout,
        ).to(device)

        X_t = torch.tensor(np.ascontiguousarray(X, dtype=np.float32))
        y_t = torch.tensor(np.asarray(y, dtype=np.int64))
        loader = DataLoader(
            TensorDataset(X_t, y_t), batch_size=self.batch_size,
            shuffle=True, num_workers=0, drop_last=False,
        )
        opt = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = _nn.CrossEntropyLoss()
        self._model.train()
        for _ in range(self.max_epochs):
            for xb, yb in loader:
                opt.zero_grad()
                loss = loss_fn(self._model(xb.to(device)), yb.to(device))
                loss.backward()
                opt.step()
        return self

    def _infer(self, X: np.ndarray) -> np.ndarray:
        self._model.eval()
        X_t = torch.tensor(np.ascontiguousarray(X, dtype=np.float32))
        # Batched deterministic inference
        probas = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                chunk = X_t[i : i + self.batch_size]
                logits = self._model(chunk)
                probas.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probas, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self._infer(X), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._infer(X)


CLASSICAL_NAMES = ("KNN", "LDA", "NuSVC", "SGD", "GauNB", "NC", "RNC", "KNFST")
MODERN_NAMES = ("LGBM", "MLP", "TabNet", "FTTransformer", "SAINT")
BASELINE_NAMES = CLASSICAL_NAMES + MODERN_NAMES

_PARAMS = {
    "SGD":   {"random_state": 42},
    "NuSVC": {"nu": 0.2, "kernel": "rbf", "random_state": 42},
    "LDA":   {"solver": "svd"},
    "KNN":   {"n_neighbors": 5},
    "NC":    {"metric": "euclidean"},
    "GauNB": {"var_smoothing": 1e-9},
    "RNC":   {"outlier_label": "most_frequent"},
    "KNFST": {"kernel": "rbf"},  # original paper's reference KNFST baseline
    "LGBM":  {"n_estimators": 100, "random_state": 42, "verbosity": -1},
    "MLP":   {"random_state": 42},
    "TabNet": {"n_d": 16, "n_a": 16, "n_steps": 4, "max_epochs": 50,
               "patience": 10, "batch_size": 256, "seed": 42},
    "FTTransformer": {"dim": 32, "depth": 3, "heads": 4, "dim_head": 16,
                       "max_epochs": 30, "batch_size": 256, "lr": 1e-3, "seed": 42},
    # SAINT (Somepalli et al. 2021): feature self-attention + intersample attention.
    "SAINT": {"dim": 32, "depth": 3, "heads": 4, "dropout": 0.1,
               "max_epochs": 30, "batch_size": 256, "lr": 1e-3, "seed": 42},
}

_BUILDERS = {
    "SGD": SGDClassifier,
    "NuSVC": NuSVC,
    "LDA": LinearDiscriminantAnalysis,
    "KNN": KNeighborsClassifier,
    "NC": NearestCentroid,
    "GauNB": GaussianNB,
    "RNC": RadiusNeighborsClassifier,
    "KNFST": KNFSTClassifier,
    "LGBM": LGBMClassifier,
    "MLP": MLPClassifier,
    "TabNet": TabNetClassifier,
    "FTTransformer": FTTransformerClassifier,
    "SAINT": SAINTClassifier,
}


def build_baseline(name: str):
    """Return an instantiated classifier."""
    if name not in _BUILDERS:
        raise ValueError(f"Unknown baseline '{name}'. Valid: {BASELINE_NAMES}")
    if name == "LGBM" and not _HAS_LGBM:
        raise RuntimeError("LightGBM not installed.")
    if name == "TabNet" and not _HAS_TABNET:
        raise RuntimeError("pytorch-tabnet not installed.")
    if name == "FTTransformer" and not _HAS_FTT:
        raise RuntimeError("tab-transformer-pytorch not installed.")
    if name == "SAINT" and not _HAS_SAINT:
        raise RuntimeError("torch not installed.")
    return _BUILDERS[name](**_PARAMS[name])
