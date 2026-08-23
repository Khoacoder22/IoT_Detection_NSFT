"""CPAI: Controlled-Point Discriminant Analysis for IoT Network Intrusion Detection."""

# Preload torch BEFORE anything that pulls in scipy.linalg. On macOS, importing
# scipy (via Accelerate/libomp) before torch causes TabNet fit() to segfault
# because libomp gets double-initialized. Harmless if torch isn't installed.
try:
    import torch  # noqa: F401
except ImportError:
    pass

from .paths import PROJECT_ROOT, DATA_DIR, RESULTS_DIR
from .datasets import load_dataset, DATASETS
from .models import HHH, HHHv2, KNFST, SpectralNFST
from .baselines import build_baseline, BASELINE_NAMES
from .metrics import calc_index, evaluate
from .preprocessing import preprocess
