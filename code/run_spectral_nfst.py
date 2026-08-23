#!/usr/bin/env python3
"""Backward-compatible alias for the Spectral NFST classification CLI.

New code should call ``run_spectral_nfst_classification.py``. This wrapper keeps
all commands documented before the classification/novelty split working without
duplicating experiment logic.
"""

from cpai.paths import RESULTS_DIR
from run_spectral_nfst_classification import main


if __name__ == "__main__":
    # Preserve the original runner's default output path exactly.
    raise SystemExit(main(RESULTS_DIR / "spectral_nfst" / "summary.csv"))
