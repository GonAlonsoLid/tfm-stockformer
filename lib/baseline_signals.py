"""Zero-cost sanity baseline signals for the complexity ladder (TFM).

These set the "zero skill" floor of the ladder: no training, no parameters.
Each returns a [T, N] score matrix on the panel grid; the runner slices the
test window and evaluates it with the same harness as every other model.

Signals are built from the Alpha360 CLOSE_d{k} columns (z-scored trailing
returns), so they use only past information — no look-ahead.
"""
from __future__ import annotations

import numpy as np


def zero_signal(X: np.ndarray, feature_names: list[str]) -> np.ndarray:
    """Predict zero for every stock — the literal no-skill anchor."""
    T, N, _ = X.shape
    return np.zeros((T, N))


def momentum_signal(X: np.ndarray, feature_names: list[str], lags: int = 20) -> np.ndarray:
    """Cross-sectional momentum: mean of the most recent CLOSE_d{1..lags} columns."""
    idx = _close_lag_indices(feature_names, lags)
    if not idx:
        raise ValueError("No CLOSE_d{k} features found for momentum signal")
    return X[:, :, idx].mean(axis=2)


def reversal_signal(X: np.ndarray, feature_names: list[str], lags: int = 5) -> np.ndarray:
    """Short-term reversal: negative of the recent CLOSE_d{1..lags} momentum."""
    return -momentum_signal(X, feature_names, lags=lags)


def _close_lag_indices(feature_names: list[str], lags: int) -> list[int]:
    """Column indices of CLOSE_d1 .. CLOSE_d{lags}, in lag order."""
    wanted = {f"CLOSE_d{k}": k for k in range(1, lags + 1)}
    pairs = [(wanted[name], i) for i, name in enumerate(feature_names) if name in wanted]
    return [i for _, i in sorted(pairs)]
