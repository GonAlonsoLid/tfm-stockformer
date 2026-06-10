"""Causal cross-sectional 'architecture-transplant' signals for the RQ2 signal search.

Each signal has the convention used by resid_mom: f(daily_y, d) -> [N], computed using
ONLY daily_y[:d] (strictly before the decision day d). This guarantees no look-ahead.
"""
from __future__ import annotations

import numpy as np


def _trailing_returns(daily_y: np.ndarray, d: int, window: int) -> np.ndarray | None:
    """Trailing [window, N] daily returns ending at (and excluding) decision day d.

    Returns None if there is not enough history (d - window < 0). NaNs -> 0.0.
    """
    a = d - window
    if a < 0:
        return None
    return np.nan_to_num(daily_y[a:d], nan=0.0)


def xz(v: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score (mean 0, std 1). Constant/empty -> zeros."""
    v = np.asarray(v, dtype=float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    return (v - m) / s if s > 1e-12 else np.zeros_like(v)
