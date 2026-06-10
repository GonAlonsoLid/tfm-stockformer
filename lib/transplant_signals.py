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


def peer_graph_signal(daily_y: np.ndarray, d: int,
                      corr_window: int = 120, k: int = 15,
                      mom_window: int = 21) -> np.ndarray:
    """Causal peer-momentum (relational-graph transplant).

    Builds a trailing correlation graph from daily_y[d-corr_window:d] (past only),
    selects each stock's top-k |corr| peers, and returns the correlation-weighted
    mean of those peers' own recent momentum (cumulative return over the last
    mom_window days, also past only). Lead-lag / connected-stock momentum.
    """
    R = _trailing_returns(daily_y, d, corr_window)
    own_R = _trailing_returns(daily_y, d, mom_window)
    if R is None or own_R is None:
        return np.full(daily_y.shape[1], np.nan)
    N = R.shape[1]
    C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    own_mom = own_R.sum(axis=0)  # [N] cumulative past momentum (guarded, strictly < d)
    out = np.full(N, np.nan)
    kk = min(k, N - 1)
    for i in range(N):
        order = np.argsort(np.abs(C[i]))[-kk:]   # top-k peers by |corr|
        wts = C[i, order]                        # signed correlation weights
        denom = np.abs(wts).sum()
        if denom > 1e-12:
            out[i] = float((wts * own_mom[order]).sum() / denom)
    return out


def filtered_trend_signal(daily_y: np.ndarray, d: int,
                          window: int = 120, halflife: float = 10.0,
                          mom_window: int = 21) -> np.ndarray:
    """Causal filtered-trend momentum (wavelet / dual-frequency transplant).

    Extracts the low-frequency (trend) component of each stock's return series via a
    one-sided causal EWMA over daily_y[d-window:d] (strictly past), then takes the mean
    of the most recent mom_window smoothed (denoised) returns.
    """
    R = _trailing_returns(daily_y, d, window)
    if R is None:
        return np.full(daily_y.shape[1], np.nan)
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    sm = np.empty_like(R)
    sm[0] = R[0]
    for t in range(1, R.shape[0]):
        sm[t] = alpha * R[t] + (1.0 - alpha) * sm[t - 1]
    take = min(mom_window, R.shape[0])  # guard: never silently use more than `window` rows
    return sm[-take:].mean(axis=0)
