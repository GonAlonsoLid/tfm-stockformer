"""Alternative signal decomposition methods to replace DWT(Sym2).

Provides STL (Seasonal-Trend-Loess) and VMD (Variational Mode Decomposition)
as safer alternatives that avoid potential information leakage.

Research backing:
- STL: inherently leak-free, used in STGAT (2025) for stock prediction
- VMD: 79.84% MAPE reduction vs LSTM standalone (Scientific Reports, 2025)
- DWT(db4) > DWT(Sym2) for S&P 500 direction prediction (arXiv:2408.12408)

Usage:
    from lib.decomposition import stl_decompose_batch, sliding_vmd_batch

    # Replace disentangle() call in StockDataset:
    XL, XH = stl_decompose_batch(X, period=5)
"""

import numpy as np


def stl_decompose_batch(data, period=5):
    """STL decomposition: split into trend (low-freq) and residual (high-freq).

    Uses statsmodels STL per stock per sample. Safe against leakage since
    STL operates on each window independently.

    Parameters
    ----------
    data : np.ndarray
        Shape [B, T, N] — batch of sliding windows.
    period : int
        Seasonal period in trading days (default 5 = weekly cycle).

    Returns
    -------
    data_low : np.ndarray
        Trend + seasonal component, shape [B, T, N].
    data_high : np.ndarray
        Residual component, shape [B, T, N].
    """
    from statsmodels.tsa.seasonal import STL

    B, T, N = data.shape
    data_low = np.zeros_like(data)
    data_high = np.zeros_like(data)

    # STL requires period >= 2 and series length > 2*period
    if T <= 2 * period:
        # Fallback: simple moving average decomposition
        kernel = np.ones(period) / period
        for b in range(B):
            for n in range(N):
                series = data[b, :, n]
                trend = np.convolve(series, kernel, mode='same')
                data_low[b, :, n] = trend
                data_high[b, :, n] = series - trend
        return data_low, data_high

    for b in range(B):
        for n in range(N):
            series = data[b, :, n]
            try:
                stl = STL(series, period=period, robust=True)
                result = stl.fit()
                data_low[b, :, n] = result.trend + result.seasonal
                data_high[b, :, n] = result.resid
            except Exception:
                # Fallback for degenerate series
                data_low[b, :, n] = series
                data_high[b, :, n] = 0.0

    return data_low, data_high


def moving_average_decompose_batch(data, window=5):
    """Simple moving average decomposition (fast fallback).

    Parameters
    ----------
    data : np.ndarray
        Shape [B, T, N].
    window : int
        Moving average window size.

    Returns
    -------
    data_low, data_high : np.ndarray
        Both shape [B, T, N].
    """
    B, T, N = data.shape
    data_low = np.zeros_like(data)
    data_high = np.zeros_like(data)
    kernel = np.ones(window) / window

    for b in range(B):
        for n in range(N):
            series = data[b, :, n]
            trend = np.convolve(series, kernel, mode='same')
            data_low[b, :, n] = trend
            data_high[b, :, n] = series - trend

    return data_low, data_high


def sliding_vmd_batch(data, window=60, K=3, alpha=2000, tau=0, tol=1e-7):
    """Variational Mode Decomposition on sliding windows (leak-free).

    Applies VMD to overlapping windows of the input, preventing look-ahead.
    Returns low-frequency (sum of first K-1 modes) and high-frequency (last mode).

    Requires: pip install vmdpy

    Parameters
    ----------
    data : np.ndarray
        Shape [B, T, N].
    window : int
        VMD window size (should be <= T).
    K : int
        Number of modes to decompose into.
    alpha, tau, tol : float
        VMD hyperparameters.

    Returns
    -------
    data_low, data_high : np.ndarray
        Both shape [B, T, N].
    """
    try:
        from vmdpy import VMD
    except ImportError:
        print("WARNING: vmdpy not installed. Falling back to moving average decomposition.")
        return moving_average_decompose_batch(data, window=5)

    B, T, N = data.shape
    data_low = np.zeros_like(data)
    data_high = np.zeros_like(data)

    actual_window = min(window, T)

    for b in range(B):
        for n in range(N):
            series = data[b, :, n]
            # Apply VMD to the last `window` points (no future data)
            segment = series[-actual_window:]
            try:
                modes, _, _ = VMD(segment, alpha, tau, K, 0, 1, tol)
                # Low-freq = sum of all modes except last
                low = np.sum(modes[:-1], axis=0)
                high = modes[-1]
                # Map back to full length
                if actual_window < T:
                    # For the prefix, use simple MA
                    prefix = series[:T - actual_window]
                    kernel = np.ones(5) / 5
                    prefix_low = np.convolve(prefix, kernel, mode='same')
                    data_low[b, :T - actual_window, n] = prefix_low
                    data_high[b, :T - actual_window, n] = prefix - prefix_low
                data_low[b, -actual_window:, n] = low
                data_high[b, -actual_window:, n] = high
            except Exception:
                data_low[b, :, n] = series
                data_high[b, :, n] = 0.0

    return data_low, data_high
