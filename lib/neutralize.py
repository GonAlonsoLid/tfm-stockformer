"""Cross-sectional signal neutralization (TFM returns pipeline).

Residualizes an alpha signal against risk exposures (market beta, size, sector,
Fama-French factors) on a single rebalance date, so the portfolio carries no
unintended factor/sector bet — essential for a market-neutral book.

Standard practice: per-date OLS of signal on exposures; the residual (optionally
standardized cross-sectionally) is the neutral signal.
"""
from __future__ import annotations

import numpy as np


def neutralize(signal: np.ndarray, exposures: np.ndarray,
               standardize: bool = True) -> np.ndarray:
    """Residualize ``signal`` (length N) on ``exposures`` (N x K) via OLS.

    A constant is always added. Rows with NaN in the signal or any exposure are
    excluded from the fit and returned as NaN. If ``standardize`` and the
    residual has non-zero spread, the residual is z-scored cross-sectionally.
    """
    signal = np.asarray(signal, dtype=float).reshape(-1)
    exposures = np.asarray(exposures, dtype=float)
    if exposures.ndim == 1:
        exposures = exposures.reshape(-1, 1)
    n = signal.shape[0]

    out = np.full(n, np.nan)
    valid = ~np.isnan(signal) & ~np.isnan(exposures).any(axis=1)
    if valid.sum() < exposures.shape[1] + 1:
        return out  # not enough points to fit

    X = np.column_stack([np.ones(valid.sum()), exposures[valid]])
    y = signal[valid]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef

    if standardize:
        std = resid.std(ddof=0)
        if std > 1e-12:
            resid = (resid - resid.mean()) / std
    out[valid] = resid
    return out
