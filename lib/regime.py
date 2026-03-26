"""
VIX-based market regime detection for position scaling.

Regimes:
    - Low  (VIX < 15):  calm markets  -> 1.0x position size
    - Normal (15 <= VIX <= 25): average volatility -> 0.7x position size
    - High (VIX > 25): stressed markets -> 0.3x position size
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Regime thresholds and scalers ────────────────────────────────────────────

VIX_LOW_THRESHOLD = 15
VIX_HIGH_THRESHOLD = 25

REGIME_SCALERS = {
    "low": 1.0,
    "normal": 0.7,
    "high": 0.3,
}


# ── Public API ───────────────────────────────────────────────────────────────


def classify_regime(vix_series: pd.Series) -> pd.Series:
    """Classify each VIX value into a regime label.

    Args:
        vix_series: Series with DatetimeIndex and raw VIX closing levels.

    Returns:
        Series of regime labels ('low', 'normal', 'high') with the same index.
    """
    conditions = [
        vix_series < VIX_LOW_THRESHOLD,
        vix_series > VIX_HIGH_THRESHOLD,
    ]
    choices = ["low", "high"]
    return pd.Series(
        pd.Categorical(
            values=_np_select(conditions, choices, default="normal"),
            categories=["low", "normal", "high"],
        ),
        index=vix_series.index,
        name="regime",
    )


def get_regime_scaler(vix_series: pd.Series) -> pd.Series:
    """Return a position-scaling factor per date based on VIX regime.

    Args:
        vix_series: Series with DatetimeIndex and raw VIX closing levels.

    Returns:
        Series of floats (1.0 / 0.7 / 0.3) with the same index.
    """
    regimes = classify_regime(vix_series)
    return regimes.map(REGIME_SCALERS).astype(float).rename("regime_scaler")


def load_vix(
    start: str | None = None,
    end: str | None = None,
    features_dir: str | None = None,
) -> pd.Series:
    """Load VIX closing prices, trying yfinance first with a CSV fallback.

    Args:
        start: Start date string (e.g. '2020-01-01'). Required for yfinance.
        end: End date string (e.g. '2024-12-31'). Required for yfinance.
        features_dir: Path to features directory containing MACRO_VIX_level.csv
            as fallback if yfinance is unavailable.

    Returns:
        Series with DatetimeIndex and VIX closing levels.

    Raises:
        RuntimeError: If VIX data cannot be loaded from any source.
    """
    # Try yfinance first
    vix = _load_vix_yfinance(start, end)
    if vix is not None:
        return vix

    # Fallback to features directory
    if features_dir is not None:
        vix = _load_vix_from_features(features_dir)
        if vix is not None:
            return vix

    raise RuntimeError(
        "Could not load VIX data. Install yfinance (`pip install yfinance`) "
        "or provide a features_dir containing MACRO_VIX_level.csv."
    )


# ── Private helpers ──────────────────────────────────────────────────────────


def _np_select(conditions, choices, default="normal"):
    """Thin wrapper around numpy.select to keep the top-level import light."""
    import numpy as np

    return np.select(conditions, choices, default=default)


def _load_vix_yfinance(
    start: str | None,
    end: str | None,
) -> pd.Series | None:
    """Attempt to download VIX from yfinance. Returns None on failure."""
    if start is None or end is None:
        logger.debug("start/end not provided; skipping yfinance download.")
        return None
    try:
        import yfinance as yf

        df = yf.download("^VIX", start=start, end=end, progress=False)
        if df.empty:
            logger.warning("yfinance returned empty VIX data.")
            return None
        close = df["Close"].squeeze()
        close.index = pd.DatetimeIndex(close.index)
        close.name = "vix"
        return close
    except ImportError:
        logger.info("yfinance not installed; falling back to CSV.")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance download failed: %s", exc)
        return None


def _load_vix_from_features(features_dir: str) -> pd.Series | None:
    """Load VIX from a MACRO_VIX_level.csv file in the features directory."""
    csv_path = Path(features_dir) / "MACRO_VIX_level.csv"
    if not csv_path.exists():
        logger.warning("VIX CSV not found at %s", csv_path)
        return None
    try:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        col = df.columns[0]
        series = df[col].rename("vix")
        series.index = pd.DatetimeIndex(series.index)
        return series
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to read VIX CSV: %s", exc)
        return None
