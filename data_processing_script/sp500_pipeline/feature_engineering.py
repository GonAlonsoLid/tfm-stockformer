"""Feature engineering for S&P500 OHLCV data.

Computes momentum (ROC), RSI, MACD, Bollinger Bands, and volume ratios per ticker,
applies cross-sectional z-score normalization per trading day, and saves per-feature
CSVs in the format StockDataset expects ([T, N] with DatetimeIndex + ticker columns).

Also computes label.csv with raw 1-day forward returns (NOT normalized).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta  # noqa: F401
except ImportError:
    ta = None  # type: ignore[assignment]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _require_pandas_ta() -> None:
    """Raise ImportError if pandas_ta is not installed."""
    if ta is None:
        raise ImportError(
            "pandas_ta is required but not installed. "
            "Install it with: pip install pandas-ta"
        )


def _roc(series: pd.Series, length: int) -> pd.Series:
    """Rate of Change: percentage change over `length` periods."""
    return series.pct_change(length) * 100


def _rsi(series: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index (Wilder smoothing via EMA)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length, adjust=False).mean()
    avg_loss_safe = avg_loss.where(avg_loss >= 1e-8, other=1e-8)
    rs = avg_gain / avg_loss_safe
    return 100.0 - (100.0 / (1.0 + rs))


def _macd_line(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """MACD line: EMA(fast) - EMA(slow)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def _bbands(
    series: pd.Series, length: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series]:
    """Bollinger Bands: (upper, lower)."""
    sma = series.rolling(length).mean()
    std = series.rolling(length).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return upper, lower


def _cross_sectional_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional z-score: for each row, z-score across N stocks."""
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1)
    row_std = row_std.where(row_std >= 1e-8, other=1.0)
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


# ── Public API ─────────────────────────────────────────────────────────────────

def compute_features(
    df_ohlcv: pd.DataFrame,
    windows: Tuple[int, ...] = (5, 10, 20, 60),
) -> pd.DataFrame:
    """Compute TA features for a single ticker.

    Parameters
    ----------
    df_ohlcv : DataFrame with columns [Open, High, Low, Close, Volume] and a DatetimeIndex.
    windows : Rolling windows to use for ROC, RSI, and VOL_ratio.

    Returns
    -------
    DataFrame with 16 feature columns (same DatetimeIndex as input).
    NaN rows from warmup are NOT dropped here — caller handles that.
    """
    close = df_ohlcv["Close"]
    volume = df_ohlcv["Volume"]
    features: Dict[str, pd.Series] = {}

    # ROC
    for w in windows:
        features[f"ROC_{w}"] = _roc(close, w)

    # RSI
    for w in windows:
        features[f"RSI_{w}"] = _rsi(close, w)

    # MACD (standard 12/26/9 — only the MACD line)
    features["MACD"] = _macd_line(close, fast=12, slow=26)

    # Bollinger Bands — standard period = 20
    bb_upper, bb_lower = _bbands(close, length=20)
    features["BB_upper_20"] = bb_upper
    features["BB_lower_20"] = bb_lower
    features["BB_width_20"] = bb_upper - bb_lower

    # VOL_ratio
    for w in windows:
        vol_ma = volume.rolling(w).mean()
        features[f"VOL_ratio_{w}"] = volume / vol_ma.replace(0, np.nan)

    return pd.DataFrame(features, index=df_ohlcv.index)


def build_feature_matrix(
    ohlcv_dir: str,
    tickers: list,
) -> Dict[str, pd.DataFrame]:
    """Build a dict of wide feature DataFrames, one per feature name.

    Parameters
    ----------
    ohlcv_dir : Directory containing ``{ticker}.parquet`` files.
    tickers : List of ticker symbols to include.

    Returns
    -------
    Dict mapping feature_name -> DataFrame[T, N] (raw, pre-normalization).
    First 60 rows (warmup for 60-day window) are dropped.
    """
    ohlcv_dir = str(ohlcv_dir)
    per_ticker: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        parquet_path = os.path.join(ohlcv_dir, f"{ticker}.parquet")
        df = pd.read_parquet(parquet_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        per_ticker[ticker] = compute_features(df)

    if not per_ticker:
        return {}

    # Common DatetimeIndex (intersection across all tickers)
    common_index = per_ticker[tickers[0]].index
    for ticker in tickers[1:]:
        common_index = common_index.intersection(per_ticker[ticker].index)

    # Drop first 60 rows as warmup
    common_index = common_index[60:]

    # Get feature column names from first ticker
    feature_cols = list(per_ticker[tickers[0]].columns)

    feature_dict: Dict[str, pd.DataFrame] = {}
    for feat in feature_cols:
        wide = pd.DataFrame(
            {ticker: per_ticker[ticker].loc[common_index, feat] for ticker in tickers},
            index=common_index,
        )
        feature_dict[feat] = wide

    return feature_dict


def save_feature_csvs(
    feature_dict: Dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """Apply cross-sectional normalization and save one CSV per feature.

    Parameters
    ----------
    feature_dict : Dict mapping feature_name -> DataFrame[T, N] (raw values).
    output_dir : Root output directory; CSVs are written to ``output_dir/features/``.

    Notes
    -----
    - The CSVs saved on disk are NORMALIZED (cross-sectional z-score per day).
    - The in-memory ``feature_dict`` values remain raw.
    - Each CSV: index=DatetimeIndex, columns=ticker symbols (shape [T, N]).
    """
    features_dir = Path(output_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    for feature_name, wide_df in feature_dict.items():
        if not isinstance(wide_df.index, pd.DatetimeIndex):
            raise ValueError(
                f"Feature '{feature_name}': index must be DatetimeIndex, "
                f"got {type(wide_df.index).__name__}"
            )
        # Verify orientation [T, N]: dates on rows, tickers on columns
        if wide_df.shape[0] < wide_df.shape[1]:
            # Likely transposed — raise to catch orientation bugs early
            raise ValueError(
                f"Feature '{feature_name}': DataFrame appears transposed "
                f"(shape {wide_df.shape}). Expected [T, N] with T > N."
            )
        normalized_df = _cross_sectional_normalize(wide_df)
        out_path = features_dir / f"{feature_name}.csv"
        normalized_df.to_csv(out_path)


def compute_label_csv(
    ohlcv_dir: str,
    tickers: list,
    output_dir: str,
) -> None:
    """Compute 1-day forward returns and save as label.csv (raw, NOT normalized).

    Parameters
    ----------
    ohlcv_dir : Directory containing ``{ticker}.parquet`` files.
    tickers : List of ticker symbols.
    output_dir : Root output directory; label.csv written here.

    Notes
    -----
    - Forward return: close_{t+1} / close_t - 1
    - Last row is dropped (no forward price available).
    - NaN filled with 0.
    - Saved as ``output_dir/label.csv`` with DatetimeIndex rows and ticker columns.
    """
    ohlcv_dir = str(ohlcv_dir)
    close_dict: Dict[str, pd.Series] = {}
    for ticker in tickers:
        parquet_path = os.path.join(ohlcv_dir, f"{ticker}.parquet")
        df = pd.read_parquet(parquet_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        close_dict[ticker] = df["Close"]

    close = pd.DataFrame(close_dict)
    # Forward return: shift(-1) gives tomorrow's close
    forward_return = close.shift(-1) / close - 1
    # Drop last row (no future price)
    forward_return = forward_return.iloc[:-1]
    forward_return = forward_return.fillna(0)

    out_path = Path(output_dir) / "label.csv"
    forward_return.to_csv(out_path)


def main() -> None:
    """CLI entry point for feature engineering pipeline."""
    parser = argparse.ArgumentParser(
        description="Compute TA features and label.csv from OHLCV Parquet files."
    )
    parser.add_argument("--data_dir", required=True, help="Root data directory")
    parser.add_argument(
        "--ohlcv_subdir",
        default="ohlcv",
        help="Subdirectory under data_dir containing Parquet files (default: ohlcv)",
    )
    parser.add_argument(
        "--tickers_file",
        default="tickers.txt",
        help="File listing ticker symbols, one per line (default: tickers.txt)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    ohlcv_dir = data_dir / args.ohlcv_subdir
    tickers_path = data_dir / args.tickers_file

    with open(tickers_path) as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"Computing features for {len(tickers)} tickers from {ohlcv_dir} ...")
    feature_dict = build_feature_matrix(str(ohlcv_dir), tickers)
    save_feature_csvs(feature_dict, str(data_dir))
    print(f"Saved {len(feature_dict)} feature CSVs to {data_dir / 'features'}/")

    compute_label_csv(str(ohlcv_dir), tickers, str(data_dir))
    print(f"Saved label.csv to {data_dir}/")


if __name__ == "__main__":
    main()
