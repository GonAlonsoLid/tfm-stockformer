#!/usr/bin/env python3
"""
Macro-Economic Feature Builder

Downloads macro-economic indicators (VIX, Treasury yields) and saves them as
feature CSVs in {data_dir}/features/. These provide market regime information
that pure price/volume data cannot capture.

Features generated:
  MACRO_VIX_level.csv      - VIX daily close, time-series z-scored
  MACRO_VIX_change5d.csv   - VIX 5-day percentage change, z-scored
  MACRO_VIX_zscore20d.csv  - VIX z-score over 20-day rolling window
  MACRO_yield_10y2y.csv    - 10Y-2Y Treasury yield spread (TNX - IRX)
  MACRO_yield_change5d.csv - Yield spread 5-day change

All features are time-series z-scored with a 60-day rolling window, then
broadcast to all stocks (same value per date across all tickers).

Usage:
  python scripts/build_macro_features.py --data_dir ./data/Stock_SP500_2018-01-01_2026-03-16
  python scripts/build_macro_features.py --config config/Multitask_Stock_SP500.conf
"""
import argparse
import configparser
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


# ── Constants ─────────────────────────────────────────────────────────────────

MACRO_FILES = [
    "MACRO_VIX_level.csv",
    "MACRO_VIX_change5d.csv",
    "MACRO_VIX_zscore20d.csv",
    "MACRO_yield_10y2y.csv",
    "MACRO_yield_change5d.csv",
]

ROLLING_ZSCORE_WINDOW = 60


# ── Helper functions ──────────────────────────────────────────────────────────

def rolling_zscore(series: pd.Series, window: int = ROLLING_ZSCORE_WINDOW) -> pd.Series:
    """Time-series z-score with a rolling window.

    Normalizes scale while preserving the temporal signal. Uses rolling
    statistics so there is no look-ahead leakage.
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, 1)


def load_tickers(data_dir: str) -> list:
    """Read ticker list from tickers.txt in data_dir."""
    tickers_path = os.path.join(data_dir, "tickers.txt")
    with open(tickers_path) as f:
        return [line.strip() for line in f if line.strip()]


def load_label_index(data_dir: str) -> pd.DatetimeIndex:
    """Load label.csv and return its date index."""
    label_path = os.path.join(data_dir, "label.csv")
    label = pd.read_csv(label_path, index_col=0, parse_dates=True)
    return label.index


def broadcast_to_stocks(
    series: pd.Series,
    dates: pd.DatetimeIndex,
    tickers: list,
) -> pd.DataFrame:
    """Broadcast a scalar time series to a wide DataFrame (dates x tickers).

    1. Reindex to match label.csv dates with forward-fill for missing dates.
    2. Replicate the same value across all ticker columns.
    """
    # Align to target dates
    aligned = series.reindex(dates, method="ffill")
    # Broadcast: same value for every stock on a given date
    df = pd.DataFrame(
        np.tile(aligned.values.reshape(-1, 1), (1, len(tickers))),
        index=dates,
        columns=tickers,
    )
    return df


def download_with_retry(ticker: str, start: str, end: str, retries: int = 2) -> pd.DataFrame:
    """Download data from yfinance with basic retry logic."""
    for attempt in range(retries + 1):
        try:
            data = yf.download(ticker, start=start, end=end, progress=False)
            if data is not None and len(data) > 0:
                return data
        except Exception as e:
            if attempt < retries:
                print(f"  [retry] Attempt {attempt + 1} failed for {ticker}: {e}")
            else:
                raise RuntimeError(f"Failed to download {ticker} after {retries + 1} attempts: {e}")
    return pd.DataFrame()


# ── Main function ─────────────────────────────────────────────────────────────

def main(data_dir: str) -> None:
    """Build macro-economic feature CSVs.

    Args:
        data_dir: Base data directory containing tickers.txt, label.csv,
                  and features/ subdirectory.
    """
    data_dir = str(Path(data_dir).resolve())
    features_dir = os.path.join(data_dir, "features")

    print(f"[config] data_dir     = {data_dir}")
    print(f"[config] features_dir = {features_dir}")

    # ── 1. Check idempotency ──────────────────────────────────────────────────
    existing = [f for f in MACRO_FILES if os.path.exists(os.path.join(features_dir, f))]
    if len(existing) == len(MACRO_FILES):
        print(f"[skip] All {len(MACRO_FILES)} MACRO_* files already exist. Nothing to do.")
        return

    if existing:
        print(f"[info] Found {len(existing)}/{len(MACRO_FILES)} existing MACRO files; regenerating all.")

    # ── 2. Load tickers and date index ────────────────────────────────────────
    tickers = load_tickers(data_dir)
    print(f"[tickers] {len(tickers)} tickers loaded")

    dates = load_label_index(data_dir)
    start_date = str(dates.min().date())
    end_date = str(dates.max().date())
    print(f"[dates] Range: {start_date} to {end_date} ({len(dates)} trading days)")

    # ── 3. Download macro data ────────────────────────────────────────────────
    # Extend start date back by 90 days to have enough history for rolling windows
    download_start = str((dates.min() - pd.Timedelta(days=90)).date())

    print(f"[download] VIX (^VIX) from {download_start} to {end_date}...")
    vix_raw = download_with_retry("^VIX", start=download_start, end=end_date)
    # Handle multi-level columns from yfinance
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_close = vix_raw["Close"].squeeze()
    else:
        vix_close = vix_raw["Close"]
    print(f"  -> {len(vix_close)} rows")

    print(f"[download] 10Y Treasury (^TNX) from {download_start} to {end_date}...")
    tnx_raw = download_with_retry("^TNX", start=download_start, end=end_date)
    if isinstance(tnx_raw.columns, pd.MultiIndex):
        tnx_close = tnx_raw["Close"].squeeze()
    else:
        tnx_close = tnx_raw["Close"]
    print(f"  -> {len(tnx_close)} rows")

    print(f"[download] 3M Treasury (^IRX) from {download_start} to {end_date}...")
    irx_raw = download_with_retry("^IRX", start=download_start, end=end_date)
    if isinstance(irx_raw.columns, pd.MultiIndex):
        irx_close = irx_raw["Close"].squeeze()
    else:
        irx_close = irx_raw["Close"]
    print(f"  -> {len(irx_close)} rows")

    # ── 4. Compute derived series ─────────────────────────────────────────────
    # Yield spread: 10Y - 3M (already in percentage points)
    # Align indices before subtraction
    yield_spread = tnx_close.reindex(tnx_close.index.union(irx_close.index))
    irx_aligned = irx_close.reindex(yield_spread.index)
    yield_spread = yield_spread - irx_aligned
    yield_spread = yield_spread.dropna()

    # ── 5. Build feature definitions ──────────────────────────────────────────
    features = {
        "MACRO_VIX_level": rolling_zscore(vix_close),
        "MACRO_VIX_change5d": rolling_zscore(vix_close.pct_change(5)),
        "MACRO_VIX_zscore20d": rolling_zscore(
            # Inner z-score: 20-day rolling z-score of VIX
            (vix_close - vix_close.rolling(20).mean()) / vix_close.rolling(20).std().replace(0, 1)
        ),
        "MACRO_yield_10y2y": rolling_zscore(yield_spread),
        "MACRO_yield_change5d": rolling_zscore(yield_spread.pct_change(5)),
    }

    # ── 6. Broadcast and save ─────────────────────────────────────────────────
    os.makedirs(features_dir, exist_ok=True)

    for name, series in features.items():
        print(f"[build] {name}...")

        # Drop NaN from rolling window warmup
        series = series.dropna()

        # Broadcast to all stocks, aligned to label.csv dates
        df = broadcast_to_stocks(series, dates, tickers)

        # Replace any remaining NaN with 0.0 (edges of rolling windows)
        df = df.fillna(0.0)

        # Replace inf/-inf with 0.0
        df = df.replace([float("inf"), float("-inf")], 0.0)

        out_path = os.path.join(features_dir, f"{name}.csv")
        df.to_csv(out_path)
        print(f"  -> Saved {out_path} | shape={df.shape} | NaN={df.isna().sum().sum()}")

    # ── 7. Validation summary ─────────────────────────────────────────────────
    print("\n[validate] Macro feature summary:")
    for name in features:
        fpath = os.path.join(features_dir, f"{name}.csv")
        df = pd.read_csv(fpath, index_col=0)
        nan_count = df.isna().sum().sum()
        inf_count = int(np.isinf(df.values).sum())
        print(f"  {name}: shape={df.shape}, NaN={nan_count}, Inf={inf_count}")

    print(f"\n[done] {len(features)} macro feature CSVs written to {features_dir}")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build macro-economic feature CSVs (VIX, yield spread) for Stockformer."
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Base data directory (e.g., ./data/Stock_SP500_2018-01-01_2026-03-16)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to .conf config file (derives data_dir from [file] section)",
    )
    args = parser.parse_args()

    # Resolve data_dir from arguments
    if args.data_dir:
        resolved_data_dir = args.data_dir
    elif args.config:
        cfg = configparser.ConfigParser()
        cfg.read(args.config)
        resolved_data_dir = cfg["file"]["data_dir"]
        resolved_data_dir = str(Path(resolved_data_dir).resolve())
    else:
        parser.error("Either --data_dir or --config must be provided.")
        sys.exit(1)

    main(data_dir=resolved_data_dir)
