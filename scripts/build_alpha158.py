#!/usr/bin/env python3
"""
Alpha158 Feature Builder — extends Alpha360 with technical analysis indicators.

Reads OHLCV parquet files from {data_dir}/ohlcv/ and generates ~15 additional
feature CSVs in {data_dir}/features/, all cross-sectionally z-scored per day
(same normalization as build_alpha360.py).

Usage:
    python scripts/build_alpha158.py --data_dir data/Stock_SP500_2018-01-01_2026-03-16
    python scripts/build_alpha158.py --config config/Multitask_Stock_SP500.conf

Features generated (15 total):
    RSI_6, RSI_12, RSI_24           - Relative Strength Index
    BBANDS_width_20, BBANDS_pctb_20 - Bollinger Bands width and %B
    MACD_signal, MACD_hist          - MACD signal line and histogram
    ATR_14, ATR_21                  - Average True Range (normalized by close)
    ROC_5, ROC_10, ROC_20           - Rate of Change
    RVOL_5, RVOL_10, RVOL_20       - Realized volatility (annualized)
"""
import argparse
import configparser
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────────

# Alpha360 drops the first 60 rows due to lag buffer; we must align to the same
# date range so that all feature CSVs share the same DatetimeIndex.
LAG_BUFFER = 60

FEATURE_NAMES = [
    "RSI_6", "RSI_12", "RSI_24",
    "BBANDS_width_20", "BBANDS_pctb_20",
    "MACD_signal", "MACD_hist",
    "ATR_14", "ATR_21",
    "ROC_5", "ROC_10", "ROC_20",
    "RVOL_5", "RVOL_10", "RVOL_20",
]


# ── Helper functions ──────────────────────────────────────────────────────────

def load_tickers(data_dir: str) -> list:
    """Read ticker list from tickers.txt in data_dir. One ticker per line."""
    tickers_path = os.path.join(data_dir, "tickers.txt")
    with open(tickers_path) as f:
        return [line.strip() for line in f if line.strip()]


def load_wide(ohlcv_dir: str, tickers: list, field: str) -> pd.DataFrame:
    """Load one OHLCV field into wide DataFrame: rows=dates, cols=tickers.

    Column order is enforced from the given tickers list (not from os.listdir).
    """
    frames = {}
    for ticker in tickers:
        path = os.path.join(ohlcv_dir, f"{ticker}.parquet")
        df = pd.read_parquet(path)
        frames[ticker] = df[field]
    return pd.DataFrame(frames)[tickers]


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: normalize each row across tickers (axis=1).

    Uses axis=1 (across tickers per day) — NOT axis=0 (time-series), which would
    introduce look-ahead leakage.
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1)
    result = df.sub(row_mean, axis=0).div(row_std.replace(0, 1), axis=0)
    return result.fillna(0.0)


def clean_and_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN, apply cross-sectional z-score, fill remaining NaN."""
    df = df.replace([float("inf"), float("-inf")], float("nan"))
    return zscore_rows(df)


# ── Indicator computation (vectorized over all stocks) ────────────────────────

def compute_rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """RSI computed column-wise (per stock) using rolling mean of gains/losses."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bbands_width(close: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands width: (upper - lower) / sma."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma.replace(0, float("nan"))
    return width


def compute_bbands_pctb(close: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands %B: (close - lower) / (upper - lower)."""
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    band_range = (upper - lower).replace(0, float("nan"))
    pctb = (close - lower) / band_range
    return pctb


def compute_macd_signal(close: pd.DataFrame) -> pd.DataFrame:
    """MACD signal line (EMA-9 of MACD line)."""
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return signal


def compute_macd_hist(close: pd.DataFrame) -> pd.DataFrame:
    """MACD histogram (MACD line - signal line)."""
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd - signal
    return hist


def compute_atr_norm(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Average True Range normalized by close price."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    # Element-wise max across three true range components
    tr = tr1.where(tr1 >= tr2, tr2)
    tr = tr.where(tr >= tr3, tr3)
    atr = tr.rolling(window).mean()
    atr_norm = atr / close.replace(0, float("nan"))
    return atr_norm


def compute_roc(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Rate of Change: (close / close[t-n]) - 1."""
    shifted = close.shift(n).replace(0, float("nan"))
    return (close / shifted) - 1


def compute_rvol(close: pd.DataFrame, n: int) -> pd.DataFrame:
    """Realized volatility: rolling std of log returns, annualized."""
    log_returns = np.log(close / close.shift(1).replace(0, float("nan")))
    return log_returns.rolling(n).std() * np.sqrt(252)


# ── Main function ─────────────────────────────────────────────────────────────

def main(data_dir: str) -> None:
    """Build 15 Alpha158-style feature CSVs from OHLCV Parquet files.

    Args:
        data_dir: Path to base data directory containing ohlcv/ and features/.
    """
    data_dir = str(Path(data_dir).resolve())
    ohlcv_dir = os.path.join(data_dir, "ohlcv")
    features_dir = os.path.join(data_dir, "features")

    print(f"[config] data_dir     = {data_dir}")
    print(f"[config] ohlcv_dir    = {ohlcv_dir}")
    print(f"[config] features_dir = {features_dir}")

    # ── 1. Check idempotency: skip if all 15 feature files already exist ──────
    os.makedirs(features_dir, exist_ok=True)
    existing = {f for f in os.listdir(features_dir) if f.endswith(".csv")}
    expected_files = {f"{name}.csv" for name in FEATURE_NAMES}
    if expected_files.issubset(existing):
        print(f"[skip] All {len(FEATURE_NAMES)} Alpha158 features already exist. Nothing to do.")
        return

    # ── 2. Load tickers ───────────────────────────────────────────────────────
    tickers = load_tickers(data_dir)
    print(f"[tickers] {len(tickers)} tickers loaded from tickers.txt")

    # ── 3. Load OHLCV wide DataFrames ─────────────────────────────────────────
    print("[load] Loading OHLCV Parquet files...")
    df_close = load_wide(ohlcv_dir, tickers, "Close")
    df_high = load_wide(ohlcv_dir, tickers, "High")
    df_low = load_wide(ohlcv_dir, tickers, "Low")
    # Volume not needed for current indicators but loaded if future features need it
    print(f"[load] Date range: {df_close.index[0]} to {df_close.index[-1]}, {len(df_close)} rows")

    # ── 4. Build feature registry ─────────────────────────────────────────────
    # Each entry: (feature_name, raw_df) — raw_df is z-scored and trimmed before saving.
    feature_registry = [
        ("RSI_6",           lambda: compute_rsi(df_close, 6)),
        ("RSI_12",          lambda: compute_rsi(df_close, 12)),
        ("RSI_24",          lambda: compute_rsi(df_close, 24)),
        ("BBANDS_width_20", lambda: compute_bbands_width(df_close, 20, 2.0)),
        ("BBANDS_pctb_20",  lambda: compute_bbands_pctb(df_close, 20, 2.0)),
        ("MACD_signal",     lambda: compute_macd_signal(df_close)),
        ("MACD_hist",       lambda: compute_macd_hist(df_close)),
        ("ATR_14",          lambda: compute_atr_norm(df_high, df_low, df_close, 14)),
        ("ATR_21",          lambda: compute_atr_norm(df_high, df_low, df_close, 21)),
        ("ROC_5",           lambda: compute_roc(df_close, 5)),
        ("ROC_10",          lambda: compute_roc(df_close, 10)),
        ("ROC_20",          lambda: compute_roc(df_close, 20)),
        ("RVOL_5",          lambda: compute_rvol(df_close, 5)),
        ("RVOL_10",         lambda: compute_rvol(df_close, 10)),
        ("RVOL_20",         lambda: compute_rvol(df_close, 20)),
    ]

    # ── 5. Generate and save each feature ─────────────────────────────────────
    written = 0
    for name, compute_fn in feature_registry:
        out_path = os.path.join(features_dir, f"{name}.csv")

        # Skip individual features that already exist
        if os.path.exists(out_path):
            print(f"  [{name}] already exists, skipping")
            written += 1
            continue

        print(f"  [{name}] computing...", end=" ", flush=True)

        raw = compute_fn()
        zscored = clean_and_zscore(raw)

        # Trim to match Alpha360 date range (drop first LAG_BUFFER rows)
        zscored = zscored.iloc[LAG_BUFFER:]

        zscored.to_csv(out_path)
        written += 1
        print(f"saved ({zscored.shape[0]} dates x {zscored.shape[1]} tickers)")

    print(f"\n[done] Written {written}/{len(FEATURE_NAMES)} Alpha158 feature CSVs to {features_dir}")

    # ── 6. Validation summary ─────────────────────────────────────────────────
    for name in FEATURE_NAMES:
        path = os.path.join(features_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"[warn] Missing: {name}.csv")
            continue
    sample = pd.read_csv(os.path.join(features_dir, f"{FEATURE_NAMES[0]}.csv"), index_col=0)
    print(f"[validate] Sample shape: {sample.shape}")
    print(f"[validate] First date:   {sample.index[0]}")
    nan_count = sample.isna().sum().sum()
    inf_count = int(np.isinf(sample.values).sum())
    print(f"[validate] NaN count: {nan_count} (expected 0)")
    print(f"[validate] Inf count: {inf_count} (expected 0)")


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Alpha158 technical analysis feature CSVs from OHLCV Parquet files."
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to data directory (e.g., data/Stock_SP500_2018-01-01_2026-03-16)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to .conf config file (reads data_dir from [file] section)",
    )
    args = parser.parse_args()

    # Resolve data_dir from either --data_dir or --config
    if args.data_dir:
        data_dir = args.data_dir
    elif args.config:
        cfg = configparser.ConfigParser()
        cfg.read(args.config)
        alpha_360_dir = cfg["file"]["alpha_360_dir"]
        data_dir = str(Path(alpha_360_dir).resolve().parent)
    else:
        parser.error("Either --data_dir or --config must be provided.")
        sys.exit(1)

    main(data_dir=data_dir)
