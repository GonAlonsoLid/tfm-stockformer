#!/usr/bin/env python3
"""
Alpha360 Feature Builder — Phase 8

Replaces 69 TA feature CSVs with 360 Alpha360-style price/volume ratio features.

Usage: python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf

Feature set (6 fields x 60 lags = 360 features per stock per day):
  CLOSE_d{d} = Close[t] / Close[t-d]
  OPEN_d{d}  = Open[t]  / Close[t-d]
  HIGH_d{d}  = High[t]  / Close[t-d]
  LOW_d{d}   = Low[t]   / Close[t-d]
  VWAP_d{d}  = ((High[t]+Low[t]+Close[t])/3) / Close[t-d]
  VOL_d{d}   = Volume[t] / Volume[t-d]
  Where d = 1, 2, 3, ..., 60

All features are cross-sectionally z-scored per day. NaN and inf values are
replaced with 0.0 (the cross-sectional mean after z-scoring).
"""
import argparse
import configparser
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # noqa: N801
        """Minimal context-manager-compatible tqdm fallback."""
        def __init__(self, total=None, desc=None, **kw):
            self._total = total
            self._desc = desc
            self._n = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            self._n += n


# ── Constants ─────────────────────────────────────────────────────────────────

PRICE_FIELDS = ["Close", "Open", "High", "Low"]  # numerator field names in Parquet
LAGS = list(range(1, 61))                          # d = 1..60

# LAG_BUFFER: number of rows to drop from the beginning of the output.
# After shift(d=60), rows 0..59 are NaN. Slicing at [60:] gives
# first valid row = ohlcv index[60] = 2018-03-29 (verified live).
LAG_BUFFER = 60


# ── Helper functions ───────────────────────────────────────────────────────────

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
    return pd.DataFrame(frames)[tickers]  # enforce column order from tickers list


def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score: normalize each row across tickers (axis=1).

    Uses axis=1 (across tickers per day) — NOT axis=0 (time-series), which would
    introduce look-ahead leakage.
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1)
    # std=0 rows (degenerate days with all-identical values) produce NaN;
    # fillna(0.0) is applied downstream in safe_ratio() to handle this case.
    return df.sub(row_mean, axis=0).div(row_std, axis=0)


def safe_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    """Compute numerator/denominator with zero-safe division, z-score, and NaN/inf cleanup.

    Zero denominator handling: replace 0 with NaN BEFORE dividing.
    Using fillna AFTER division is insufficient — division by zero yields inf, not NaN.
    """
    denom = denominator.replace(0, float("nan"))  # prevent silent inf from zero division
    ratio = numerator / denom
    ratio = ratio.replace([float("inf"), float("-inf")], float("nan"))
    z = zscore_rows(ratio)
    return z.fillna(0.0)  # 0.0 = cross-sectional mean after z-scoring


def backup_features(features_dir: str, data_dir: str) -> str:
    """Copy existing features/ to features_backup_YYYYMMDD/ adjacent to it.

    If a backup already exists for today, a _HHMMSS suffix is added to avoid collision.
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    backup_name = f"features_backup_{timestamp}"
    backup_path = os.path.join(data_dir, backup_name)
    # Avoid collision with an existing same-day backup
    if os.path.exists(backup_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"features_backup_{timestamp}"
        backup_path = os.path.join(data_dir, backup_name)
    shutil.copytree(features_dir, backup_path)
    print(f"[backup] Existing features/ backed up to {backup_path}")
    return backup_path


# ── Main function ──────────────────────────────────────────────────────────────

def main(config_path: str, data_dir: str = None) -> None:
    """Build 360 Alpha360-style feature CSVs from OHLCV Parquet files.

    Args:
        config_path: Path to .conf file (provides alpha_360_dir under [file] section).
        data_dir: Optional override for base data directory. When None, derived from
                  alpha_360_dir's parent. When provided (e.g., in tests), ohlcv_dir and
                  tickers_file are derived from data_dir.
    """
    # ── 1. Parse config ────────────────────────────────────────────────────────
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    alpha_360_dir = cfg["file"]["alpha_360_dir"]
    # Resolve relative paths against CWD (script may be run from project root)
    alpha_360_dir = str(Path(alpha_360_dir).resolve())

    # ── 2. Derive directory paths ──────────────────────────────────────────────
    if data_dir is None:
        data_dir = str(Path(alpha_360_dir).parent)
    else:
        data_dir = str(Path(data_dir).resolve())
    ohlcv_dir = os.path.join(data_dir, "ohlcv")
    features_dir = alpha_360_dir  # output dir (from config, already resolved above)

    print(f"[config] alpha_360_dir = {features_dir}")
    print(f"[config] ohlcv_dir     = {ohlcv_dir}")
    print(f"[config] data_dir      = {data_dir}")

    # ── 3. Load tickers (defines canonical column order for all output CSVs) ───
    tickers = load_tickers(data_dir)
    print(f"[tickers] {len(tickers)} tickers loaded from tickers.txt")

    # ── 4. Load all OHLCV wide DataFrames (one per field; all share same date index) ─
    print("[load] Loading OHLCV Parquet files...")
    df_close = load_wide(ohlcv_dir, tickers, "Close")
    df_open  = load_wide(ohlcv_dir, tickers, "Open")
    df_high  = load_wide(ohlcv_dir, tickers, "High")
    df_low   = load_wide(ohlcv_dir, tickers, "Low")
    df_vol   = load_wide(ohlcv_dir, tickers, "Volume")
    df_vwap  = (df_high + df_low + df_close) / 3.0  # VWAP approximation

    # Field map: (prefix, numerator_df)
    # Denominator is always df_close.shift(d) EXCEPT for VOL features.
    field_map = [
        ("CLOSE", df_close),   # Close[t] / Close[t-d]
        ("OPEN",  df_open),    # Open[t]  / Close[t-d]
        ("HIGH",  df_high),    # High[t]  / Close[t-d]
        ("LOW",   df_low),     # Low[t]   / Close[t-d]
        ("VWAP",  df_vwap),    # VWAP[t]  / Close[t-d]
        ("VOL",   df_vol),     # Volume[t] / Volume[t-d]  <- different denominator
    ]

    # ── 5. Backup existing features BEFORE overwriting ─────────────────────────
    if os.path.exists(features_dir) and os.listdir(features_dir):
        backup_features(features_dir, data_dir)
        # Clear features_dir so only the new 360 CSVs remain after the run.
        # Existing files (old TA CSVs or previous Alpha360 CSVs) are already
        # preserved in the backup directory.
        for old_file in os.listdir(features_dir):
            old_path = os.path.join(features_dir, old_file)
            if os.path.isfile(old_path):
                os.remove(old_path)
        print(f"[clear] Cleared existing files from {features_dir}")
    else:
        os.makedirs(features_dir, exist_ok=True)

    # ── 6. Generate all 360 features ──────────────────────────────────────────
    total = len(field_map) * len(LAGS)
    print(f"[build] Generating {total} feature CSVs (6 fields x 60 lags)...")
    written = 0

    with tqdm(total=total, desc="Alpha360 features") as pbar:
        for prefix, df_num in field_map:
            # Choose denominator: VOL uses Volume[t-d], all price fields use Close[t-d]
            if prefix == "VOL":
                df_denom_base = df_vol
            else:
                df_denom_base = df_close

            for d in LAGS:
                feature_name = f"{prefix}_d{d}"
                denominator_shifted = df_denom_base.shift(d)

                # Compute z-scored ratio with safe zero division
                feature_df = safe_ratio(df_num, denominator_shifted)

                # ── 7. Slice: drop first LAG_BUFFER rows ──────────────────────
                # After shift(d=60), rows 0..59 are NaN. Slicing at [60:] gives
                # first valid row = ohlcv index[60] = 2018-03-29 (real data).
                feature_df = feature_df.iloc[LAG_BUFFER:]

                # Write CSV: index is date, columns are tickers in tickers.txt order
                out_path = os.path.join(features_dir, f"{feature_name}.csv")
                feature_df.to_csv(out_path)

                written += 1
                pbar.update(1)

    print(f"[done] Written {written} feature CSVs to {features_dir}")

    # ── 8. Final validation summary ────────────────────────────────────────────
    csv_files = [f for f in os.listdir(features_dir) if f.endswith(".csv")]
    print(f"[validate] CSV count: {len(csv_files)} (expected 360)")
    if csv_files:
        sample = pd.read_csv(os.path.join(features_dir, csv_files[0]), index_col=0)
        print(f"[validate] Sample shape: {sample.shape}")
        print(f"[validate] First date:   {sample.index[0]}")
        nan_count = sample.isna().sum().sum()
        inf_count = int(np.isinf(sample.values).sum())
        print(f"[validate] NaN count: {nan_count} (expected 0)")
        print(f"[validate] Inf count: {inf_count} (expected 0)")


# ── CLI entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build 360 Alpha360-style feature CSVs from OHLCV Parquet files."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to .conf config file (e.g., config/Multitask_Stock_SP500.conf)",
    )
    args = parser.parse_args()
    main(config_path=args.config)
