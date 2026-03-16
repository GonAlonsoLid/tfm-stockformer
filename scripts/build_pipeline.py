#!/usr/bin/env python
"""
End-to-end S&P500 data pipeline orchestrator.

Runs all five steps in order:
  1. download_ohlcv.py      -- fetch daily OHLCV from yfinance for the S&P500 universe
  2. feature_engineering.py -- compute TA features (RSI, MACD, BB, ROC, VOL) + label.csv
  3. normalize_split.py     -- cross-sectional normalization and date-based train/val/test split
  4. serialize_arrays.py    -- save flow.npz and trend_indicator.npz for model training
  5. graph_embedding.py     -- build Struc2Vec embedding [N, 128] from correlation graph

Usage:
  python scripts/build_pipeline.py --data_dir ./data/Stock_SP500_2018-01-01_2024-01-01

Options:
  --data_dir   Output directory (default: ./data/Stock_SP500_2018-01-01_2024-01-01)
  --start      Start date for OHLCV download (default: 2018-01-01)
  --end        End date for OHLCV download (default: 2024-01-01)
  --force      Re-run all steps even if their output already exists

Each step checks for its sentinel output file and skips itself if the file is already
present, making re-runs idempotent. Use --force to override this behaviour.

Prerequisites:
  pip install -r requirements.txt
  pip install git+https://github.com/shenweichen/GraphEmbedding.git  # for step 5
"""
import argparse
import os
import subprocess
import sys

# Resolve path to the sp500_pipeline package relative to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "data_processing_script", "sp500_pipeline")
)

# (script_name, sentinel_file_relative_to_data_dir)
# Each step is skipped when its sentinel already exists (unless --force).
STEPS = [
    ("download_ohlcv.py",      "tickers.txt"),
    ("feature_engineering.py", "label.csv"),
    ("normalize_split.py",     "split_indices.json"),
    ("serialize_arrays.py",    "flow.npz"),
    ("graph_embedding.py",     "128_corr_struc2vec_adjgat.npy"),
]


def _sentinel_exists(data_dir: str, sentinel: str) -> bool:
    return os.path.exists(os.path.join(data_dir, sentinel))


def run_step(
    script_name: str,
    sentinel: str,
    data_dir: str,
    extra_args: list,
    force: bool = False,
) -> None:
    """
    Execute a single pipeline step via subprocess.

    Parameters
    ----------
    script_name : str
        Filename of the step script inside PIPELINE_DIR.
    sentinel : str
        Relative path (from data_dir) of the file that indicates the step is done.
    data_dir : str
        Root data directory passed to each step as --data_dir.
    extra_args : list
        Additional CLI arguments forwarded to the step (e.g. ['--start', '2018-01-01']).
    force : bool
        When True, run the step even if the sentinel already exists.
    """
    if not force and _sentinel_exists(data_dir, sentinel):
        print(f"[SKIP] {script_name}  (sentinel: {sentinel} already exists)")
        return

    script_path = os.path.join(PIPELINE_DIR, script_name)
    cmd = [sys.executable, script_path, "--data_dir", data_dir] + extra_args
    print(f"\n[RUN]  {script_name}")
    result = subprocess.run(cmd, check=True)
    print(f"[DONE] {script_name}  (exit code: {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S&P500 data pipeline orchestrator — runs all five steps end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        default="./data/Stock_SP500_2018-01-01_2024-01-01",
        help="Root directory for all pipeline inputs and outputs",
    )
    parser.add_argument(
        "--start",
        default="2018-01-01",
        help="Start date for OHLCV download (passed to download_ohlcv.py)",
    )
    parser.add_argument(
        "--end",
        default="2024-01-01",
        help="End date for OHLCV download (passed to download_ohlcv.py)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if their sentinel output already exists",
    )
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    # Extra arguments forwarded only to download_ohlcv.py
    download_extra = ["--start", args.start, "--end", args.end]

    for i, (script_name, sentinel) in enumerate(STEPS, start=1):
        extra = download_extra if script_name == "download_ohlcv.py" else []
        print(f"\n--- Step {i}/{len(STEPS)}: {script_name} ---")
        run_step(script_name, sentinel, args.data_dir, extra, force=args.force)

    print("\n[COMPLETE] All pipeline steps finished.")
    print(f"Output directory: {os.path.abspath(args.data_dir)}")


if __name__ == "__main__":
    main()
