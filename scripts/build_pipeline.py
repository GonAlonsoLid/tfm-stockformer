#!/usr/bin/env python
"""
S&P500 data pipeline orchestrator — static inputs for model training.

Runs steps in order:
  1. download_ohlcv.py   -- fetch daily OHLCV from yfinance for the S&P500 universe
  2. normalize_split.py  -- record train/val/test date boundaries (split_indices.json)
  3. serialize_arrays.py -- save flow.npz and trend_indicator.npz from label.csv
  4. graph_embedding.py  -- build Struc2Vec embedding [N, 128] from correlation graph
  5. build_alpha360.py   -- build 360 Alpha360-style feature CSVs (requires --config)

The data directory is always derived from --start/--end dates:
  ./data/Stock_SP500_{start}_{end}/

This means different date ranges automatically use different directories — no --force
needed when changing dates. Steps are skipped only when data for the exact same
date range already exists.

Usage (recommended — runs all five steps):
  python scripts/build_pipeline.py --config config/Multitask_Stock_SP500.conf \\
      --start 2018-01-01 --end 2024-01-01

Usage (different date range — runs fresh, no conflict with existing data):
  python scripts/build_pipeline.py --config config/Multitask_Stock_SP500.conf \\
      --start 2018-01-01 --end 2026-03-16

Options:
  --config     Path to .conf file; enables step 5 (Alpha360)
  --start      Start date for OHLCV download (default: 2018-01-01)
  --end        End date for OHLCV download (default: 2024-01-01)
  --data_dir   Override the auto-derived directory (advanced use only)
  --force      Re-run all steps even if their output already exists

Each step checks for its sentinel output file and skips itself if the file is already
present, making re-runs idempotent. Use --force to override this behaviour.

Prerequisites:
  pip install -r requirements.txt
  pip install git+https://github.com/shenweichen/GraphEmbedding.git  # for step 4
"""
import argparse
import configparser
import os
import subprocess
import sys
from pathlib import Path

# Resolve path to the sp500_pipeline package relative to this script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(_SCRIPT_DIR, "sp500_pipeline")

# Ensure scripts/ is on sys.path so build_alpha360 can be imported directly
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Steps 1-4 run via subprocess; step 5 (Alpha360) runs via direct import.
# Each step is skipped when its sentinel already exists (unless --force).
SUBPROCESS_STEPS = [
    ("download_ohlcv.py",   "tickers.txt"),
    ("normalize_split.py",  "split_indices.json"),
    ("serialize_arrays.py", "flow.npz"),
    ("graph_embedding.py",  "128_corr_struc2vec_adjgat.npy"),
]
TOTAL_STEPS = 5  # 4 subprocess + 1 Alpha360


def _sentinel_exists(data_dir: str, sentinel: str) -> bool:
    return os.path.exists(os.path.join(data_dir, sentinel))


def _data_dir_from_config(config_path: str) -> tuple:
    """Return (data_dir, alpha_360_dir) derived from the .conf file.

    Reads cfg["file"]["traffic"] and cfg["file"]["alpha_360_dir"].
    data_dir = parent of the traffic file path.
    alpha_360_dir = alpha_360_dir value resolved to absolute path.

    Parameters
    ----------
    config_path : str
        Path to the .conf file (e.g. config/Multitask_Stock_SP500.conf).

    Returns
    -------
    tuple[str, str]
        (data_dir, alpha_360_dir) as absolute path strings.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    traffic = cfg["file"]["traffic"]
    data_dir = str(Path(traffic).resolve().parent)
    alpha_360_dir = str(Path(cfg["file"]["alpha_360_dir"]).resolve())
    return data_dir, alpha_360_dir


def _alpha360_done(features_dir: str) -> bool:
    """Return True if features/ contains exactly 360 CSV files.

    Used as the sentinel check for step 5 (Alpha360). Unlike steps 1-4
    whose sentinels are single files, step 5 is complete only when all
    360 feature CSVs are present.

    Parameters
    ----------
    features_dir : str
        Path to the features/ subdirectory inside data_dir.

    Returns
    -------
    bool
        True if exactly 360 .csv files exist in features_dir.
    """
    if not os.path.isdir(features_dir):
        return False
    return sum(1 for f in os.listdir(features_dir) if f.endswith(".csv")) == 360


def run_alpha360_step(config_path: str, features_dir: str, force: bool = False) -> None:
    """Execute step 5: build Alpha360 feature CSVs via direct import.

    Calls build_alpha360.main(config_path=...) directly (not subprocess)
    for cleaner error propagation and to avoid spawning a new Python process.
    Skips if _alpha360_done(features_dir) is True, unless force=True.

    Parameters
    ----------
    config_path : str
        Path to the .conf file passed through to build_alpha360.main().
    features_dir : str
        Path to features/ directory for sentinel check.
    force : bool
        When True, run even if sentinel is satisfied.
    """
    if not force and _alpha360_done(features_dir):
        print("[SKIP] build_alpha360  (sentinel: 360 CSVs already in features/)")
        return
    print("\n[RUN]  build_alpha360")
    from build_alpha360 import main as build_alpha360_main  # noqa: PLC0415
    build_alpha360_main(config_path=config_path, data_dir=os.path.dirname(features_dir))
    print("[DONE] build_alpha360")


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


def _date_range_from_config(config_path: str) -> tuple:
    """Read date_range from config [DEFAULT] and return (start, end).

    date_range format: 'YYYY-MM-DD_YYYY-MM-DD' (e.g. '2018-01-01_2024-01-01').
    Returns (None, None) if config has no date_range key.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    date_range = cfg.defaults().get("date_range")
    if not date_range:
        return None, None
    parts = date_range.split("_")
    return parts[0], parts[1]


def main() -> None:
    # Pre-parse --config so we can read date_range as defaults for --start/--end.
    # This makes the config the single source of truth: change date_range once,
    # and both the pipeline (data writing) and training (data reading) align.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    cfg_start, cfg_end = "2018-01-01", "2024-01-01"
    if pre_args.config:
        s, e = _date_range_from_config(pre_args.config)
        if s and e:
            cfg_start, cfg_end = s, e

    parser = argparse.ArgumentParser(
        description="S&P500 data pipeline orchestrator — runs all five steps end-to-end",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Path to .conf file (e.g. config/Multitask_Stock_SP500.conf). "
            "Required to enable step 5 (Alpha360). date_range in the config "
            "sets the default start/end dates."
        ),
    )
    parser.add_argument(
        "--start",
        default=cfg_start,
        help="Start date for OHLCV download (default: read from config date_range)",
    )
    parser.add_argument(
        "--end",
        default=cfg_end,
        help="End date for OHLCV download (default: read from config date_range)",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help=(
            "Override the data directory. If omitted, derived automatically from "
            "--start/--end as ./data/Stock_SP500_{start}_{end}/ (recommended)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all steps even if their sentinel output already exists",
    )
    args = parser.parse_args()

    # data_dir is derived from dates so different date ranges use different directories.
    # --data_dir overrides this only when explicitly provided.
    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = os.path.join(".", "data", f"Stock_SP500_{args.start}_{args.end}")

    config_path = args.config
    alpha_360_dir = os.path.join(data_dir, "features")

    os.makedirs(data_dir, exist_ok=True)

    # Extra arguments forwarded only to download_ohlcv.py
    download_extra = ["--start", args.start, "--end", args.end]

    for i, (script_name, sentinel) in enumerate(SUBPROCESS_STEPS, start=1):
        extra = download_extra if script_name == "download_ohlcv.py" else []
        print(f"\n--- Step {i}/{TOTAL_STEPS}: {script_name} ---")
        run_step(script_name, sentinel, data_dir, extra, force=args.force)

    # Step 5: Alpha360 feature builder (requires --config; skipped silently if no config)
    if config_path:
        print(f"\n--- Step {TOTAL_STEPS}/{TOTAL_STEPS}: build_alpha360 ---")
        run_alpha360_step(config_path=config_path, features_dir=alpha_360_dir, force=args.force)
    else:
        print("\n[INFO] Skipping step 5 (build_alpha360): pass --config to enable.")
        print("       Next: python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf")

    print("\n[COMPLETE] All pipeline steps finished.")
    print(f"Output directory: {os.path.abspath(data_dir)}")
    if not config_path:
        print("Next: python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf")


if __name__ == "__main__":
    main()
