#!/usr/bin/env python3
"""Ablation study runner for Stockformer S&P 500 experiments.

Runs experiments E1-E6 sequentially and collects test metrics into a CSV.

Usage:
    # Run all experiments end-to-end
    python scripts/run_ablation.py

    # Run a single experiment
    python scripts/run_ablation.py --only E1

    # Collect results from existing logs without re-training
    python scripts/run_ablation.py --collect-only

    # Run experiments and collect into a custom output path
    python scripts/run_ablation.py --output results/my_ablation.csv
"""

import argparse
import configparser
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS: dict[str, dict] = {
    "E1": {
        "name": "MSE + Alpha360 + Static Graph",
        "config": "config/experiment_E1_mse_baseline.conf",
        "type": "stockformer",
    },
    "E2": {
        "name": "Ranking + Alpha360 + Static Graph",
        "config": "config/experiment_E2_ranking_baseline.conf",
        "type": "stockformer",
    },
    "E3": {
        "name": "Ranking + Alpha360 + Dynamic Graph",
        "config": "config/experiment_E3_dynamic_graph.conf",
        "type": "stockformer",
    },
    "E4": {
        "name": "Ranking + All Features + Static Graph",
        "config": "config/experiment_E4_rich_features.conf",
        "type": "stockformer",
    },
    "E5": {
        "name": "Ranking + All Features + Dynamic Graph",
        "config": "config/experiment_E5_all_improvements.conf",
        "type": "stockformer",
    },
    "E6": {
        "name": "LightGBM + Alpha158 Baseline",
        "config": "config/Multitask_Stock_SP500.conf",
        "type": "lightgbm",
    },
}

# Regex for Stockformer test log lines (with or without IC field)
_LOG_PATTERN = re.compile(
    r"average,\s*acc:\s*(?P<acc>[\d.]+),\s*mae:\s*(?P<mae>[\d.]+),"
    r"\s*rmse:\s*(?P<rmse>[\d.]+),\s*mape:\s*(?P<mape>[\d.eE+-]+)"
    r"(?:,\s*IC:\s*(?P<ic>[-\d.eE+]+))?"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    """Resolve a path relative to PROJECT_ROOT."""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _get_log_path(config_path: str) -> Optional[Path]:
    """Parse the [file] log value from an INI config and return its Path."""
    cfg = configparser.ConfigParser()
    cfg.read(_resolve(config_path))
    try:
        log_value = cfg.get("file", "log")
    except (configparser.NoSectionError, configparser.NoOptionError):
        return None
    return _resolve(log_value)


def _parse_stockformer_log(log_path: Path) -> Optional[dict]:
    """Return the *last* test-period metrics from a Stockformer log file.

    The training loop writes one 'average, acc: ...' line per epoch.  We take
    the last line because it reflects the final (best-checkpoint) evaluation.
    """
    if not log_path.exists():
        return None

    last_match = None
    with log_path.open("r", errors="replace") as fh:
        for line in fh:
            m = _LOG_PATTERN.search(line)
            if m:
                last_match = m

    if last_match is None:
        return None

    ic_raw = last_match.group("ic")
    return {
        "ic": float(ic_raw) if ic_raw is not None else float("nan"),
        "accuracy": float(last_match.group("acc")),
        "mae": float(last_match.group("mae")),
        "rmse": float(last_match.group("rmse")),
        "mape": float(last_match.group("mape")),
    }


def _parse_lightgbm_results() -> Optional[dict]:
    """Read LightGBM metrics from output/lightgbm_baseline/evaluation_summary.csv."""
    summary_path = PROJECT_ROOT / "output" / "lightgbm_baseline" / "evaluation_summary.csv"
    if not summary_path.exists():
        return None

    with summary_path.open() as fh:
        reader = csv.DictReader(fh)
        rows = {row["metric"]: float(row["value"]) for row in reader}

    if "mean_ic" not in rows:
        return None

    return {
        "ic": rows.get("mean_ic", float("nan")),
        "accuracy": float("nan"),
        "mae": float("nan"),
        "rmse": float("nan"),
        "mape": float("nan"),
    }


# ---------------------------------------------------------------------------
# Training runners
# ---------------------------------------------------------------------------


def _run_stockformer(exp_id: str, config: str) -> int:
    """Launch Stockformer training and return the subprocess return code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "MultiTask_Stockformer_train.py"),
        "--config",
        str(_resolve(config)),
    ]
    print(f"\n[{exp_id}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


def _run_lightgbm(config: str) -> int:
    """Launch LightGBM baseline and return the subprocess return code."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_lightgbm_baseline.py"),
        "--config",
        str(_resolve(config)),
    ]
    print(f"\n[E6] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode


# ---------------------------------------------------------------------------
# Results collection
# ---------------------------------------------------------------------------


def collect_metrics(exp_id: str, exp_def: dict) -> Optional[dict]:
    """Return a metrics dict for one experiment, or None if not available."""
    if exp_def["type"] == "lightgbm":
        metrics = _parse_lightgbm_results()
    else:
        log_path = _get_log_path(exp_def["config"])
        if log_path is None:
            print(f"  [{exp_id}] WARNING: Could not determine log path from config.")
            return None
        metrics = _parse_stockformer_log(log_path)
        if metrics is None:
            print(f"  [{exp_id}] WARNING: No metrics found in {log_path}")
            return None

    return metrics


def save_csv(rows: list[dict], output_path: Path) -> None:
    """Write rows to the ablation results CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "name", "ic", "accuracy", "mae", "rmse", "mape"]
    with output_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {output_path}")


def print_summary(rows: list[dict]) -> None:
    """Print a formatted summary table to stdout."""
    if not rows:
        print("\nNo results to display.")
        return

    col_widths = {
        "experiment": 10,
        "name": 42,
        "ic": 10,
        "accuracy": 10,
        "mae": 10,
        "rmse": 10,
        "mape": 10,
    }
    header = (
        f"{'Exp':<{col_widths['experiment']}}"
        f"{'Name':<{col_widths['name']}}"
        f"{'IC':>{col_widths['ic']}}"
        f"{'Accuracy':>{col_widths['accuracy']}}"
        f"{'MAE':>{col_widths['mae']}}"
        f"{'RMSE':>{col_widths['rmse']}}"
        f"{'MAPE':>{col_widths['mape']}}"
    )
    separator = "-" * len(header)

    print("\n" + separator)
    print("Ablation Study Results")
    print(separator)
    print(header)
    print(separator)

    for row in rows:
        def _fmt(v: object) -> str:
            try:
                return f"{float(v):.4f}"  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return str(v)

        print(
            f"{row['experiment']:<{col_widths['experiment']}}"
            f"{row['name']:<{col_widths['name']}}"
            f"{_fmt(row['ic']):>{col_widths['ic']}}"
            f"{_fmt(row['accuracy']):>{col_widths['accuracy']}}"
            f"{_fmt(row['mae']):>{col_widths['mae']}}"
            f"{_fmt(row['rmse']):>{col_widths['rmse']}}"
            f"{_fmt(row['mape']):>{col_widths['mape']}}"
        )

    print(separator + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ablation experiments E1-E6 and collect results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--only",
        metavar="EX",
        help="Run (or collect) a single experiment, e.g. --only E1",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Skip training; only parse existing logs and write the CSV.",
    )
    parser.add_argument(
        "--output",
        default="results/ablation_results.csv",
        help="Path for the output CSV (default: results/ablation_results.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = _resolve(args.output)

    # Determine which experiments to process
    if args.only:
        exp_id = args.only.upper()
        if exp_id not in EXPERIMENTS:
            print(
                f"ERROR: Unknown experiment '{args.only}'. "
                f"Valid options: {', '.join(EXPERIMENTS)}"
            )
            sys.exit(1)
        selected = {exp_id: EXPERIMENTS[exp_id]}
    else:
        selected = EXPERIMENTS

    failures: list[str] = []

    # -----------------------------------------------------------------------
    # Training phase
    # -----------------------------------------------------------------------
    if not args.collect_only:
        for exp_id, exp_def in selected.items():
            print(f"\n{'=' * 60}")
            print(f"  Starting experiment {exp_id}: {exp_def['name']}")
            print(f"{'=' * 60}")

            if exp_def["type"] == "lightgbm":
                rc = _run_lightgbm(exp_def["config"])
            else:
                rc = _run_stockformer(exp_id, exp_def["config"])

            if rc != 0:
                print(f"\n  [{exp_id}] WARNING: training returned exit code {rc}")
                failures.append(exp_id)

    # -----------------------------------------------------------------------
    # Collection phase
    # -----------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Collecting metrics from logs")
    print(f"{'=' * 60}")

    rows: list[dict] = []
    for exp_id, exp_def in selected.items():
        metrics = collect_metrics(exp_id, exp_def)
        if metrics is None:
            print(f"  [{exp_id}] No metrics collected — skipping.")
            continue

        row = {
            "experiment": exp_id,
            "name": exp_def["name"],
            **metrics,
        }
        rows.append(row)
        print(f"  [{exp_id}] IC={row['ic']:.4f}  acc={row['accuracy']:.4f}  "
              f"mae={row['mae']:.4f}  rmse={row['rmse']:.4f}  mape={row['mape']:.4f}")

    # -----------------------------------------------------------------------
    # Save and display
    # -----------------------------------------------------------------------
    if rows:
        save_csv(rows, output_path)
        print_summary(rows)
    else:
        print("\nNo results collected. Nothing written to CSV.")

    if failures:
        print(
            f"WARNING: The following experiments had non-zero exit codes: "
            f"{', '.join(failures)}"
        )

    if not rows and not args.collect_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
