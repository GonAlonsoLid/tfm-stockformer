#!/usr/bin/env python3
"""Evaluate Stockformer inference outputs with IC, ICIR, MAE, RMSE, accuracy, F1.

Usage:
    python scripts/compute_ic.py --output_dir output/Multitask_output_SP500_2018-2024

Reads four CSVs produced by scripts/run_inference.py:
    regression/regression_pred_last_step.csv     -- (n_days, n_stocks) float matrix
    regression/regression_label_last_step.csv    -- (n_days, n_stocks) float matrix
    classification/classification_pred_last_step.csv   -- (n_days, n_stocks) stringified logit arrays
    classification/classification_label_last_step.csv  -- (n_days, n_stocks) binary int labels

Writes two output files into output_dir:
    evaluation_summary.csv  -- one row: IC_mean, ICIR, IC_pearson, MAE, RMSE, Accuracy, F1_macro
    ic_by_day.csv           -- one row per test day: day, IC (Spearman), IC_pearson
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score


def load_regression(output_dir: str):
    """Load regression prediction and label CSVs from output_dir.

    Returns:
        reg_pred: np.ndarray float64 of shape (n_days, n_stocks)
        reg_label: np.ndarray float64 of shape (n_days, n_stocks)
    """
    pred_path = os.path.join(output_dir, "regression", "regression_pred_last_step.csv")
    label_path = os.path.join(output_dir, "regression", "regression_label_last_step.csv")

    for path in (pred_path, label_path):
        if not os.path.isfile(path):
            print(f"ERROR: Required file not found: {path}", file=sys.stderr)
            sys.exit(1)

    reg_pred = pd.read_csv(pred_path, header=None).values.astype(np.float64)
    reg_label = pd.read_csv(label_path, header=None).values.astype(np.float64)
    return reg_pred, reg_label


def parse_cls_pred_csv(path: str) -> np.ndarray:
    """Load classification_pred_last_step.csv and convert to integer class predictions.

    Each cell contains a stringified logit array like '[ 0.16767871 -0.08977825]'.
    Parses floats with regex and applies argmax to get the predicted class index.

    Returns:
        np.ndarray of shape (n_days, n_stocks) with dtype int
    """
    df = pd.read_csv(path, header=None)
    preds = np.zeros(df.shape, dtype=int)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            nums = [
                float(x)
                for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", str(df.iloc[i, j]))
            ]
            preds[i, j] = int(np.argmax(nums)) if len(nums) >= 2 else 0
    return preds


def load_classification(output_dir: str):
    """Load classification prediction and label CSVs from output_dir.

    Returns:
        cls_preds_2d: np.ndarray int of shape (n_days, n_stocks)
        cls_label_flat: np.ndarray int of shape (n_days * n_stocks,)
    """
    pred_path = os.path.join(
        output_dir, "classification", "classification_pred_last_step.csv"
    )
    label_path = os.path.join(
        output_dir, "classification", "classification_label_last_step.csv"
    )

    for path in (pred_path, label_path):
        if not os.path.isfile(path):
            print(f"ERROR: Required file not found: {path}", file=sys.stderr)
            sys.exit(1)

    cls_preds_2d = parse_cls_pred_csv(pred_path)
    cls_label_flat = (
        pd.read_csv(label_path, header=None).values.flatten().astype(int)
    )
    return cls_preds_2d, cls_label_flat


def compute_ic_metrics(reg_pred: np.ndarray, reg_label: np.ndarray):
    """Compute Spearman IC per day, plus ICIR.

    IC (Information Coefficient) is the cross-sectional rank correlation between
    predicted returns and realized returns, computed independently for each trading day.

    Args:
        reg_pred: (n_days, n_stocks) float array of predicted returns
        reg_label: (n_days, n_stocks) float array of realized returns

    Returns:
        3-tuple:
            ic_mean: float              -- mean of valid Spearman IC days
            icir: float                 -- ic_mean / std(valid_ic, ddof=1)
            ic_per_day: np.ndarray (n_days,) -- per-day Spearman IC (NaN for degenerate days)
    """
    n_days = reg_pred.shape[0]

    ic_per_day = np.empty(n_days)
    for d in range(n_days):
        result = spearmanr(reg_pred[d], reg_label[d])
        ic_per_day[d] = result.correlation if hasattr(result, "correlation") else result.statistic

    # Filter NaN days before computing ICIR
    nan_count = int(np.isnan(ic_per_day).sum())
    if nan_count > 0:
        print(
            f"WARNING: {nan_count} day(s) had constant predictions (IC=NaN), "
            "excluded from ICIR.",
            file=sys.stderr,
        )

    valid_ic = ic_per_day[~np.isnan(ic_per_day)]
    ic_mean = float(np.mean(valid_ic))
    ic_std = float(np.std(valid_ic, ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else float("nan")

    return ic_mean, icir, ic_per_day


def _compute_pearson_ic(reg_pred: np.ndarray, reg_label: np.ndarray):
    """Compute Pearson IC per day (bonus column for thesis completeness).

    Args:
        reg_pred: (n_days, n_stocks) float array
        reg_label: (n_days, n_stocks) float array

    Returns:
        2-tuple:
            ic_mean_pearson: float
            ic_per_day_pearson: np.ndarray (n_days,)
    """
    n_days = reg_pred.shape[0]
    ic_per_day_pearson = np.empty(n_days)
    for d in range(n_days):
        corr_matrix = np.corrcoef(reg_pred[d], reg_label[d])
        ic_per_day_pearson[d] = corr_matrix[0, 1]

    valid_pearson = ic_per_day_pearson[~np.isnan(ic_per_day_pearson)]
    ic_mean_pearson = float(np.mean(valid_pearson)) if len(valid_pearson) > 0 else float("nan")
    return ic_mean_pearson, ic_per_day_pearson


def compute_regression_metrics(reg_pred: np.ndarray, reg_label: np.ndarray):
    """Compute MAE and RMSE between predicted and realized returns.

    Args:
        reg_pred: float array of predicted values
        reg_label: float array of realized values (same shape)

    Returns:
        (mae, rmse): tuple of floats
    """
    mae = float(np.mean(np.abs(reg_pred - reg_label)))
    rmse = float(np.sqrt(np.mean((reg_pred - reg_label) ** 2)))
    return mae, rmse


def compute_classification_metrics(cls_preds_2d: np.ndarray, cls_label_flat: np.ndarray):
    """Compute accuracy and macro-F1 for the classification head.

    Args:
        cls_preds_2d: (n_days, n_stocks) int array of predicted classes
        cls_label_flat: (n_days * n_stocks,) int array of ground-truth classes

    Returns:
        (acc, f1): tuple of floats
    """
    preds_flat = cls_preds_2d.flatten()
    acc = float(np.mean(preds_flat == cls_label_flat))
    f1 = float(f1_score(cls_label_flat, preds_flat, average="macro"))
    return acc, f1


def main(output_dir: str = None):
    """Compute and print all evaluation metrics; write summary CSVs to output_dir.

    Can be called programmatically with output_dir keyword arg (for testing),
    or invoked via CLI with --output_dir argument.

    Args:
        output_dir: path to directory containing regression/ and classification/
                    subdirectories. If None, parsed from --output_dir CLI argument.
    """
    if output_dir is None:
        parser = argparse.ArgumentParser(
            description="Compute evaluation metrics on Stockformer inference outputs."
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            required=True,
            help="Directory produced by run_inference.py (contains regression/ and classification/)",
        )
        args = parser.parse_args()
        output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        print(f"ERROR: output_dir does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all four CSVs
    reg_pred, reg_label = load_regression(output_dir)
    cls_preds, cls_label = load_classification(output_dir)

    # Compute metrics
    ic_mean, icir, ic_per_day = compute_ic_metrics(reg_pred, reg_label)
    ic_mean_pearson, ic_per_day_pearson = _compute_pearson_ic(reg_pred, reg_label)
    mae, rmse = compute_regression_metrics(reg_pred, reg_label)
    acc, f1 = compute_classification_metrics(cls_preds, cls_label)

    # Console table
    print("\n=== Evaluation Summary ===")
    print(f"  IC mean (Spearman) : {ic_mean:+.6f}")
    print(f"  IC mean (Pearson)  : {ic_mean_pearson:+.6f}")
    print(f"  ICIR               : {icir:+.6f}")
    print(f"  MAE                : {mae:.6f}")
    print(f"  RMSE               : {rmse:.6f}")
    print(f"  Accuracy           : {acc:.4f}")
    print(f"  F1 (macro)         : {f1:.4f}")

    # Write evaluation_summary.csv
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    pd.DataFrame(
        [
            {
                "IC_mean": ic_mean,
                "ICIR": icir,
                "IC_pearson": ic_mean_pearson,
                "MAE": mae,
                "RMSE": rmse,
                "Accuracy": acc,
                "F1_macro": f1,
            }
        ]
    ).to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Write ic_by_day.csv
    ic_day_path = os.path.join(output_dir, "ic_by_day.csv")
    pd.DataFrame(
        {
            "day": range(len(ic_per_day)),
            "IC": ic_per_day,
            "IC_pearson": ic_per_day_pearson,
        }
    ).to_csv(ic_day_path, index=False)
    print(f"Daily IC saved to: {ic_day_path}")


if __name__ == "__main__":
    main()
