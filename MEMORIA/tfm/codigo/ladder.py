#!/usr/bin/env python3
"""Run the CPU model-complexity ladder and evaluate every model identically.

Ladder (low -> high complexity), all on the SAME panel/features/split/harness:
    L0  sanity   : zero, cross-sectional momentum, short-term reversal
    L1  linear   : Ridge, Lasso, ElasticNet
    L2  trees    : LightGBM (+ XGBoost if installed)

Outputs:
    results/ladder_results.csv          one row per model, full harness metrics
    results/ladder_daily_ic/<model>.csv per-day rank IC (for stat tests + figures)

GPU models (MLP, StockMixer, Stockformer) are evaluated separately via the
cluster bundle and merged into the same table in the analysis step.

Usage:
    python scripts/run_cpu_ladder.py --data_dir data/Stock_SP500_2018-01-01_2026-03-16
    python scripts/run_cpu_ladder.py --only ridge,lightgbm
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib import baseline_signals as bs  # noqa: E402
from lib import data_panel as dp  # noqa: E402
from lib import eval_harness as eh  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DAILY_IC_DIR = os.path.join(RESULTS_DIR, "ladder_daily_ic")
LS_RETURNS_DIR = os.path.join(RESULTS_DIR, "ladder_ls_returns")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"

QUANTILE = 0.1   # decile long-short
FEE = 0.001      # 10 bps per side
N_BOOT = 2000
SEED = 0


# ── Prediction builders ─────────────────────────────────────────────────────────

def predict_sanity(name: str, panel: dp.Panel) -> np.ndarray:
    """Return a [T, N] score matrix for a no-training sanity signal."""
    if name == "zero":
        return bs.zero_signal(panel.X, panel.feature_names)
    if name == "momentum":
        return bs.momentum_signal(panel.X, panel.feature_names, lags=20)
    if name == "reversal":
        return bs.reversal_signal(panel.X, panel.feature_names, lags=5)
    raise ValueError(name)


def predict_sklearn(model, panel: dp.Panel, split: dict) -> tuple[np.ndarray, int]:
    """Fit a tabular sklearn-style regressor and return (test_predictions, n_params).

    Features standardized with train-only statistics (no leakage).
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(split["X_train"])
    Xtr = scaler.transform(split["X_train"])
    Xte = scaler.transform(split["X_test"])
    model.fit(Xtr, split["y_train"])
    yhat = model.predict(Xte)
    n_params = int(np.sum(getattr(model, "coef_", np.zeros(1)) != 0)) + 1
    return yhat, n_params


def predict_lightgbm(panel: dp.Panel, split: dict) -> tuple[np.ndarray, int]:
    import lightgbm as lgb

    num_leaves = 31
    model = lgb.LGBMRegressor(
        objective="huber", n_estimators=300, learning_rate=0.01,
        num_leaves=num_leaves, subsample=0.7, colsample_bytree=0.5,
        min_child_samples=100, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(split["X_train"], split["y_train"],
              eval_set=[(split["X_val"], split["y_val"])],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    yhat = model.predict(split["X_test"])
    n_params = int(model.booster_.num_trees() * num_leaves)
    return yhat, n_params


def predict_xgboost(panel: dp.Panel, split: dict) -> tuple[np.ndarray, int]:
    import xgboost as xgb

    model = xgb.XGBRegressor(
        objective="reg:pseudohubererror", n_estimators=300, learning_rate=0.01,
        max_depth=5, subsample=0.7, colsample_bytree=0.5, reg_alpha=0.1,
        reg_lambda=1.0, random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(split["X_train"], split["y_train"],
              eval_set=[(split["X_val"], split["y_val"])], verbose=False)
    yhat = model.predict(split["X_test"])
    n_params = int(300 * (2 ** 5))
    return yhat, n_params


# ── Registry ─────────────────────────────────────────────────────────────────

def build_registry():
    from sklearn.linear_model import ElasticNet, Lasso, Ridge

    return {
        "zero":       dict(level="L0", family="sanity", kind="sanity"),
        "momentum":   dict(level="L0", family="sanity", kind="sanity"),
        "reversal":   dict(level="L0", family="sanity", kind="sanity"),
        "ridge":      dict(level="L1", family="linear", kind="sklearn",
                           model=lambda: Ridge(alpha=10.0)),
        "lasso":      dict(level="L1", family="linear", kind="sklearn",
                           model=lambda: Lasso(alpha=1e-4, max_iter=5000)),
        "elasticnet": dict(level="L1", family="linear", kind="sklearn",
                           model=lambda: ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000)),
        "lightgbm":   dict(level="L2", family="trees", kind="lightgbm"),
        "xgboost":    dict(level="L2", family="trees", kind="xgboost"),
    }


# ── Evaluation ───────────────────────────────────────────────────────────────

def _test_frames(panel: dp.Panel, score_matrix: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Slice the test window from a full [T, N] score matrix into canonical frames."""
    test_dates = panel.dates[panel.val_end:]
    pred = dp.to_canonical(score_matrix[panel.val_end:], test_dates, panel.tickers)
    label = dp.to_canonical(panel.y[panel.val_end:], test_dates, panel.tickers)
    return pred, label


def _scatter_test(panel: dp.Panel, split: dict, yhat: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild canonical test frames from flat tabular predictions."""
    test_dates = panel.dates[panel.val_end:]
    n_days, N = len(test_dates), len(panel.tickers)
    P = np.full((n_days, N), np.nan)
    L = np.full((n_days, N), np.nan)
    for v, di, si in zip(yhat, split["date_idx_test"], split["stock_idx_test"]):
        P[di - panel.val_end, si] = v
    for v, di, si in zip(split["y_test"], split["date_idx_test"], split["stock_idx_test"]):
        L[di - panel.val_end, si] = v
    return (dp.to_canonical(P, test_dates, panel.tickers),
            dp.to_canonical(L, test_dates, panel.tickers))


def evaluate_model(name: str, spec: dict, panel: dp.Panel, split: dict) -> dict:
    t0 = time.time()
    if spec["kind"] == "sanity":
        pred, label = _test_frames(panel, predict_sanity(name, panel))
        n_params = 0
    elif spec["kind"] == "sklearn":
        yhat, n_params = predict_sklearn(spec["model"](), panel, split)
        pred, label = _scatter_test(panel, split, yhat)
    elif spec["kind"] == "lightgbm":
        yhat, n_params = predict_lightgbm(panel, split)
        pred, label = _scatter_test(panel, split, yhat)
    elif spec["kind"] == "xgboost":
        yhat, n_params = predict_xgboost(panel, split)
        pred, label = _scatter_test(panel, split, yhat)
    else:
        raise ValueError(spec["kind"])

    metrics = eh.evaluate(pred, label, quantile=QUANTILE, fee=FEE,
                          n_boot=N_BOOT, seed=SEED)
    daily_ic = eh.daily_rank_ic(pred, label)
    os.makedirs(DAILY_IC_DIR, exist_ok=True)
    daily_ic.rename("rank_ic").to_csv(os.path.join(DAILY_IC_DIR, f"{name}.csv"),
                                      header=True)
    # Persist the long-short net return series for equity-curve figures
    ls = eh.longshort_returns(pred, label, quantile=QUANTILE, fee=FEE)
    os.makedirs(LS_RETURNS_DIR, exist_ok=True)
    ls.to_csv(os.path.join(LS_RETURNS_DIR, f"{name}.csv"))

    row = {"model": name, "level": spec["level"], "family": spec["family"],
           "n_params": n_params, **metrics, "seconds": round(time.time() - t0, 1)}
    print(f"  {name:11s} [{spec['level']}] IC={metrics['ic_mean']:+.5f} "
          f"ICIR={metrics['icir']:+.3f} t={metrics['tstat']:+.2f} "
          f"Sharpe={metrics['sharpe']:+.2f} ({row['seconds']}s)")
    return row


def main():
    parser = argparse.ArgumentParser(description="Run the CPU model-complexity ladder")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--only", default=None,
                        help="Comma-separated subset of model names to run")
    args = parser.parse_args()

    registry = build_registry()
    selected = list(registry) if not args.only else [m.strip() for m in args.only.split(",")]

    print(f"Loading standard panel from {args.data_dir} ...")
    t0 = time.time()
    panel = dp.load_panel(args.data_dir)
    split = dp.flatten_split(panel)
    print(f"Panel X={panel.X.shape}  F={len(panel.feature_names)}  "
          f"test={panel.X.shape[0] - panel.val_end} days  ({time.time() - t0:.1f}s)\n")

    rows = []
    for name in selected:
        spec = registry[name]
        try:
            rows.append(evaluate_model(name, spec, panel, split))
        except ImportError as e:
            print(f"  {name:11s} SKIPPED ({e})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["model", "level", "family", "n_params", "ic_mean", "ic_std", "icir",
            "tstat", "pvalue", "pct_positive", "ic_ci_low", "ic_ci_high",
            "ic_pearson", "sharpe", "ann_return", "max_drawdown", "total_return",
            "beta", "turnover_mean", "n_days", "seconds"]
    df = df[[c for c in cols if c in df.columns]]
    out_path = os.path.join(RESULTS_DIR, "ladder_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} model results to {out_path}")
    print(df[["model", "level", "ic_mean", "icir", "tstat", "sharpe"]].to_string(index=False))


if __name__ == "__main__":
    main()
