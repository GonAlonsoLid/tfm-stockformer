#!/usr/bin/env python3
"""LightGBM baseline for Alpha360 features on S&P 500.

Establishes the achievable IC (Information Coefficient) with Alpha360
features using a gradient-boosted tree model. This is critical to
determine whether poor IC comes from features or model architecture.

Usage:
    python scripts/run_lightgbm_baseline.py \
        --config config/Multitask_Stock_SP500.conf

    python scripts/run_lightgbm_baseline.py \
        --config config/Multitask_Stock_SP500.conf \
        --data_dir ./data/Stock_SP500_2018-01-01_2026-03-16
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import lightgbm as lgb
except ImportError:
    print(
        "ERROR: lightgbm is not installed. "
        "Install it with:  pip install lightgbm"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
# Allow running from the repo root (scripts/ lives one level below).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_alpha360_features(features_dir: str) -> np.ndarray:
    """Load all Alpha360 CSVs and stack into a 3D array [T, N, F].

    Each CSV has shape [T, N] with DatetimeIndex rows and ticker columns.
    Files are loaded in sorted order for deterministic feature indexing.

    Returns
    -------
    features : np.ndarray, shape [T, N, F]
    feature_names : list[str]
    """
    csv_files = sorted(
        f for f in os.listdir(features_dir) if f.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {features_dir}. "
            "Run the Alpha360 builder first."
        )

    feature_names = [os.path.splitext(f)[0] for f in csv_files]
    slices = []
    ref_shape = None
    skipped = []
    for fname in csv_files:
        df = pd.read_csv(
            os.path.join(features_dir, fname), index_col=0, parse_dates=True
        )
        arr = df.values  # [T, N]
        if ref_shape is None:
            ref_shape = arr.shape
        if arr.shape != ref_shape:
            # Align to reference shape (truncate or pad)
            T_min = min(arr.shape[0], ref_shape[0])
            N_min = min(arr.shape[1], ref_shape[1])
            arr = arr[:T_min, :N_min]
            if ref_shape != (T_min, N_min):
                # Update ref_shape to the minimum common shape
                ref_shape = (T_min, N_min)
                # Re-trim earlier slices
                slices = [s[:T_min, :N_min] for s in slices]
        slices.append(arr)

    # Stack along a new last axis -> [T, N, F]
    features = np.stack(slices, axis=-1)
    feature_names = [os.path.splitext(f)[0] for f in csv_files]
    print(f"Loaded {len(csv_files)} features  ->  shape {features.shape}")
    return features, feature_names


def load_labels(flow_path: str) -> np.ndarray:
    """Load forward returns from flow.npz.

    Returns
    -------
    labels : np.ndarray, shape [T, N]
    """
    data = np.load(flow_path)
    labels = data["result"]  # [T, N]
    print(f"Loaded labels  ->  shape {labels.shape}")
    return labels


def flatten_panel(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten 3D features and 2D labels to a 2D panel.

    Each row corresponds to a (date_idx, stock_idx) pair.
    NaN rows (missing feature or label) are dropped.

    Returns
    -------
    X : np.ndarray, shape [valid_rows, F]
    y : np.ndarray, shape [valid_rows]
    date_idx : np.ndarray, shape [valid_rows]
    stock_idx : np.ndarray, shape [valid_rows]
    """
    T, N, F = features.shape
    assert labels.shape == (T, N), (
        f"Shape mismatch: features {features.shape} vs labels {labels.shape}"
    )

    # Reshape to [T*N, F] and [T*N]
    X = features.reshape(T * N, F)
    y = labels.reshape(T * N)

    # Build index arrays
    date_indices = np.repeat(np.arange(T), N)
    stock_indices = np.tile(np.arange(N), T)

    # Drop rows with any NaN in features or label
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid]
    y = y[valid]
    date_idx = date_indices[valid]
    stock_idx = stock_indices[valid]

    print(
        f"Panel: {T}x{N} = {T * N} rows  ->  {len(y)} valid rows "
        f"({100 * len(y) / (T * N):.1f}%)"
    )
    return X, y, date_idx, stock_idx


def walk_forward_split(
    X: np.ndarray,
    y: np.ndarray,
    date_idx: np.ndarray,
    stock_idx: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> dict:
    """Walk-forward temporal split.

    Train and val are combined for LightGBM training; val is used as
    early-stopping evaluation set.

    Returns
    -------
    dict with keys: X_train, y_train, X_val, y_val, X_test, y_test,
                    date_idx_test, stock_idx_test
    """
    unique_dates = np.unique(date_idx)
    T = len(unique_dates)

    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates = set(unique_dates[val_end:])

    train_mask = np.isin(date_idx, list(train_dates))
    val_mask = np.isin(date_idx, list(val_dates))
    test_mask = np.isin(date_idx, list(test_dates))

    print(
        f"Split: train {train_end} dates | "
        f"val {val_end - train_end} dates | "
        f"test {T - val_end} dates"
    )

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val": X[val_mask],
        "y_val": y[val_mask],
        "X_test": X[test_mask],
        "y_test": y[test_mask],
        "date_idx_test": date_idx[test_mask],
        "stock_idx_test": stock_idx[test_mask],
    }


def compute_daily_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    date_idx: np.ndarray,
) -> pd.DataFrame:
    """Compute daily Spearman rank IC between predictions and actuals.

    Returns a DataFrame with columns [date_idx, ic, n_stocks].
    """
    records = []
    for d in np.unique(date_idx):
        mask = date_idx == d
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < 5:
            continue
        ic, _ = spearmanr(yt, yp)
        records.append({"date_idx": d, "ic": ic, "n_stocks": len(yt)})

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LightGBM baseline with Alpha360 features"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to INI config file (e.g. config/Multitask_Stock_SP500.conf)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )
    cli_args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = load_config(cli_args.config)
    data_dir = cli_args.data_dir or os.path.dirname(cfg.traffic_file)
    features_dir = os.path.join(data_dir, "features")
    flow_path = cfg.traffic_file if cli_args.data_dir is None else os.path.join(data_dir, "flow.npz")

    train_ratio = cfg.train_ratio
    val_ratio = cfg.val_ratio

    print("=" * 60)
    print("LightGBM Baseline  --  Alpha360 on S&P 500")
    print("=" * 60)
    print(f"Config       : {cli_args.config}")
    print(f"Data dir     : {data_dir}")
    print(f"Features dir : {features_dir}")
    print(f"Flow path    : {flow_path}")
    print(f"Split ratios : train={train_ratio} / val={val_ratio} / test={cfg.test_ratio}")
    print()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    features, feature_names = load_alpha360_features(features_dir)
    labels = load_labels(flow_path)

    # Temporal alignment: Alpha360 discards the first LAG_BUFFER=60 rows of
    # OHLCV. label.csv starts at OHLCV date d_0, while features start at d_60.
    # We must use offset=60 (the actual lag buffer), NOT T_lab - T_feat (=59),
    # because label.csv also lost 1 row from pct_change().shift(-1).iloc[:-1].
    # Using 59 causes a 1-day misalignment where CLOSE_d1 ≡ label + 1 (leakage).
    LAG_BUFFER = 60  # Must match build_alpha360.py
    T_feat, N_feat, F = features.shape
    T_lab, N_lab = labels.shape
    N = min(N_feat, N_lab)
    T = min(T_feat, T_lab - LAG_BUFFER)
    features = features[:T, :N, :]
    labels = labels[LAG_BUFFER:LAG_BUFFER + T, :N]
    print(f"Aligned to T={T}, N={N}, F={F} (label offset={LAG_BUFFER}, matching Alpha360 lag buffer)")
    print()

    # ------------------------------------------------------------------
    # Filter out MACRO features (constant across stocks, can't rank)
    # ------------------------------------------------------------------
    macro_mask = np.array([not name.startswith("MACRO_") for name in feature_names])
    n_excluded = (~macro_mask).sum()
    if n_excluded > 0:
        features = features[:, :, macro_mask]
        feature_names = [n for n, keep in zip(feature_names, macro_mask) if keep]
        F = features.shape[-1]
        print(f"Excluded {n_excluded} MACRO features (constant cross-sectionally, can't rank stocks)")
        print(f"Remaining features: {F}")
    print()

    # ------------------------------------------------------------------
    # Flatten and split
    # ------------------------------------------------------------------
    X, y, date_idx, stock_idx = flatten_panel(features, labels)
    splits = walk_forward_split(X, y, date_idx, stock_idx, train_ratio, val_ratio)
    print()

    # ------------------------------------------------------------------
    # Train LightGBM
    # ------------------------------------------------------------------
    print("Training LightGBM ...")
    model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=300,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.7,
        colsample_bytree=0.5,
        min_child_samples=100,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        splits["X_train"],
        splits["y_train"],
        eval_set=[(splits["X_val"], splits["y_val"])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    print(f"Best iteration: {model.best_iteration_}")
    print()

    # ------------------------------------------------------------------
    # Predict on test set
    # ------------------------------------------------------------------
    y_pred = model.predict(splits["X_test"])

    # ------------------------------------------------------------------
    # Compute IC
    # ------------------------------------------------------------------
    ic_df = compute_daily_ic(
        splits["y_test"], y_pred, splits["date_idx_test"]
    )

    mean_ic = ic_df["ic"].mean()
    std_ic = ic_df["ic"].std()
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0.0
    pct_positive = (ic_df["ic"] > 0).mean() * 100

    print("=" * 60)
    print("Test-Period IC Summary")
    print("=" * 60)
    print(f"  Number of test dates : {len(ic_df)}")
    print(f"  Mean IC              : {mean_ic:.4f}")
    print(f"  Std IC               : {std_ic:.4f}")
    print(f"  IC IR (mean/std)     : {ic_ir:.4f}")
    print(f"  % dates with IC > 0  : {pct_positive:.1f}%")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Feature importance (top 20)
    # ------------------------------------------------------------------
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:20]
    print("\nTop 20 features by split importance:")
    for rank, idx in enumerate(top_idx, 1):
        name = feature_names[idx] if idx < len(feature_names) else f"f{idx}"
        print(f"  {rank:2d}. {name:30s}  importance={importance[idx]}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = os.path.join("output", "lightgbm_baseline")
    os.makedirs(output_dir, exist_ok=True)

    summary = pd.DataFrame(
        {
            "metric": ["mean_ic", "std_ic", "ic_ir", "pct_positive", "n_test_dates"],
            "value": [mean_ic, std_ic, ic_ir, pct_positive, len(ic_df)],
        }
    )
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Also save daily IC series
    daily_path = os.path.join(output_dir, "daily_ic.csv")
    ic_df.to_csv(daily_path, index=False)
    print(f"Daily IC saved to {daily_path}")


if __name__ == "__main__":
    main()
