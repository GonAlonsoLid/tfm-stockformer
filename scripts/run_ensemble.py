#!/usr/bin/env python3
"""Heterogeneous ensemble combining Stockformer + LightGBM predictions.

Trains a Ridge meta-learner on out-of-fold daily cross-sectional predictions
and reports IC for each component model, the ensemble, and a simple average.

Usage:
    python scripts/run_ensemble.py \
        --stockformer_dir output/Multitask_output_SP500_2018-01-01_2026-03-16 \
        --lightgbm_dir output/lightgbm_baseline \
        --config config/Multitask_Stock_SP500.conf \
        --output_dir output/ensemble

If --lightgbm_dir does not contain predictions.csv, the script trains a fresh
LightGBM model inline using the same logic as run_lightgbm_baseline.py and
saves the predictions before proceeding.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib.config import load_config  # noqa: E402


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_stockformer_predictions(stockformer_dir: str) -> np.ndarray:
    """Load Stockformer regression predictions [T_test, N_stocks].

    Reads the headerless CSV produced by run_inference.py.
    """
    pred_path = os.path.join(
        stockformer_dir, "regression", "regression_pred_last_step.csv"
    )
    if not os.path.isfile(pred_path):
        print(f"ERROR: Stockformer predictions not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    preds = pd.read_csv(pred_path, header=None).values.astype(np.float64)
    print(f"Loaded Stockformer predictions  ->  shape {preds.shape}")
    return preds


def load_lightgbm_predictions(lightgbm_dir: str) -> np.ndarray:
    """Load LightGBM predictions [T_test, N_stocks] from predictions.csv."""
    pred_path = os.path.join(lightgbm_dir, "predictions.csv")
    if not os.path.isfile(pred_path):
        return None

    preds = pd.read_csv(pred_path, header=None).values.astype(np.float64)
    print(f"Loaded LightGBM predictions  ->  shape {preds.shape}")
    return preds


def train_lightgbm_inline(cfg, data_dir: str) -> tuple[np.ndarray, int]:
    """Train a LightGBM model inline and return (test_preds_2d, test_start_idx).

    Returns
    -------
    test_preds_2d : np.ndarray, shape [T_test, N]
        Predicted returns for the test period.
    test_date_offset : int
        Index of the first test date within the aligned label array,
        so the caller can align with Stockformer's test-only output.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        print(
            "ERROR: lightgbm not installed. Install with:  pip install lightgbm",
            file=sys.stderr,
        )
        sys.exit(1)

    features_dir = os.path.join(data_dir, "features")
    flow_path = cfg.traffic_file

    # Load Alpha360 features
    csv_files = sorted(f for f in os.listdir(features_dir) if f.endswith(".csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {features_dir}")

    feature_names = [os.path.splitext(f)[0] for f in csv_files]
    slices = []
    ref_shape = None
    for fname in csv_files:
        df = pd.read_csv(
            os.path.join(features_dir, fname), index_col=0, parse_dates=True
        )
        arr = df.values
        if ref_shape is None:
            ref_shape = arr.shape
        if arr.shape != ref_shape:
            T_min = min(arr.shape[0], ref_shape[0])
            N_min = min(arr.shape[1], ref_shape[1])
            arr = arr[:T_min, :N_min]
            if ref_shape != (T_min, N_min):
                ref_shape = (T_min, N_min)
                slices = [s[:T_min, :N_min] for s in slices]
        slices.append(arr)

    features = np.stack(slices, axis=-1)
    print(f"  LightGBM inline: loaded {len(csv_files)} features -> {features.shape}")

    # Load labels
    data = np.load(flow_path)
    labels = data["result"]

    # Align with offset=60
    LAG_BUFFER = 60
    T_feat, N_feat, F = features.shape
    T_lab, N_lab = labels.shape
    N = min(N_feat, N_lab)
    T = min(T_feat, T_lab - LAG_BUFFER)
    features = features[:T, :N, :]
    labels = labels[LAG_BUFFER : LAG_BUFFER + T, :N]

    # Filter MACRO features
    macro_mask = np.array(
        [not name.startswith("MACRO_") for name in feature_names]
    )
    if (~macro_mask).sum() > 0:
        features = features[:, :, macro_mask]
        feature_names = [n for n, keep in zip(feature_names, macro_mask) if keep]

    F = features.shape[-1]
    print(f"  LightGBM inline: aligned T={T}, N={N}, F={F}")

    # Flatten
    X = features.reshape(T * N, F)
    y = labels.reshape(T * N)
    date_indices = np.repeat(np.arange(T), N)
    stock_indices = np.tile(np.arange(N), T)
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[valid], y[valid]
    date_idx, stock_idx = date_indices[valid], stock_indices[valid]

    # Walk-forward split
    unique_dates = np.unique(date_idx)
    T_dates = len(unique_dates)
    train_end = int(T_dates * cfg.train_ratio)
    val_end = int(T_dates * (cfg.train_ratio + cfg.val_ratio))

    train_dates = set(unique_dates[:train_end])
    val_dates = set(unique_dates[train_end:val_end])
    test_dates_set = set(unique_dates[val_end:])

    train_mask = np.isin(date_idx, list(train_dates))
    val_mask = np.isin(date_idx, list(val_dates))
    test_mask = np.isin(date_idx, list(test_dates_set))

    print(
        f"  LightGBM inline: split train={train_end} | "
        f"val={val_end - train_end} | test={T_dates - val_end} dates"
    )

    # Train
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
        X[train_mask],
        y[train_mask],
        eval_set=[(X[val_mask], y[val_mask])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    print(f"  LightGBM inline: best iteration = {model.best_iteration_}")

    # Predict on test set
    y_pred_test = model.predict(X[test_mask])
    test_date_idx = date_idx[test_mask]
    test_stock_idx = stock_idx[test_mask]

    # Reconstruct 2D array [T_test, N]
    test_dates_sorted = sorted(test_dates_set)
    T_test = len(test_dates_sorted)
    preds_2d = np.full((T_test, N), np.nan)
    date_to_row = {d: i for i, d in enumerate(test_dates_sorted)}
    for k in range(len(y_pred_test)):
        row = date_to_row[test_date_idx[k]]
        col = test_stock_idx[k]
        preds_2d[row, col] = y_pred_test[k]

    print(f"  LightGBM inline: test predictions shape = {preds_2d.shape}")
    return preds_2d, val_end


def load_labels_for_test(cfg, data_dir: str) -> np.ndarray:
    """Load and align labels, returning only the test portion [T_test, N].

    Uses the same offset=60 alignment as run_lightgbm_baseline.py.
    """
    flow_path = cfg.traffic_file
    data = np.load(flow_path)
    labels = data["result"]

    # We need to figure out T from features to apply the same alignment.
    features_dir = os.path.join(data_dir, "features")
    sample_csv = sorted(f for f in os.listdir(features_dir) if f.endswith(".csv"))[0]
    sample_df = pd.read_csv(
        os.path.join(features_dir, sample_csv), index_col=0, parse_dates=True
    )
    T_feat = sample_df.shape[0]
    N_feat = sample_df.shape[1]

    LAG_BUFFER = 60
    T_lab, N_lab = labels.shape
    N = min(N_feat, N_lab)
    T = min(T_feat, T_lab - LAG_BUFFER)
    labels_aligned = labels[LAG_BUFFER : LAG_BUFFER + T, :N]

    # Test portion
    T_dates = T
    val_end = int(T_dates * (cfg.train_ratio + cfg.val_ratio))
    test_labels = labels_aligned[val_end:, :]

    print(f"Loaded test labels  ->  shape {test_labels.shape}")
    return test_labels


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def compute_daily_ic(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Compute cross-sectional Spearman IC per day.

    Parameters
    ----------
    y_true : np.ndarray, shape [T, N]
    y_pred : np.ndarray, shape [T, N]

    Returns
    -------
    ic_per_day : np.ndarray, shape [T]
    """
    T = y_true.shape[0]
    ic_per_day = np.full(T, np.nan)
    for d in range(T):
        yt = y_true[d]
        yp = y_pred[d]
        # Require at least 5 non-NaN stocks
        mask = ~(np.isnan(yt) | np.isnan(yp))
        if mask.sum() < 5:
            continue
        result = spearmanr(yt[mask], yp[mask])
        ic_val = (
            result.correlation if hasattr(result, "correlation") else result.statistic
        )
        ic_per_day[d] = ic_val
    return ic_per_day


def ic_summary(ic_per_day: np.ndarray) -> dict:
    """Compute mean IC, std, ICIR, pct positive from daily IC array."""
    valid = ic_per_day[~np.isnan(ic_per_day)]
    if len(valid) == 0:
        return {"mean_ic": np.nan, "std_ic": np.nan, "icir": np.nan, "pct_pos": np.nan}
    mean_ic = float(np.mean(valid))
    std_ic = float(np.std(valid, ddof=1))
    icir = mean_ic / std_ic if std_ic > 0 else 0.0
    pct_pos = float((valid > 0).mean() * 100)
    return {
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "icir": icir,
        "pct_pos": pct_pos,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Heterogeneous ensemble: Stockformer + LightGBM with Ridge meta-learner"
    )
    parser.add_argument(
        "--stockformer_dir",
        type=str,
        required=True,
        help="Stockformer output directory (contains regression/ subdirectory)",
    )
    parser.add_argument(
        "--lightgbm_dir",
        type=str,
        required=True,
        help="LightGBM output directory (contains predictions.csv, or will recompute)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to INI config file for data paths and split ratios",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/ensemble",
        help="Where to save ensemble results (default: output/ensemble)",
    )
    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge regularization strength (default: 1.0)",
    )
    cli_args = parser.parse_args()

    cfg = load_config(cli_args.config)
    data_dir = os.path.dirname(cfg.traffic_file)

    print("=" * 70)
    print("Heterogeneous Ensemble: Stockformer + LightGBM")
    print("=" * 70)
    print(f"Config          : {cli_args.config}")
    print(f"Stockformer dir : {cli_args.stockformer_dir}")
    print(f"LightGBM dir    : {cli_args.lightgbm_dir}")
    print(f"Output dir      : {cli_args.output_dir}")
    print(f"Ridge alpha     : {cli_args.ridge_alpha}")
    print()

    # ------------------------------------------------------------------
    # 1. Load Stockformer predictions [T_test, N_stocks]
    # ------------------------------------------------------------------
    sf_preds = load_stockformer_predictions(cli_args.stockformer_dir)

    # ------------------------------------------------------------------
    # 2. Load or recompute LightGBM predictions [T_test, N_stocks]
    # ------------------------------------------------------------------
    lgbm_preds = load_lightgbm_predictions(cli_args.lightgbm_dir)
    if lgbm_preds is None:
        print(
            f"LightGBM predictions.csv not found in {cli_args.lightgbm_dir}. "
            "Training LightGBM inline..."
        )
        print()
        lgbm_preds, _ = train_lightgbm_inline(cfg, data_dir)

        # Save for future reuse
        os.makedirs(cli_args.lightgbm_dir, exist_ok=True)
        lgbm_pred_path = os.path.join(cli_args.lightgbm_dir, "predictions.csv")
        pd.DataFrame(lgbm_preds).to_csv(lgbm_pred_path, header=False, index=False)
        print(f"  Saved LightGBM predictions to {lgbm_pred_path}")
        print()

    # ------------------------------------------------------------------
    # 3. Load actual test labels [T_test, N_stocks]
    # ------------------------------------------------------------------
    test_labels = load_labels_for_test(cfg, data_dir)

    # ------------------------------------------------------------------
    # 4. Align shapes across all three arrays
    # ------------------------------------------------------------------
    T_sf, N_sf = sf_preds.shape
    T_lgbm, N_lgbm = lgbm_preds.shape
    T_lab, N_lab = test_labels.shape

    T = min(T_sf, T_lgbm, T_lab)
    N = min(N_sf, N_lgbm, N_lab)

    # Align from the END of each array (test periods should end on the same
    # date; differences arise from varying start offsets).
    sf_preds = sf_preds[T_sf - T : T_sf, :N]
    lgbm_preds = lgbm_preds[T_lgbm - T : T_lgbm, :N]
    test_labels = test_labels[T_lab - T : T_lab, :N]

    print(f"Aligned shapes: T={T}, N={N}")
    print(
        f"  Stockformer original T={T_sf}, LightGBM original T={T_lgbm}, "
        f"Labels original T={T_lab}"
    )
    print()

    # ------------------------------------------------------------------
    # 5. Build valid mask (no NaN in any of the three arrays)
    # ------------------------------------------------------------------
    valid_mask = (
        ~np.isnan(sf_preds) & ~np.isnan(lgbm_preds) & ~np.isnan(test_labels)
    )

    # ------------------------------------------------------------------
    # 6. Split overlapping test dates: first half trains Ridge, second evaluates
    # ------------------------------------------------------------------
    mid = T // 2
    print(f"Meta-learner split: train on days 0..{mid - 1}, evaluate on days {mid}..{T - 1}")
    print()

    # Flatten train half
    train_sf = sf_preds[:mid]
    train_lgbm = lgbm_preds[:mid]
    train_labels = test_labels[:mid]
    train_valid = valid_mask[:mid]

    X_meta_train = np.column_stack(
        [train_sf[train_valid], train_lgbm[train_valid]]
    )
    y_meta_train = train_labels[train_valid]

    # Flatten eval half
    eval_sf = sf_preds[mid:]
    eval_lgbm = lgbm_preds[mid:]
    eval_labels = test_labels[mid:]
    eval_valid = valid_mask[mid:]

    X_meta_eval = np.column_stack(
        [eval_sf[eval_valid], eval_lgbm[eval_valid]]
    )
    y_meta_eval = eval_labels[eval_valid]

    print(f"Meta-learner training samples: {len(y_meta_train)}")
    print(f"Meta-learner eval samples    : {len(y_meta_eval)}")
    print()

    # ------------------------------------------------------------------
    # 7. Train Ridge meta-learner
    # ------------------------------------------------------------------
    ridge = Ridge(alpha=cli_args.ridge_alpha, fit_intercept=True)
    ridge.fit(X_meta_train, y_meta_train)
    print(
        f"Ridge coefficients: stockformer={ridge.coef_[0]:.4f}, "
        f"lightgbm={ridge.coef_[1]:.4f}, intercept={ridge.intercept_:.6f}"
    )
    print()

    # ------------------------------------------------------------------
    # 8. Generate ensemble predictions on eval half
    # ------------------------------------------------------------------
    ridge_pred_flat = ridge.predict(X_meta_eval)

    # Simple average on eval half
    avg_pred_flat = 0.5 * eval_sf[eval_valid] + 0.5 * eval_lgbm[eval_valid]

    # ------------------------------------------------------------------
    # 9. Compute daily IC for each method on eval half
    # ------------------------------------------------------------------
    T_eval = T - mid
    N_eval = N

    # Reconstruct 2D arrays for daily IC computation
    sf_eval_2d = eval_sf.copy()
    lgbm_eval_2d = eval_lgbm.copy()
    labels_eval_2d = eval_labels.copy()

    ridge_eval_2d = np.full((T_eval, N_eval), np.nan)
    avg_eval_2d = np.full((T_eval, N_eval), np.nan)

    # Fill in Ridge and average predictions where valid
    flat_idx = 0
    for d in range(T_eval):
        for s in range(N_eval):
            if eval_valid[d, s]:
                ridge_eval_2d[d, s] = ridge_pred_flat[flat_idx]
                avg_eval_2d[d, s] = avg_pred_flat[flat_idx]
                flat_idx += 1

    ic_sf = compute_daily_ic(labels_eval_2d, sf_eval_2d)
    ic_lgbm = compute_daily_ic(labels_eval_2d, lgbm_eval_2d)
    ic_ridge = compute_daily_ic(labels_eval_2d, ridge_eval_2d)
    ic_avg = compute_daily_ic(labels_eval_2d, avg_eval_2d)

    summary_sf = ic_summary(ic_sf)
    summary_lgbm = ic_summary(ic_lgbm)
    summary_ridge = ic_summary(ic_ridge)
    summary_avg = ic_summary(ic_avg)

    # ------------------------------------------------------------------
    # 10. Print summary table
    # ------------------------------------------------------------------
    print("=" * 70)
    print("IC Comparison on Evaluation Half (days {}-{})".format(mid, T - 1))
    print("=" * 70)
    header = f"{'Model':<20s} {'Mean IC':>10s} {'Std IC':>10s} {'ICIR':>10s} {'%Pos':>8s}"
    print(header)
    print("-" * len(header))

    for name, s in [
        ("Stockformer", summary_sf),
        ("LightGBM", summary_lgbm),
        ("Simple Avg (0.5/0.5)", summary_avg),
        ("Ridge Ensemble", summary_ridge),
    ]:
        print(
            f"{name:<20s} {s['mean_ic']:>10.4f} {s['std_ic']:>10.4f} "
            f"{s['icir']:>10.4f} {s['pct_pos']:>7.1f}%"
        )
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # 11. Save outputs
    # ------------------------------------------------------------------
    os.makedirs(cli_args.output_dir, exist_ok=True)

    # Daily IC comparison CSV
    daily_ic_df = pd.DataFrame(
        {
            "eval_day": np.arange(T_eval),
            "ic_stockformer": ic_sf,
            "ic_lightgbm": ic_lgbm,
            "ic_simple_avg": ic_avg,
            "ic_ridge_ensemble": ic_ridge,
        }
    )
    daily_path = os.path.join(cli_args.output_dir, "daily_ic_comparison.csv")
    daily_ic_df.to_csv(daily_path, index=False)
    print(f"Daily IC comparison saved to {daily_path}")

    # Summary CSV
    summary_rows = []
    for name, s in [
        ("stockformer", summary_sf),
        ("lightgbm", summary_lgbm),
        ("simple_avg", summary_avg),
        ("ridge_ensemble", summary_ridge),
    ]:
        summary_rows.append(
            {
                "model": name,
                "mean_ic": s["mean_ic"],
                "std_ic": s["std_ic"],
                "icir": s["icir"],
                "pct_positive": s["pct_pos"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(cli_args.output_dir, "ensemble_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # Ridge model info
    meta_info = {
        "ridge_alpha": cli_args.ridge_alpha,
        "coef_stockformer": ridge.coef_[0],
        "coef_lightgbm": ridge.coef_[1],
        "intercept": ridge.intercept_,
        "meta_train_days": mid,
        "meta_eval_days": T_eval,
        "meta_train_samples": len(y_meta_train),
        "meta_eval_samples": len(y_meta_eval),
    }
    meta_path = os.path.join(cli_args.output_dir, "ridge_meta_info.csv")
    pd.DataFrame([meta_info]).to_csv(meta_path, index=False)
    print(f"Ridge meta-learner info saved to {meta_path}")


if __name__ == "__main__":
    main()
