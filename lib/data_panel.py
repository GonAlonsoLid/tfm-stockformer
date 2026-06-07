"""Shared standard-panel loader for the model-complexity ladder (TFM).

Guarantees every model — linear, trees, MLP, StockMixer, Stockformer — sees
the SAME features, universe, temporal alignment and fixed split. This encodes
the "innegociable" methodology: identical inputs, only the model changes.

Standard fixed by the thesis:
    - Features  : F-base (Alpha360 + technicals), MACRO excluded for ranking
    - Alignment : labels offset by the Alpha360 lag buffer (60), matching both
                  run_lightgbm_baseline.py and lib/Multitask_Stockformer_utils.py
    - Split     : fixed temporal 0.75 / 0.125 / 0.125, normalization train-only

The label/feature offset is the single anti-leakage invariant verified by
scripts/audit_alignment.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ALPHA360_LAG = 60  # must match build_alpha360.py / StockDataset / LightGBM baseline


@dataclass(frozen=True)
class Panel:
    """An aligned feature/label panel with a fixed temporal split.

    X            : float array [T, N, F]   (NaN-free, cross-sectionally z-scored)
    y            : float array [T, N]      (realized forward returns)
    dates        : DatetimeIndex, length T (post-alignment trading calendar)
    tickers      : list[str], length N
    feature_names: list[str], length F
    train_end    : int  (first val index)
    val_end      : int  (first test index)
    """
    X: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    tickers: list[str]
    feature_names: list[str]
    train_end: int
    val_end: int


def load_panel(data_dir: str, exclude_macro: bool = True, lag: int = ALPHA360_LAG,
               train_ratio: float = 0.75, val_ratio: float = 0.125) -> Panel:
    """Load the standard aligned panel from a built data directory.

    Parameters
    ----------
    data_dir : str
        Directory with features/, flow.npz, label.csv, tickers.txt.
    exclude_macro : bool
        Drop MACRO_* features (constant cross-sectionally, useless for ranking).
    lag : int
        Label offset matching the Alpha360 lag buffer (default 60).
    train_ratio, val_ratio : float
        Fixed split fractions applied to the post-alignment length.
    """
    features_dir = os.path.join(data_dir, "features")
    csv_files = sorted(f for f in os.listdir(features_dir) if f.endswith(".csv"))
    if exclude_macro:
        csv_files = [f for f in csv_files if not f.startswith("MACRO_")]
    if not csv_files:
        raise FileNotFoundError(f"No usable feature CSVs in {features_dir}")
    feature_names = [os.path.splitext(f)[0] for f in csv_files]

    slices = []
    ref_shape = None
    for fname in csv_files:
        arr = pd.read_csv(os.path.join(features_dir, fname), index_col=0).values
        if ref_shape is None:
            ref_shape = arr.shape
        if arr.shape != ref_shape:
            t = min(arr.shape[0], ref_shape[0])
            n = min(arr.shape[1], ref_shape[1])
            ref_shape = (t, n)
            slices = [s[:t, :n] for s in slices]
            arr = arr[:t, :n]
        slices.append(arr)
    X = np.stack(slices, axis=-1).astype(np.float64)  # [T_feat, N, F]
    X = np.nan_to_num(X, nan=0.0)  # z-scored features: 0 == cross-sectional mean

    labels = np.load(os.path.join(data_dir, "flow.npz"))["result"]  # [T_lab, N]

    # Align: features start `lag` rows after the raw grid; labels[lag:] pair with X.
    T_feat, N_feat, F = X.shape
    T_lab, N_lab = labels.shape
    N = min(N_feat, N_lab)
    T = min(T_feat, T_lab - lag)
    X = X[:T, :N, :]
    y = labels[lag:lag + T, :N].astype(np.float64)

    dates = _load_dates(data_dir, lag, T)
    tickers = _load_tickers(data_dir, N)

    train_end = int(train_ratio * T)
    val_end = int((train_ratio + val_ratio) * T)
    return Panel(X=X, y=y, dates=dates, tickers=tickers,
                 feature_names=feature_names, train_end=train_end, val_end=val_end)


def to_canonical(matrix: np.ndarray, dates, tickers: list[str]) -> pd.DataFrame:
    """Wrap a [days x stocks] matrix as the canonical [dates x tickers] frame."""
    return pd.DataFrame(np.asarray(matrix), index=pd.DatetimeIndex(dates),
                        columns=list(tickers))


def flatten_split(panel: Panel) -> dict:
    """Flatten the panel to a tabular train/val/test split for non-sequential models.

    Each row is a (date, stock) pair. Rows with any NaN feature or NaN label are
    dropped. Test rows carry their global date/stock indices so predictions can be
    rebuilt into the canonical [dates x tickers] frame.
    """
    T, N, F = panel.X.shape
    Xflat = panel.X.reshape(T * N, F)
    yflat = panel.y.reshape(T * N)
    date_idx = np.repeat(np.arange(T), N)
    stock_idx = np.tile(np.arange(N), T)

    valid = ~(np.isnan(Xflat).any(axis=1) | np.isnan(yflat))
    Xflat, yflat = Xflat[valid], yflat[valid]
    date_idx, stock_idx = date_idx[valid], stock_idx[valid]

    tr = date_idx < panel.train_end
    va = (date_idx >= panel.train_end) & (date_idx < panel.val_end)
    te = date_idx >= panel.val_end
    return {
        "X_train": Xflat[tr], "y_train": yflat[tr],
        "X_val": Xflat[va], "y_val": yflat[va],
        "X_test": Xflat[te], "y_test": yflat[te],
        "date_idx_test": date_idx[te], "stock_idx_test": stock_idx[te],
    }


# ── Internals ──────────────────────────────────────────────────────────────────

def _load_dates(data_dir: str, lag: int, T: int) -> pd.DatetimeIndex:
    """Post-alignment trading calendar: label.csv Date column from row `lag`."""
    label_path = os.path.join(data_dir, "label.csv")
    if os.path.isfile(label_path):
        col = pd.read_csv(label_path, usecols=[0])
        all_dates = pd.to_datetime(col.iloc[:, 0])
        aligned = all_dates.iloc[lag:lag + T]
        return pd.DatetimeIndex(aligned.values)
    # Fallback: synthetic business-day calendar (keeps loader usable without label.csv)
    return pd.bdate_range("2000-01-01", periods=T)


def _load_tickers(data_dir: str, n: int) -> list[str]:
    path = os.path.join(data_dir, "tickers.txt")
    if os.path.isfile(path):
        with open(path) as f:
            tickers = [line.strip() for line in f if line.strip()]
        return tickers[:n]
    return [f"S{i}" for i in range(n)]
