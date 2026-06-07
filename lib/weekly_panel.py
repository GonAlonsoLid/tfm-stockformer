"""Daily -> weekly resampling for the market-neutral weekly strategy (TFM).

Weekly rebalancing is the single biggest net-Sharpe lever (cuts turnover ~5x).
Each weekly observation samples features at the last trading day of the ISO week
and uses the compounded forward return to the next rebalance day as the label.
The daily return matrix is retained for rolling-beta and vol-target windows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lib.data_panel import Panel


@dataclass(frozen=True)
class WeeklyPanel:
    Xw: np.ndarray            # [W, N, F] features at each rebalance day
    yw: np.ndarray            # [W, N] compounded forward weekly returns
    rebal_idx: np.ndarray     # [W] daily index of each rebalance day
    dates_w: pd.DatetimeIndex # [W] rebalance dates
    tickers: list[str]
    feature_names: list[str]
    daily_y: np.ndarray       # [T, N] daily forward returns (for beta/vol windows)
    train_end_w: int
    val_end_w: int


def build_weekly(panel: Panel) -> WeeklyPanel:
    """Resample a daily Panel to a weekly panel (last trading day per ISO week)."""
    dates = pd.DatetimeIndex(panel.dates)
    iso = dates.isocalendar()
    week_key = (iso["year"].to_numpy() * 100 + iso["week"].to_numpy())

    # last daily index of each consecutive ISO week
    rebal = []
    for i in range(len(dates)):
        if i == len(dates) - 1 or week_key[i] != week_key[i + 1]:
            rebal.append(i)
    rebal = np.array(rebal)

    # forward weekly return between consecutive rebalance days; drop last (no fwd)
    W = len(rebal) - 1
    N, F = panel.X.shape[1], panel.X.shape[2]
    Xw = np.empty((W, N, F))
    yw = np.empty((W, N))
    for i in range(W):
        a, b = rebal[i], rebal[i + 1]
        Xw[i] = panel.X[a]
        yw[i] = np.prod(1 + panel.y[a:b], axis=0) - 1

    rebal_idx = rebal[:W]
    dates_w = dates[rebal_idx]
    train_end_w = int(np.searchsorted(rebal_idx, panel.train_end))
    val_end_w = int(np.searchsorted(rebal_idx, panel.val_end))

    return WeeklyPanel(Xw=Xw, yw=yw, rebal_idx=rebal_idx, dates_w=dates_w,
                       tickers=panel.tickers, feature_names=panel.feature_names,
                       daily_y=panel.y, train_end_w=train_end_w, val_end_w=val_end_w)
