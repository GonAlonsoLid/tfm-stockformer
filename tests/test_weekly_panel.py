"""Tests for daily->weekly resampling (lib/weekly_panel.py)."""
import numpy as np
import pandas as pd

from lib import data_panel as dp
from lib import weekly_panel as wp


def _daily_panel(n_days=20, N=3, F=2):
    dates = pd.bdate_range("2024-01-01", periods=n_days)  # business days
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_days, N, F))
    y = rng.normal(0, 0.01, size=(n_days, N))            # daily forward returns
    return dp.Panel(X=X, y=y, dates=dates, tickers=[f"S{i}" for i in range(N)],
                    feature_names=[f"f{j}" for j in range(F)],
                    train_end=int(0.6 * n_days), val_end=int(0.8 * n_days))


def test_weekly_count_matches_completed_weeks():
    p = _daily_panel(n_days=20)        # 4 ISO weeks of business days
    w = wp.build_weekly(p)
    # last week has no forward return -> dropped; expect (#weeks - 1) rows
    assert w.Xw.shape[0] == w.yw.shape[0]
    assert w.Xw.shape[0] >= 2
    assert w.Xw.shape[1:] == p.X.shape[1:]


def test_weekly_label_is_compounded_daily_return():
    p = _daily_panel(n_days=20)
    w = wp.build_weekly(p)
    # first weekly forward return = product of daily forward returns between
    # the first two rebalance days, per stock
    a, b = w.rebal_idx[0], w.rebal_idx[1]
    expected = np.prod(1 + p.y[a:b], axis=0) - 1
    np.testing.assert_allclose(w.yw[0], expected, rtol=1e-9)


def test_weekly_features_sampled_at_rebalance_day():
    p = _daily_panel(n_days=20)
    w = wp.build_weekly(p)
    np.testing.assert_allclose(w.Xw[0], p.X[w.rebal_idx[0]])


def test_weekly_split_boundaries_are_monotonic():
    p = _daily_panel(n_days=40)
    w = wp.build_weekly(p)
    assert 0 < w.train_end_w <= w.val_end_w <= w.Xw.shape[0]
