"""Tests for the shared standard-panel loader (lib/data_panel.py).

Guarantees every model in the ladder sees the SAME features, universe,
alignment (offset) and fixed temporal split.
"""
import json

import numpy as np
import pandas as pd
import pytest

from lib import data_panel as dp


# ── Synthetic data_dir fixture ─────────────────────────────────────────────────

@pytest.fixture
def fake_data_dir(tmp_path):
    """Build a minimal data_dir: 3 features (1 MACRO) + flow.npz + label.csv.

    T_label = 12 rows, N = 4 tickers. Features start 'lag' rows later than the
    raw OHLCV grid (mimicking the Alpha360 lag buffer), so with lag=2 the
    feature panel has T_feat = 10 rows.
    """
    T_label, N, lag = 12, 4, 2
    tickers = ["AAA", "BBB", "CCC", "DDD"]
    dates = pd.bdate_range("2020-01-01", periods=T_label)

    # labels: distinct deterministic values
    rng = np.random.default_rng(0)
    labels = rng.normal(0, 0.02, (T_label, N)).astype(np.float32)
    np.savez(tmp_path / "flow.npz", result=labels)

    # label.csv carries the trading calendar (Date column + ticker columns)
    lab_df = pd.DataFrame(labels, columns=tickers)
    lab_df.insert(0, "Date", dates.strftime("%Y-%m-%d"))
    lab_df.to_csv(tmp_path / "label.csv", index=False)

    # features start at date[lag]; T_feat = T_label - lag = 10 rows
    feat_dates = dates[lag:]
    T_feat = len(feat_dates)
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    for name in ["CLOSE_d1", "VOL_d1", "MACRO_VIX_level"]:
        vals = rng.normal(0, 1, (T_feat, N))
        df = pd.DataFrame(vals, index=feat_dates, columns=tickers)
        df.to_csv(feat_dir / f"{name}.csv")

    (tmp_path / "tickers.txt").write_text("\n".join(tickers) + "\n")
    return tmp_path, dict(T_label=T_label, N=N, lag=lag, T_feat=T_feat,
                          tickers=tickers, dates=dates)


# ── load_panel ─────────────────────────────────────────────────────────────────

def test_excludes_macro_features_by_default(fake_data_dir):
    data_dir, meta = fake_data_dir
    panel = dp.load_panel(str(data_dir), lag=meta["lag"])
    assert "MACRO_VIX_level" not in panel.feature_names
    assert set(panel.feature_names) == {"CLOSE_d1", "VOL_d1"}
    assert panel.X.shape[2] == 2


def test_keep_macro_when_requested(fake_data_dir):
    data_dir, meta = fake_data_dir
    panel = dp.load_panel(str(data_dir), lag=meta["lag"], exclude_macro=False)
    assert "MACRO_VIX_level" in panel.feature_names
    assert panel.X.shape[2] == 3


def test_labels_aligned_with_offset(fake_data_dir):
    data_dir, meta = fake_data_dir
    panel = dp.load_panel(str(data_dir), lag=meta["lag"])
    # After offset, T = T_label - lag = T_feat
    T = meta["T_label"] - meta["lag"]
    assert panel.X.shape[0] == T
    assert panel.y.shape == (T, meta["N"])
    # y row 0 must equal raw label row `lag` (offset applied)
    raw = np.load(data_dir / "flow.npz")["result"]
    np.testing.assert_allclose(panel.y[0], raw[meta["lag"]], rtol=1e-6)


def test_dates_match_aligned_calendar(fake_data_dir):
    data_dir, meta = fake_data_dir
    panel = dp.load_panel(str(data_dir), lag=meta["lag"])
    # post-alignment dates are the label calendar from row `lag` onward
    expected = meta["dates"][meta["lag"]:]
    assert list(panel.dates) == list(expected)
    assert panel.tickers == meta["tickers"]


def test_fixed_split_boundaries(fake_data_dir):
    data_dir, meta = fake_data_dir
    panel = dp.load_panel(str(data_dir), lag=meta["lag"],
                          train_ratio=0.6, val_ratio=0.2)
    T = meta["T_label"] - meta["lag"]  # 10
    assert panel.train_end == int(0.6 * T)        # 6
    assert panel.val_end == int(0.8 * T)          # 8


def test_no_nan_in_features(fake_data_dir):
    data_dir, meta = fake_data_dir
    panel = dp.load_panel(str(data_dir), lag=meta["lag"])
    assert not np.isnan(panel.X).any()


# ── to_canonical ───────────────────────────────────────────────────────────────

def test_to_canonical_builds_dates_by_tickers_frame(fake_data_dir):
    data_dir, meta = fake_data_dir
    panel = dp.load_panel(str(data_dir), lag=meta["lag"])
    test_dates = panel.dates[panel.val_end:]
    matrix = np.arange(len(test_dates) * meta["N"]).reshape(len(test_dates), meta["N"])
    frame = dp.to_canonical(matrix, test_dates, panel.tickers)
    assert list(frame.columns) == meta["tickers"]
    assert list(frame.index) == list(test_dates)
    assert frame.shape == (len(test_dates), meta["N"])


# ── flatten_split (for tabular models) ─────────────────────────────────────────

def test_flatten_split_shapes_and_test_indexing(fake_data_dir):
    data_dir, meta = fake_data_dir
    panel = dp.load_panel(str(data_dir), lag=meta["lag"])
    s = dp.flatten_split(panel)
    F = panel.X.shape[2]
    assert s["X_train"].shape[1] == F
    # test rows = (T - val_end) * N  (no NaNs in synthetic data)
    n_test_days = panel.X.shape[0] - panel.val_end
    assert s["X_test"].shape[0] == n_test_days * meta["N"]
    assert s["date_idx_test"].shape[0] == s["X_test"].shape[0]
    # date_idx_test values are global date indices into panel.dates
    assert s["date_idx_test"].min() >= panel.val_end
