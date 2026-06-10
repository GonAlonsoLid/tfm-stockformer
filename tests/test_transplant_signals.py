import numpy as np
import pytest
from lib import transplant_signals as ts


def test_trailing_returns_uses_only_past():
    dy = np.arange(40 * 3, dtype=float).reshape(40, 3)
    d = 30
    R = ts._trailing_returns(dy, d, window=10)
    assert R.shape == (10, 3)
    np.testing.assert_array_equal(R, dy[20:30])  # [d-window, d), strictly past


def test_trailing_returns_none_when_insufficient_history():
    dy = np.zeros((5, 3))
    assert ts._trailing_returns(dy, d=2, window=10) is None


def test_xz_zero_mean_unit_std():
    z = ts.xz(np.array([1.0, 2.0, 3.0, 4.0]))
    assert abs(float(np.mean(z))) < 1e-9
    assert abs(float(np.std(z)) - 1.0) < 1e-9


def test_peer_graph_signal_shape_and_nan_guard():
    rng = np.random.default_rng(0)
    dy = rng.normal(0, 0.01, size=(400, 12))
    sig = ts.peer_graph_signal(dy, d=300)
    assert sig.shape == (12,)
    assert np.all(np.isnan(ts.peer_graph_signal(dy, d=10)))


def test_peer_graph_signal_is_causal():
    rng = np.random.default_rng(1)
    dy = rng.normal(0, 0.01, size=(400, 12))
    d = 300
    s1 = ts.peer_graph_signal(dy, d)
    dy2 = dy.copy()
    dy2[d:] = rng.normal(0, 0.5, size=dy2[d:].shape)  # perturb only the FUTURE
    s2 = ts.peer_graph_signal(dy2, d)
    np.testing.assert_allclose(np.nan_to_num(s1), np.nan_to_num(s2), atol=1e-12)
    # ...but it MUST depend on the last past row (tight boundary, no fencepost off-by-one)
    dy3 = dy.copy()
    dy3[d - 1] = dy3[d - 1] * 1000.0
    s3 = ts.peer_graph_signal(dy3, d)
    assert not np.allclose(np.nan_to_num(s1), np.nan_to_num(s3), atol=1e-12), \
        "signal must depend on the last past row daily_y[d-1]"


def test_filtered_trend_signal_shape_and_nan_guard():
    rng = np.random.default_rng(2)
    dy = rng.normal(0, 0.01, size=(400, 8))
    sig = ts.filtered_trend_signal(dy, d=300)
    assert sig.shape == (8,)
    assert np.all(np.isnan(ts.filtered_trend_signal(dy, d=5)))


def test_filtered_trend_signal_is_causal():
    rng = np.random.default_rng(3)
    dy = rng.normal(0, 0.01, size=(400, 8))
    d = 300
    s1 = ts.filtered_trend_signal(dy, d)
    dy2 = dy.copy()
    dy2[d:] = rng.normal(0, 0.5, size=dy2[d:].shape)
    s2 = ts.filtered_trend_signal(dy2, d)
    np.testing.assert_allclose(np.nan_to_num(s1), np.nan_to_num(s2), atol=1e-12)
    # ...but it MUST depend on the last past row (tight boundary, no fencepost off-by-one)
    dy3 = dy.copy()
    dy3[d - 1] = dy3[d - 1] * 1000.0
    s3 = ts.filtered_trend_signal(dy3, d)
    assert not np.allclose(np.nan_to_num(s1), np.nan_to_num(s3), atol=1e-12), \
        "signal must depend on the last past row daily_y[d-1]"


def test_filtered_trend_signal_positive_for_uptrend():
    dy = np.zeros((200, 2))
    dy[:, 0] = 0.002   # steady uptrend
    dy[:, 1] = -0.002  # steady downtrend
    sig = ts.filtered_trend_signal(dy, d=180, window=120, halflife=10, mom_window=21)
    assert sig[0] > sig[1]


@pytest.mark.parametrize("fn", [
    ts.low_ivol_signal, ts.bab_signal, ts.volmanaged_momentum_signal,
    ts.fiftytwo_week_high_signal, ts.ts_momentum_signal,
])
def test_bench_signal_is_causal(fn):
    rng = np.random.default_rng(10)
    dy = rng.normal(0, 0.01, size=(500, 10))
    d = 400
    s1 = fn(dy, d)
    assert s1.shape == (10,)
    dy2 = dy.copy()
    dy2[d:] = rng.normal(0, 0.5, size=dy2[d:].shape)  # perturb only the FUTURE
    s2 = fn(dy2, d)
    np.testing.assert_allclose(np.nan_to_num(s1), np.nan_to_num(s2), atol=1e-12)


def test_seasonality_signal_is_causal():
    import pandas as pd
    rng = np.random.default_rng(11)
    dy = rng.normal(0, 0.01, size=(500, 10))
    dates = pd.bdate_range("2018-01-01", periods=500)
    d = 400
    s1 = ts.seasonality_signal(dy, d, dates)
    assert s1.shape == (10,)
    dy2 = dy.copy()
    dy2[d:] = rng.normal(0, 0.5, size=dy2[d:].shape)
    s2 = ts.seasonality_signal(dy2, d, dates)
    np.testing.assert_allclose(np.nan_to_num(s1), np.nan_to_num(s2), atol=1e-12)


@pytest.mark.parametrize("fn", [
    ts.low_ivol_signal, ts.bab_signal, ts.volmanaged_momentum_signal,
    ts.fiftytwo_week_high_signal,
])
def test_magnitude_bench_signal_depends_on_last_past_row(fn):
    # Tight boundary: a degenerate all-zero function would pass the causality test;
    # these magnitude-sensitive signals must actually respond to daily_y[d-1].
    rng = np.random.default_rng(12)
    dy = rng.normal(0, 0.01, size=(500, 10))
    d = 400
    s1 = fn(dy, d)
    dy3 = dy.copy()
    dy3[d - 1] = dy3[d - 1] * 1000.0
    s3 = fn(dy3, d)
    assert not np.allclose(np.nan_to_num(s1), np.nan_to_num(s3), atol=1e-12), \
        "signal must depend on the last past row daily_y[d-1]"


@pytest.mark.parametrize("fn", [ts.high_freq_reversal_signal, ts.lead_lag_signal])
def test_arch_round2_signal_causal_and_boundary(fn):
    rng = np.random.default_rng(20)
    dy = rng.normal(0, 0.01, size=(500, 12))
    d = 400
    s1 = fn(dy, d)
    assert s1.shape == (12,)
    dy2 = dy.copy()
    dy2[d:] = rng.normal(0, 0.5, size=dy2[d:].shape)  # future perturbation -> no change
    s2 = fn(dy2, d)
    np.testing.assert_allclose(np.nan_to_num(s1), np.nan_to_num(s2), atol=1e-12)
    dy3 = dy.copy()
    dy3[d - 1] = dy3[d - 1] * 1000.0  # last past row -> must change
    s3 = fn(dy3, d)
    assert not np.allclose(np.nan_to_num(s1), np.nan_to_num(s3), atol=1e-12)


def test_sector_peer_momentum_causal_and_excludes_self():
    rng = np.random.default_rng(21)
    dy = rng.normal(0, 0.01, size=(300, 9))
    sectors = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    d = 250
    s1 = ts.sector_peer_momentum_signal(dy, d, sectors)
    assert s1.shape == (9,)
    # causal: perturbing the future leaves it unchanged
    dy2 = dy.copy()
    dy2[d:] = rng.normal(0, 0.5, size=dy2[d:].shape)
    s2 = ts.sector_peer_momentum_signal(dy2, d, sectors)
    np.testing.assert_allclose(np.nan_to_num(s1), np.nan_to_num(s2), atol=1e-12)
    # peer mean excludes self: with 3 stocks per sector, stock 0's signal is the mean of 1 and 2
    mom = ts._trailing_returns(dy, d, 21).sum(axis=0)
    assert s1[0] == pytest.approx((mom[1] + mom[2]) / 2)
