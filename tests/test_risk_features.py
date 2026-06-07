"""Tests for realized-vol / IVOL risk features (lib/risk_features.py)."""
import numpy as np

from lib import risk_features as rf


def test_stock_equal_to_market_has_beta_one_zero_ivol():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, size=80)
    R = np.column_stack([m, m * 2.0])      # stock0 == market, stock1 == 2x market
    f = rf.window_features(R, m)
    assert abs(f["beta"][0] - 1.0) < 1e-6
    assert abs(f["beta"][1] - 2.0) < 1e-6
    assert f["ivol"][0] < 1e-9            # perfectly explained by market
    assert f["ivol"][1] < 1e-9


def test_rvol_and_maxret_match_definitions():
    R = np.array([[0.01, -0.02], [0.03, 0.00], [-0.01, 0.04]])
    m = R.mean(axis=1)
    f = rf.window_features(R, m)
    np.testing.assert_allclose(f["rvol"], R.std(axis=0), rtol=1e-9)
    np.testing.assert_allclose(f["maxret"], R.max(axis=0), rtol=1e-9)


def test_constant_stock_has_zero_vol_and_zero_beta():
    m = np.array([0.01, -0.01, 0.02, 0.0])
    R = np.column_stack([np.zeros(4), m])  # stock0 constant
    f = rf.window_features(R, m)
    assert f["rvol"][0] == 0.0
    assert f["beta"][0] == 0.0
    assert f["ivol"][0] == 0.0


def test_ivol_positive_when_idiosyncratic_noise_present():
    rng = np.random.default_rng(1)
    m = rng.normal(0, 0.01, size=120)
    R = np.column_stack([m + rng.normal(0, 0.02, size=120)])  # market + noise
    f = rf.window_features(R, m)
    assert f["ivol"][0] > 0.005
