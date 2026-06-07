"""Tests for cross-sectional signal neutralization (lib/neutralize.py)."""
import numpy as np

from lib import neutralize as nz


def test_residual_zero_when_signal_is_linear_in_exposure():
    rng = np.random.default_rng(0)
    beta = rng.normal(size=30)
    signal = 3.0 + 2.0 * beta            # perfectly explained by the exposure
    resid = nz.neutralize(signal, beta.reshape(-1, 1))
    assert np.allclose(resid, 0.0, atol=1e-8)


def test_residual_uncorrelated_with_exposure():
    rng = np.random.default_rng(1)
    beta = rng.normal(size=200)
    signal = rng.normal(size=200)
    resid = nz.neutralize(signal, beta.reshape(-1, 1))
    assert abs(np.corrcoef(resid, beta)[0, 1]) < 1e-8


def test_multiple_exposures_fully_explained():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    signal = X @ np.array([1.5, -0.7, 0.3]) + 0.4
    resid = nz.neutralize(signal, X)
    assert np.allclose(resid, 0.0, atol=1e-8)


def test_output_is_standardized_by_default():
    rng = np.random.default_rng(3)
    beta = rng.normal(size=300)
    signal = rng.normal(size=300)
    resid = nz.neutralize(signal, beta.reshape(-1, 1))
    assert abs(resid.mean()) < 1e-9
    assert abs(resid.std(ddof=0) - 1.0) < 1e-6


def test_nan_rows_excluded_and_preserved():
    signal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    beta = np.array([1.0, 2.0, 3.0, 4.0, 5.0]).reshape(-1, 1)
    resid = nz.neutralize(signal, beta)
    assert np.isnan(resid[2])
    assert not np.isnan(resid[[0, 1, 3, 4]]).any()
