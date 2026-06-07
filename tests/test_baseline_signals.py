"""Tests for zero-cost sanity baseline signals (lib/baseline_signals.py)."""
import numpy as np

from lib import baseline_signals as bs


def _toy():
    """X[T=2, N=3, F=3] with named CLOSE_d1, CLOSE_d2, VOL_d1 features."""
    feature_names = ["CLOSE_d1", "CLOSE_d2", "VOL_d1"]
    # stock 0: CLOSE_d1=1, CLOSE_d2=3 ; stock1: 2,2 ; stock2: 3,1  (VOL ignored)
    X = np.zeros((2, 3, 3))
    X[:, 0, 0], X[:, 0, 1] = 1.0, 3.0
    X[:, 1, 0], X[:, 1, 1] = 2.0, 2.0
    X[:, 2, 0], X[:, 2, 1] = 3.0, 1.0
    X[:, :, 2] = 99.0  # VOL must be excluded from momentum/reversal
    return X, feature_names


def test_zero_signal_is_all_zeros():
    X, names = _toy()
    sig = bs.zero_signal(X, names)
    assert sig.shape == (2, 3)
    assert np.all(sig == 0.0)


def test_momentum_is_mean_of_close_lags():
    X, names = _toy()
    mom = bs.momentum_signal(X, names, lags=2)
    # stock0 mean(1,3)=2 ; stock1 mean(2,2)=2 ; stock2 mean(3,1)=2
    np.testing.assert_allclose(mom[0], [2.0, 2.0, 2.0])


def test_reversal_is_negative_momentum():
    X, names = _toy()
    mom = bs.momentum_signal(X, names, lags=1)
    rev = bs.reversal_signal(X, names, lags=1)
    np.testing.assert_allclose(rev, -mom)


def test_lags_limit_selected_columns():
    X, names = _toy()
    # only CLOSE_d1 with lags=1: stock0=1, stock1=2, stock2=3
    mom = bs.momentum_signal(X, names, lags=1)
    np.testing.assert_allclose(mom[0], [1.0, 2.0, 3.0])
