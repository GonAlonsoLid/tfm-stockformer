import numpy as np
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
