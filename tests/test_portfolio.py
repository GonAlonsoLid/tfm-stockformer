"""Tests for cost-aware market-neutral portfolio construction (lib/portfolio.py)."""
import numpy as np

from lib import portfolio as pf


def test_weights_are_dollar_and_beta_neutral():
    rng = np.random.default_rng(0)
    n = 20
    alpha = rng.normal(size=n)
    beta = rng.normal(1.0, 0.3, size=n)
    w = pf.costaware_weights(alpha, beta, gross=2.0, name_cap=0.2, cost_bps=0.0)
    assert abs(w.sum()) < 1e-5            # dollar neutral
    assert abs(float(beta @ w)) < 1e-4    # beta neutral


def test_high_alpha_gets_long_low_alpha_gets_short():
    n = 10
    alpha = np.linspace(-1, 1, n)         # increasing
    beta = np.ones(n)
    w = pf.costaware_weights(alpha, beta, gross=2.0, name_cap=0.5, cost_bps=0.0)
    assert w[-1] > 0 > w[0]               # top long, bottom short


def test_gross_exposure_constrained():
    rng = np.random.default_rng(1)
    n = 15
    alpha = rng.normal(size=n)
    beta = rng.normal(1.0, 0.2, size=n)
    w = pf.costaware_weights(alpha, beta, gross=2.0, name_cap=0.5, cost_bps=0.0)
    assert np.abs(w).sum() <= 2.0 + 1e-4


def test_name_cap_respected():
    rng = np.random.default_rng(2)
    n = 12
    alpha = rng.normal(size=n)
    beta = np.ones(n)
    cap = 0.15
    w = pf.costaware_weights(alpha, beta, gross=2.0, name_cap=cap, cost_bps=0.0)
    assert np.abs(w).max() <= cap + 1e-4


def test_transaction_cost_reduces_turnover():
    rng = np.random.default_rng(3)
    n = 20
    beta = np.ones(n)
    alpha = rng.normal(size=n)
    w_prev = pf.costaware_weights(alpha, beta, gross=2.0, name_cap=0.2, cost_bps=0.0)
    # new, perturbed alpha (rankings shift)
    alpha2 = alpha + rng.normal(0, 1.0, size=n)
    w_free = pf.costaware_weights(alpha2, beta, gross=2.0, name_cap=0.2, cost_bps=0.0, w_prev=w_prev)
    # a dominating cost must suppress trading -> book stays near w_prev
    w_costly = pf.costaware_weights(alpha2, beta, gross=2.0, name_cap=0.2, cost_bps=1e6, w_prev=w_prev)
    turn_free = np.abs(w_free - w_prev).sum()
    turn_costly = np.abs(w_costly - w_prev).sum()
    assert turn_costly < turn_free        # L1 cost term suppresses turnover
    assert turn_costly < 1e-3             # dominating cost -> effectively no trade


def test_vol_target_scale_matches_target():
    rng = np.random.default_rng(4)
    # build a returns window for a 2-asset book with known vol
    R = rng.normal(0, 0.01, size=(250, 2))
    w = np.array([1.0, -1.0])
    scale = pf.vol_target_scale(R, w, target_ann_vol=0.10)
    scaled_daily_vol = np.std((R @ w) * scale, ddof=1)
    assert abs(scaled_daily_vol * np.sqrt(252) - 0.10) < 1e-6
