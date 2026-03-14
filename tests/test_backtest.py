"""Test contract for Phase 5 — portfolio construction and backtesting.

Requirement-to-test mapping:
    PORT-01 → test_top_k_selection   — top-K ticker selection by score
    PORT-02 → test_equal_weight      — equal-weight allocation (1/k per ticker)
    PORT-03 → test_transaction_cost  — daily return with turnover cost deducted
    BACK-01 → test_cumulative_return — cumulative return series (1+r).cumprod()
    BACK-02 → test_performance_metrics — annualized return, Sharpe, max drawdown
    BACK-03 → test_alpha_beta        — OLS alpha/beta vs SPY benchmark

All imports from scripts.run_backtest are inside test bodies to avoid
ImportError before the implementation module exists (Wave 0 pattern).
"""
import pytest
import numpy as np
import pandas as pd


# ── PORT-01: top-K selection ──────────────────────────────────────────────────

def test_top_k_selection():
    from scripts.run_backtest import select_top_k

    tickers = [f"T{i:02d}" for i in range(10)]
    # Give T05, T02, T08 the highest scores
    scores = pd.Series(
        [0.1, 0.2, 0.5, 0.0, -0.1, 0.9, 0.3, 0.0, 0.7, -0.5],
        index=tickers,
    )
    result = select_top_k(scores, k=3)

    assert isinstance(result, pd.Index), "select_top_k must return a pd.Index"
    assert len(result) == 3, "Must return exactly k=3 tickers"
    # Top-3 by score: T05 (0.9), T08 (0.7), T02 (0.5)
    assert set(result) == {"T05", "T08", "T02"}, f"Wrong top-3 tickers: {set(result)}"


# ── PORT-02: equal-weight allocation ─────────────────────────────────────────

def test_equal_weight():
    from scripts.run_backtest import build_portfolio_weights

    all_tickers = [f"T{i:02d}" for i in range(10)]
    top_k_index = pd.Index(["T05", "T08", "T02"])
    k = 3

    weights = build_portfolio_weights(top_k_index, all_tickers, k)

    assert isinstance(weights, pd.Series), "Must return pd.Series"
    assert set(weights.index) == set(all_tickers), "Index must cover all tickers"
    # Selected tickers get 1/k
    for t in ["T05", "T08", "T02"]:
        assert abs(weights[t] - 1 / k) < 1e-9, f"{t} weight should be 1/{k}"
    # Unselected tickers get 0
    for t in all_tickers:
        if t not in ["T05", "T08", "T02"]:
            assert weights[t] == 0.0, f"{t} should have weight 0"
    # Weights sum to 1
    assert abs(weights.sum() - 1.0) < 1e-9, "Weights must sum to 1"


# ── PORT-03: transaction cost ─────────────────────────────────────────────────

def test_transaction_cost():
    from scripts.run_backtest import compute_daily_return

    tickers = ["A", "B", "C"]
    # New portfolio: equal weight across A, B, C
    weight_now = pd.Series([1 / 3, 1 / 3, 1 / 3], index=tickers)
    # Previous: all cash (zeros)
    weight_prev = pd.Series([0.0, 0.0, 0.0], index=tickers)
    # All tickers return 1% today
    price_returns = pd.Series([0.01, 0.01, 0.01], index=tickers)

    daily_return = compute_daily_return(weight_now, weight_prev, price_returns, fee=0.001)

    # Gross return: (1/3 * 0.01) * 3 = 0.01
    gross = (weight_now * price_returns).sum()
    # Turnover: |1/3 - 0| * 3 = 1.0 (buying from full cash)
    turnover = (weight_now - weight_prev).abs().sum()
    cost = turnover * 0.001
    expected = gross - cost

    assert abs(daily_return - expected) < 1e-12, (
        f"Expected {expected:.8f}, got {daily_return:.8f}"
    )
    # Buying from cash: turnover = 1.0, cost = 0.001
    assert abs(cost - 0.001) < 1e-12, f"Cost from full cash should be 0.001, got {cost}"
    # Net return = gross - cost = 0.01 - 0.001 = 0.009
    assert abs(daily_return - 0.009) < 1e-12, f"Net return should be 0.009, got {daily_return}"


# ── BACK-01: cumulative return ────────────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="Wave 0 stub — not yet implemented")
def test_cumulative_return():
    from scripts.run_backtest import compute_performance_metrics  # noqa: F401
    pytest.fail("stub")


# ── BACK-02: performance metrics ─────────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="Wave 0 stub — not yet implemented")
def test_performance_metrics():
    from scripts.run_backtest import compute_performance_metrics  # noqa: F401
    pytest.fail("stub")


# ── BACK-03: alpha/beta ───────────────────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="Wave 0 stub — not yet implemented")
def test_alpha_beta():
    from scripts.run_backtest import compute_performance_metrics  # noqa: F401
    pytest.fail("stub")
