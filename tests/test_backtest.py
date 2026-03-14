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


@pytest.mark.xfail(strict=False, reason="Wave 0 stub — not yet implemented")
def test_top_k_selection():
    from scripts.run_backtest import select_top_k  # noqa: F401
    pytest.fail("stub")


@pytest.mark.xfail(strict=False, reason="Wave 0 stub — not yet implemented")
def test_equal_weight():
    from scripts.run_backtest import build_portfolio_weights  # noqa: F401
    pytest.fail("stub")


@pytest.mark.xfail(strict=False, reason="Wave 0 stub — not yet implemented")
def test_transaction_cost():
    from scripts.run_backtest import compute_daily_return  # noqa: F401
    pytest.fail("stub")


@pytest.mark.xfail(strict=False, reason="Wave 0 stub — not yet implemented")
def test_cumulative_return():
    from scripts.run_backtest import compute_performance_metrics  # noqa: F401
    pytest.fail("stub")


@pytest.mark.xfail(strict=False, reason="Wave 0 stub — not yet implemented")
def test_performance_metrics():
    from scripts.run_backtest import compute_performance_metrics  # noqa: F401
    pytest.fail("stub")


@pytest.mark.xfail(strict=False, reason="Wave 0 stub — not yet implemented")
def test_alpha_beta():
    from scripts.run_backtest import compute_performance_metrics  # noqa: F401
    pytest.fail("stub")
