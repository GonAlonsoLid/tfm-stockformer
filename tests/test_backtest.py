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

def test_cumulative_return():
    from scripts.run_backtest import compute_performance_metrics

    # Known daily returns
    daily_returns = [0.01, -0.005, 0.02]
    spy_returns = [0.005, -0.003, 0.01]

    metrics = compute_performance_metrics(daily_returns, spy_returns)

    # Reconstruct expected cumulative series
    cum = pd.Series(daily_returns).add(1).cumprod()
    assert abs(cum.iloc[0] - 1.01) < 1e-9, f"First cum value should be 1.01, got {cum.iloc[0]}"
    # Last value: 1.01 * (1 - 0.005) * (1 + 0.02) = 1.01 * 0.995 * 1.02
    expected_last = 1.01 * 0.995 * 1.02
    assert abs(cum.iloc[-1] - expected_last) < 1e-9, (
        f"Last cum value should be ~{expected_last:.6f}, got {cum.iloc[-1]:.6f}"
    )
    assert len(cum) == 3, "Cumulative series must have same length as input"

    # Also verify metrics dict has required keys
    required_keys = {"annualized_return", "sharpe_ratio", "max_drawdown",
                     "alpha_annualized", "beta", "total_return", "n_days"}
    assert required_keys.issubset(metrics.keys()), (
        f"Missing keys: {required_keys - metrics.keys()}"
    )
    assert metrics["n_days"] == 3


# ── BACK-02: performance metrics ─────────────────────────────────────────────

def test_performance_metrics():
    from scripts.run_backtest import compute_performance_metrics

    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.01, 252)
    spy_daily_returns = np.random.normal(0.0005, 0.008, 252)

    metrics = compute_performance_metrics(daily_returns, spy_daily_returns)

    # Required keys
    required_keys = {"annualized_return", "sharpe_ratio", "max_drawdown",
                     "alpha_annualized", "beta"}
    assert required_keys.issubset(metrics.keys()), (
        f"Missing keys: {required_keys - metrics.keys()}"
    )

    # max_drawdown must be non-positive
    assert metrics["max_drawdown"] <= 0, (
        f"max_drawdown should be <= 0, got {metrics['max_drawdown']}"
    )
    # sharpe_ratio must be finite
    assert np.isfinite(metrics["sharpe_ratio"]), (
        f"sharpe_ratio must be finite, got {metrics['sharpe_ratio']}"
    )
    # beta must be finite
    assert np.isfinite(metrics["beta"]), (
        f"beta must be finite, got {metrics['beta']}"
    )
    # n_days = 252
    assert metrics["n_days"] == 252


# ── BACK-03: alpha/beta ───────────────────────────────────────────────────────

def test_alpha_beta():
    from scripts.run_backtest import compute_performance_metrics

    np.random.seed(99)
    n = 252
    spy_daily = np.random.normal(0.0005, 0.01, n)
    alpha_daily = 0.0002  # known daily alpha
    # Portfolio perfectly correlated with SPY plus constant alpha (beta=1)
    portfolio_daily = spy_daily + alpha_daily

    metrics = compute_performance_metrics(portfolio_daily, spy_daily)

    # Beta should be ~1.0 (slope of perfect linear relationship)
    assert abs(metrics["beta"] - 1.0) < 1e-6, (
        f"Beta should be ~1.0, got {metrics['beta']}"
    )
    # Alpha annualized should be ~0.0002 * 252 within 1e-6 tolerance
    expected_alpha = alpha_daily * 252
    assert abs(metrics["alpha_annualized"] - expected_alpha) < 1e-6, (
        f"alpha_annualized should be ~{expected_alpha:.6f}, "
        f"got {metrics['alpha_annualized']:.6f}"
    )


# ── Positions output ─────────────────────────────────────────────────────────

def test_positions_output():
    """run_backtest_loop returns positions list with one row per (day, ticker)."""
    import numpy as np
    import pandas as pd
    from scripts.run_backtest import run_backtest_loop

    tickers = ["A", "B", "C", "D", "E"]
    dates = pd.bdate_range("2023-01-03", periods=3)

    np.random.seed(7)
    pred_df = pd.DataFrame(
        np.random.rand(3, 5), index=dates, columns=tickers
    )
    price_cols = tickers + ["SPY"]
    prices = pd.DataFrame(
        np.ones((3, 6)) * 100 + np.arange(3)[:, None],
        index=dates,
        columns=price_cols,
    )

    _, _, positions, _ = run_backtest_loop(pred_df, prices, tickers, top_k_n=2)
    pos_df = pd.DataFrame(positions)

    assert len(pos_df) == 6, f"Expected 6 rows, got {len(pos_df)}"
    assert {"date", "ticker", "weight", "predicted_score"}.issubset(set(pos_df.columns))
    assert (pos_df["weight"] == 0.5).all(), "All weights must equal 0.5 for top_k=2"
    assert pos_df.notna().all().all(), "No null values expected"
    for d, grp in pos_df.groupby("date"):
        assert abs(grp["weight"].sum() - 1.0) < 1e-9, f"Weights don't sum to 1.0 on {d}"
