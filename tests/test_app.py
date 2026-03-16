"""Tests for app.py — Phase 6 Interface.

All tests are xfail(strict=False) until app.py is implemented.
Imports inside test bodies prevent module-level ImportError when app.py does not exist yet.
"""
import pytest
import numpy as np
import pandas as pd


# ── UI-01: app.py exists and exports expected constants ───────────────────────

@pytest.mark.xfail(strict=False, reason="app.py not yet implemented")
def test_app_imports():
    """app.py must exist at project root and import without error."""
    import app  # noqa: F401


@pytest.mark.xfail(strict=False, reason="app.py not yet implemented")
def test_sidebar_defaults():
    """app.py must export DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_TOP_K."""
    import app
    assert app.DEFAULT_CONFIG_PATH == "config/Multitask_Stock_SP500.conf"
    assert app.DEFAULT_OUTPUT_DIR == "output/Multitask_output_SP500_2018-2024"
    assert app.DEFAULT_TOP_K == 10


# ── UI-02: equity curve chart ─────────────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="app.py not yet implemented")
def test_equity_chart_shape(backtest_daily_returns_fixture):
    """build_equity_chart(df) must return a go.Figure with exactly 2 traces."""
    import plotly.graph_objects as go
    from app import build_equity_chart
    df = backtest_daily_returns_fixture
    fig = build_equity_chart(df)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


@pytest.mark.xfail(strict=False, reason="app.py not yet implemented")
def test_equity_chart_starts_at_one(backtest_daily_returns_fixture):
    """Both equity curve traces must start at y=1.0."""
    from app import build_equity_chart
    df = backtest_daily_returns_fixture
    fig = build_equity_chart(df)
    # Both traces start at 1.0 (cumulative return from 1.0)
    assert abs(fig.data[0].y[0] - 1.0) < 1e-6, f"Portfolio trace starts at {fig.data[0].y[0]}, expected 1.0"
    assert abs(fig.data[1].y[0] - 1.0) < 1e-6, f"SPY trace starts at {fig.data[1].y[0]}, expected 1.0"


# ── UI-03: metrics table ──────────────────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="app.py not yet implemented")
def test_metrics_table_columns(backtest_summary_fixture):
    """format_metrics_table(df) must return a DataFrame with all 8 required columns."""
    from app import format_metrics_table
    result = format_metrics_table(backtest_summary_fixture)
    assert isinstance(result, pd.DataFrame)
    required_cols = {"annualized_return", "total_return", "max_drawdown",
                     "alpha_annualized", "sharpe_ratio", "beta", "top_k", "n_days"}
    assert required_cols.issubset(set(result.columns)), (
        f"Missing columns: {required_cols - set(result.columns)}"
    )


# ── UI-04: prediction heatmap ─────────────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="app.py not yet implemented")
def test_heatmap_zmid(regression_pred_fixture):
    """build_heatmap must return a go.Figure whose first trace has zmid=0."""
    import plotly.graph_objects as go
    from app import build_heatmap
    pred_df = regression_pred_fixture
    tickers = [f"TICK{i}" for i in range(10)]
    fig = build_heatmap(pred_df, tickers, k=5)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    heatmap_trace = fig.data[0]
    assert isinstance(heatmap_trace, go.Heatmap)
    assert heatmap_trace.zmid == 0


@pytest.mark.xfail(strict=False, reason="app.py not yet implemented")
def test_heatmap_top_k_filter(regression_pred_fixture):
    """build_heatmap with k=3 must produce a heatmap with exactly 3 y-axis rows."""
    from app import build_heatmap
    pred_df = regression_pred_fixture
    tickers = [f"TICK{i}" for i in range(10)]
    fig = build_heatmap(pred_df, tickers, k=3)
    heatmap_trace = fig.data[0]
    # z shape: (K, n_days) — y-axis has K entries
    assert len(heatmap_trace.y) == 3, (
        f"Expected 3 y-axis entries (top-3 tickers), got {len(heatmap_trace.y)}"
    )
