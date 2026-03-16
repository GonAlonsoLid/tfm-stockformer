"""Stockformer S&P500 Dashboard — Streamlit app.

Entry point: streamlit run app.py

Architecture:
- Sidebar contains all controls (Pipeline Settings + Backtest Parameters)
- Main area shows placeholder or results depending on st.session_state
- Pure functions (build_equity_chart, format_metrics_table, build_heatmap) are
  extracted to be independently testable without Streamlit context.
- Pipeline execution uses subprocess.Popen for live log streaming (not import).
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Module-level constants (exported for tests) ───────────────────────────────
DEFAULT_CONFIG_PATH = "config/Multitask_Stock_SP500.conf"
DEFAULT_OUTPUT_DIR = "output/Multitask_output_SP500_2018-2024"
DEFAULT_TOP_K = 10
TICKERS_FILE = "data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt"


# ── Pure chart / table functions (testable, no Streamlit calls) ──────────────

def build_equity_chart(df: pd.DataFrame) -> go.Figure:
    """Build Plotly equity curve from backtest_daily_returns DataFrame.

    Args:
        df: DataFrame with columns date (str or datetime), portfolio_return (float),
            spy_return (float). Rows are trading days in chronological order.
            The caller is responsible for pre-filtering df to the desired date window.

    Returns:
        go.Figure with 2 go.Scatter traces. Both start at 1.0 (cumulative return).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Shift cumprod by 1 so the first plotted point is the starting value 1.0.
    # This means: day 0 → 1.0, day 1 → (1+r[0]), day 2 → (1+r[0])*(1+r[1]), ...
    portfolio_cum = (1 + df["portfolio_return"]).cumprod().shift(1, fill_value=1.0)
    spy_cum = (1 + df["spy_return"]).cumprod().shift(1, fill_value=1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=portfolio_cum,
        name="Portfolio",
        line=dict(color="#4C9BE8", width=2),
        mode="lines",
        hovertemplate="%{x|%Y-%m-%d}: Portfolio %{y:.3f}x<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["date"],
        y=spy_cum,
        name="SPY",
        line=dict(color="#9CA3AF", width=1.5, dash="dot"),
        mode="lines",
        hovertemplate="%{x|%Y-%m-%d}: SPY %{y:.3f}x<extra></extra>",
    ))
    fig.update_layout(
        title="Cumulative Return: Portfolio vs SPY",
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(x=0.01, y=0.99),
        xaxis=dict(
            tickformat="%b %Y",
            gridcolor="#374151",
            tickfont=dict(size=12, color="#9CA3AF"),
        ),
        yaxis=dict(
            ticksuffix="x",
            gridcolor="#374151",
            tickfont=dict(size=12, color="#9CA3AF"),
        ),
    )
    return fig


def format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare backtest_summary DataFrame for display.

    Args:
        df: One-row DataFrame from backtest_summary.csv. Expected columns:
            annualized_return, total_return, max_drawdown, alpha_annualized,
            sharpe_ratio, beta, top_k, n_days.

    Returns:
        DataFrame with the same columns (numeric types preserved — do NOT
        call fillna("—") here as that breaks NumberColumn formatting).
    """
    required = [
        "annualized_return",
        "total_return",
        "max_drawdown",
        "alpha_annualized",
        "sharpe_ratio",
        "beta",
        "top_k",
        "n_days",
    ]
    # Return only the required columns in required order; fill missing with NaN
    result = df.reindex(columns=required)
    return result


def build_heatmap(pred_df: pd.DataFrame, tickers: list, k: int) -> go.Figure:
    """Build Plotly heatmap of predicted scores for portfolio tickers.

    LOCKED DECISION: tickers is the pre-filtered list of portfolio holdings
    (same K as portfolio, sourced from backtest_positions.csv). The function
    slices the list to the first k entries so callers may pass a longer list
    with an explicit k to select how many rows appear on the y-axis.

    Args:
        pred_df: DataFrame of shape (n_days, n_stocks). No header (headerless CSV).
                 Columns correspond to the full ticker universe by position index.
        tickers: Ordered list of portfolio ticker symbols. Already filtered to
                 portfolio holdings — used as y-axis labels. The first k entries
                 are used.
        k:       Number of tickers to display. Controls y-axis length and chart title.

    Returns:
        go.Figure with one go.Heatmap trace. z shape: (k, n_days).
        zmid=0 (colorscale centered at zero). y-axis has exactly k entries.
    """
    # Use only the first k tickers for the y-axis
    display_tickers = tickers[:k]

    pred_arr = pred_df.values  # shape (n_days, n_stocks)

    # Select the first k columns — matches the k selected portfolio tickers
    filtered = pred_arr[:, :k]  # shape (n_days, k)
    z = filtered.T  # shape (k, n_days) — stocks on y-axis

    # Symmetric color range centered at 0
    max_abs = float(np.nanmax(np.abs(z))) if z.size > 0 else 1.0
    if max_abs == 0:
        max_abs = 1.0

    # X-axis: day indices (0..n_days-1) — actual dates injected by caller if available
    x_vals = list(range(pred_arr.shape[0]))

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_vals,
        y=display_tickers,
        colorscale=[[0, "#DC2626"], [0.5, "#F9FAFB"], [1, "#16A34A"]],
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(title="Predicted Score"),
        hovertemplate="%{y} day %{x}: %{z:.4f}<extra></extra>",
    ))
    height = max(300, min(k * 24, 600))
    fig.update_layout(
        title=f"Prediction Scores — Top-{k} Portfolio Stocks",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=12, color="#9CA3AF")),
        yaxis=dict(tickfont=dict(size=12, color="#9CA3AF")),
    )
    return fig


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(config_path: str, output_dir: str, top_k: int, log_placeholder) -> bool:
    """Run run_inference.py then run_backtest.py as subprocesses with live log streaming.

    Args:
        config_path:     Path to .conf file (passed to run_inference.py --config).
        output_dir:      Output directory (passed to run_backtest.py --output_dir).
        top_k:           Portfolio size K (passed to run_backtest.py --top_k).
        log_placeholder: st.empty() container for live log updates.

    Returns:
        True if both subprocesses exit with code 0, False otherwise.
    """
    accumulated: list[str] = []
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def stream(cmd: list[str]) -> int:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in iter(proc.stdout.readline, ""):
            accumulated.append(line.rstrip())
            log_placeholder.code("\n".join(accumulated[-60:]))
        proc.wait()
        return proc.returncode

    rc1 = stream([sys.executable, "scripts/run_inference.py", "--config", config_path])
    if rc1 != 0:
        return False
    rc2 = stream([
        sys.executable, "scripts/run_backtest.py",
        "--output_dir", output_dir,
        "--top_k", str(top_k),
    ])
    return rc2 == 0


# ── Result loading ────────────────────────────────────────────────────────────

def load_results(
    output_dir: str,
    top_k: int,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict | None:
    """Load all result CSVs from output_dir after a successful pipeline run.

    Args:
        output_dir:  Directory containing pipeline outputs.
        top_k:       Portfolio size K chosen by the user.
        start_date:  Optional lower bound for the equity curve display window.
        end_date:    Optional upper bound for the equity curve display window.
                     Both bounds are inclusive. If None, the full date range
                     from backtest_daily_returns.csv is used.

    Returns a dict with keys: daily_returns_df, summary_df, pred_df,
    tickers (full universe), portfolio_tickers (filtered to portfolio holdings),
    start_date, end_date.
    Returns None if any required file is missing.
    """
    daily_path = os.path.join(output_dir, "backtest_daily_returns.csv")
    summary_path = os.path.join(output_dir, "backtest_summary.csv")
    pred_path = os.path.join(output_dir, "regression", "regression_pred_last_step.csv")

    missing = [p for p in [daily_path, summary_path, pred_path] if not os.path.isfile(p)]
    if missing:
        return None

    daily_df = pd.read_csv(daily_path, parse_dates=["date"])
    summary_df = pd.read_csv(summary_path)
    pred_df = pd.read_csv(pred_path, header=None)

    # Load full ticker universe from tickers.txt (column order matches pred_df columns)
    tickers: list[str] = []
    if os.path.isfile(TICKERS_FILE):
        with open(TICKERS_FILE) as f:
            tickers = [t.strip() for t in f if t.strip()]

    # Extract portfolio tickers from backtest_positions.csv (locked decision:
    # heatmap shows only stocks actually held in the portfolio, not top-K by score).
    portfolio_tickers: list[str] = []
    positions_path = os.path.join(output_dir, "backtest_positions.csv")
    if os.path.isfile(positions_path):
        positions_df = pd.read_csv(positions_path)
        # Preserve order of first appearance; drop duplicates across dates
        portfolio_tickers = list(dict.fromkeys(
            positions_df["ticker"].dropna().tolist()
        ))

    return {
        "daily_returns_df": daily_df,
        "summary_df": summary_df,
        "pred_df": pred_df,
        "tickers": tickers,
        "portfolio_tickers": portfolio_tickers,
        "output_dir": output_dir,
        "top_k": top_k,
        "start_date": start_date,
        "end_date": end_date,
    }


# ── Result rendering ──────────────────────────────────────────────────────────

def render_results(results: dict) -> None:
    """Render equity curve, metrics table, and heatmap in the main area."""
    daily_df = results["daily_returns_df"]
    summary_df = results["summary_df"]
    pred_df = results["pred_df"]
    portfolio_tickers = results["portfolio_tickers"]
    top_k = results["top_k"]
    start_date = results.get("start_date")
    end_date = results.get("end_date")

    # ── Equity curve ──────────────────────────────────────────────────────
    st.subheader("Equity Curve")
    if not daily_df.empty:
        # Filter to selected date window (locked decision: date range controls
        # the evaluation window displayed in the equity curve chart).
        display_df = daily_df.copy()
        if start_date is not None:
            display_df = display_df[
                display_df["date"] >= pd.Timestamp(start_date)
            ]
        if end_date is not None:
            display_df = display_df[
                display_df["date"] <= pd.Timestamp(end_date)
            ]
        if display_df.empty:
            st.warning(
                "No data in selected date range. "
                "Adjust the start/end date in the sidebar."
            )
        else:
            fig_equity = build_equity_chart(display_df)
            st.plotly_chart(fig_equity, use_container_width=True)
    else:
        st.warning("No daily returns data found.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics table ─────────────────────────────────────────────────────
    st.subheader("Performance Metrics")
    if not summary_df.empty:
        display_metrics = format_metrics_table(summary_df)
        st.dataframe(
            display_metrics,
            use_container_width=True,
            column_config={
                "annualized_return": st.column_config.NumberColumn(
                    "Ann. Return", format="%.2f %%"
                ),
                "total_return": st.column_config.NumberColumn(
                    "Total Return", format="%.2f %%"
                ),
                "max_drawdown": st.column_config.NumberColumn(
                    "Max Drawdown", format="%.2f %%"
                ),
                "alpha_annualized": st.column_config.NumberColumn(
                    "Alpha (Ann.)", format="%.2f %%"
                ),
                "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
                "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
                "top_k": st.column_config.NumberColumn("Top-K", format="%d"),
                "n_days": st.column_config.NumberColumn("Days", format="%d"),
            },
        )
    else:
        st.info("No metrics found. Run the pipeline first.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Prediction heatmap ────────────────────────────────────────────────
    st.subheader("Prediction Heatmap")
    if not pred_df.empty and len(portfolio_tickers) > 0:
        k = min(top_k, len(portfolio_tickers))
        display_tickers = portfolio_tickers[:k]
        fig_heatmap = build_heatmap(pred_df, display_tickers, k=k)

        # Inject actual dates into x-axis if daily_df has date column
        if not daily_df.empty and "date" in daily_df.columns:
            n_pred = pred_df.shape[0]
            n_dates = len(daily_df)
            if n_pred == n_dates:
                date_strs = daily_df["date"].dt.strftime("%Y-%m-%d").tolist()
                fig_heatmap.data[0].x = date_strs

        st.plotly_chart(fig_heatmap, use_container_width=True)
    elif pred_df.empty:
        st.info(
            "No prediction data found. "
            "Ensure regression_pred_last_step.csv exists in the output directory."
        )
    else:
        st.warning(
            "No portfolio positions found. "
            "Ensure backtest_positions.csv exists in the output directory."
        )


# ── Streamlit app layout ──────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Stockformer Dashboard", layout="wide")
    st.title("Stockformer S&P500 Dashboard")

    # Session state initialisation
    for key, default in [
        ("run_complete", False),
        ("running", False),
        ("results", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("**Pipeline Settings**")
        config_path = st.text_input("Config path", DEFAULT_CONFIG_PATH)
        output_dir = st.text_input("Output directory", DEFAULT_OUTPUT_DIR)

        st.markdown("---")

        st.markdown("**Backtest Parameters**")
        top_k = st.number_input(
            "Top-K", min_value=1, max_value=100, step=1, value=DEFAULT_TOP_K
        )
        _txn_cost = st.number_input(
            "Transaction cost (bps)", min_value=0, max_value=100, step=1, value=10
        )
        _benchmark = st.text_input("Benchmark ticker", "SPY")
        start_date = st.date_input("Start date", value=date(2023, 1, 1))
        end_date = st.date_input("End date", value=date(2024, 12, 31))

        st.markdown("---")

        run_clicked = st.button(
            "Run Pipeline",
            type="primary",
            disabled=st.session_state["running"],
        )

        # Download button appears only after successful run
        if st.session_state["run_complete"]:
            summary_path = os.path.join(output_dir, "backtest_summary.csv")
            if os.path.isfile(summary_path):
                with open(summary_path, "rb") as f:
                    st.download_button(
                        label="Download backtest_summary.csv",
                        data=f,
                        file_name="backtest_summary.csv",
                        mime="text/csv",
                    )

    # ── Main area ─────────────────────────────────────────────────────────
    if not st.session_state["run_complete"] and not st.session_state["running"]:
        st.info(
            "**Configure settings to run your first analysis**\n\n"
            "Configure your settings in the sidebar, then click Run Pipeline "
            "to run inference and backtesting."
        )

    if run_clicked:
        st.session_state["running"] = True
        st.session_state["run_complete"] = False
        st.session_state["results"] = None

        log_placeholder = st.empty()
        with st.spinner("Running inference and backtest — this may take several minutes..."):
            success = run_pipeline(config_path, output_dir, top_k, log_placeholder)

        st.session_state["running"] = False

        if success:
            results = load_results(
                output_dir, top_k,
                start_date=start_date,
                end_date=end_date,
            )
            if results is not None:
                st.session_state["run_complete"] = True
                st.session_state["results"] = results
            else:
                st.error(
                    f"Output files not found in {output_dir}. "
                    "Verify the output directory path is correct."
                )
        else:
            st.error("Pipeline failed. Check the log above for details.")

    if st.session_state["run_complete"] and st.session_state["results"] is not None:
        render_results(st.session_state["results"])


# ── Entry point guard ─────────────────────────────────────────────────────────
# Use get_script_run_ctx() to detect whether Streamlit's runtime is active.
# When `streamlit run app.py` is used, the ctx is non-None — call main().
# When `import app` is used in tests, the ctx is None — do not call main().
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
    if _get_ctx() is not None:
        main()
except Exception:
    pass
