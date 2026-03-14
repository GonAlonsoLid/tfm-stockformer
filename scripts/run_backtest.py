#!/usr/bin/env python3
"""Complete backtest script for Stockformer portfolio simulation.

Usage:
    python scripts/run_backtest.py --output_dir output/Multitask_output_SP500_2018-2024 [--top_k 10] [--tickers_file data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt] [--config config/Multitask_Stock_SP500.conf]

Reads:
    output_dir/regression/regression_pred_last_step.csv  -- headerless (n_days × n_stocks)

Writes:
    output_dir/equity_curve.png           -- cumulative return chart (Stockformer vs SPY)
    output_dir/backtest_summary.csv       -- one row with performance statistics
    output_dir/backtest_daily_returns.csv -- daily return time series (date, portfolio, SPY)

Pure functions (select_top_k, build_portfolio_weights, compute_daily_return,
compute_performance_metrics) are defined here and imported by tests.
"""
import argparse
import configparser
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False


# ── Portfolio construction ────────────────────────────────────────────────────

def select_top_k(scores: pd.Series, k: int) -> pd.Index:
    """Return the k tickers with the highest predicted return scores.

    Args:
        scores: pd.Series indexed by ticker symbol, values are predicted returns.
        k:      Number of tickers to select.

    Returns:
        pd.Index of exactly k ticker symbols (the k highest-scoring ones).
        Ties are broken by the order in which pandas encounters them (nlargest
        default: first occurrence wins).
    """
    return scores.nlargest(k).index


def build_portfolio_weights(
    top_k_index: pd.Index,
    all_tickers: list,
    k: int,
) -> pd.Series:
    """Build equal-weight portfolio: 1/k for selected tickers, 0 for all others.

    Args:
        top_k_index: pd.Index of the k selected tickers (from select_top_k).
        all_tickers: list of all universe ticker symbols (determines index).
        k:           Number of selected tickers (used to compute 1/k weight).

    Returns:
        pd.Series indexed by all_tickers. Selected tickers = 1/k; rest = 0.0.
        Weights sum exactly to 1.0 when k tickers are selected.
    """
    weights = pd.Series(0.0, index=all_tickers)
    weights[top_k_index] = 1.0 / k
    return weights


def compute_daily_return(
    weight_now: pd.Series,
    weight_prev: pd.Series,
    price_returns: pd.Series,
    fee: float = 0.001,
) -> float:
    """Compute net daily portfolio return after transaction costs.

    Transaction cost = turnover * fee, where:
        turnover = (weight_now - weight_prev).abs().sum()

    Missing tickers in price_returns receive 0 return (handles delisted stocks).

    Args:
        weight_now:    Portfolio weights at end of current day (pd.Series by ticker).
        weight_prev:   Portfolio weights from prior day; zeros = buying from cash.
        price_returns: Realized price returns indexed by ticker symbol.
        fee:           Round-trip transaction cost rate (default 0.001 = 10 bps).

    Returns:
        Net daily return as a float (gross_return - transaction_cost).
    """
    # Reindex price returns to weight universe; missing tickers get 0 return
    aligned_returns = price_returns.reindex(weight_now.index).fillna(0.0)

    gross = float((weight_now * aligned_returns).sum())
    turnover = float((weight_now - weight_prev).abs().sum())
    cost = turnover * fee

    return gross - cost


# ── Performance measurement ───────────────────────────────────────────────────

def compute_performance_metrics(
    daily_returns,
    spy_daily_returns,
) -> dict:
    """Compute portfolio performance metrics against an SPY benchmark.

    All formulas match CONTEXT.md exactly:
        - Annualized return : (1 + total_return)^(252/n_days) - 1
        - Sharpe ratio      : (mean / std) * sqrt(252), rf=0, std uses ddof=1
        - Max drawdown      : (cum_returns / cum_returns.cummax() - 1).min()
        - Alpha (annualized): intercept of linregress(SPY, portfolio) * 252
        - Beta              : slope of linregress(SPY, portfolio)
        linregress convention: x=SPY (independent), y=portfolio (dependent)

    Args:
        daily_returns:     Daily portfolio returns. list or pd.Series of floats.
        spy_daily_returns: Daily SPY returns aligned to the same days.

    Returns:
        dict with keys:
            annualized_return  (float)
            sharpe_ratio       (float)
            max_drawdown       (float, <= 0)
            alpha_annualized   (float)
            beta               (float)
            total_return       (float)
            n_days             (int)
    """
    r = pd.Series(daily_returns, dtype=float)
    spy_r = np.asarray(spy_daily_returns, dtype=float)

    n_days = len(r)

    # Cumulative return series
    cum_returns = (1 + r).cumprod()

    # Total return
    total_return = float(cum_returns.iloc[-1] - 1)

    # Annualized return (geometric)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1

    # Sharpe ratio (rf = 0, sample std)
    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1))
    sharpe_ratio = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else float("nan")

    # Maximum drawdown
    max_drawdown = float((cum_returns / cum_returns.cummax() - 1).min())

    # Alpha and beta vs SPY (OLS: x=SPY, y=portfolio)
    r_arr = r.values
    slope, intercept, _, _, _ = linregress(spy_r, r_arr)
    beta = float(slope)
    alpha_annualized = float(intercept) * 252  # annualize daily alpha

    return {
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "alpha_annualized": alpha_annualized,
        "beta": beta,
        "total_return": total_return,
        "n_days": n_days,
    }


# ── CLI helpers ───────────────────────────────────────────────────────────────

def load_predictions(output_dir: str):
    """Load regression prediction CSV (headerless) from output_dir.

    Args:
        output_dir: Path to directory produced by run_inference.py.

    Returns:
        3-tuple: (reg_pred, n_days, n_stocks) where reg_pred is a float64
        ndarray of shape (n_days, n_stocks).

    Exits with code 1 if the file is not found.
    """
    pred_path = os.path.join(
        output_dir, "regression", "regression_pred_last_step.csv"
    )
    if not os.path.isfile(pred_path):
        print(f"ERROR: Prediction file not found: {pred_path}", file=sys.stderr)
        sys.exit(1)

    reg_pred = pd.read_csv(pred_path, header=None).values.astype(np.float64)
    n_days, n_stocks = reg_pred.shape
    return reg_pred, n_days, n_stocks


def derive_date_index(config_path: str, n_days: int, output_dir: str = None) -> pd.DatetimeIndex:
    """Derive the test-period trading day DatetimeIndex.

    Strategy (in order):
    1. Read [dataset] test_start / test_end from the config using configparser.
       Generate bdate_range and assert length matches n_days.
    2. If bdate_range count does not match (US market holidays not in bdate_range),
       fall back to reading the actual trading dates from label.csv in the data
       directory referenced by the config [file] traffic key.
    3. If label.csv not available, raise RuntimeError with clear message.

    Args:
        config_path: Path to the .conf file.
        n_days:      Expected number of trading days (must equal n_pred rows).
        output_dir:  Optional output directory path (unused, kept for signature).

    Returns:
        pd.DatetimeIndex of length n_days.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    # --- Strategy 1: read test_start / test_end directly from config ---
    test_start = None
    test_end = None
    for section in ("dataset", "backtest", "data"):
        if cfg.has_section(section):
            if cfg.has_option(section, "test_start") and cfg.has_option(section, "test_end"):
                test_start = cfg.get(section, "test_start").strip()
                test_end = cfg.get(section, "test_end").strip()
                break

    if test_start is not None and test_end is not None:
        date_index = pd.bdate_range(start=test_start, end=test_end)
        if len(date_index) == n_days:
            return date_index
        # Length mismatch — likely due to US market holidays; fall through to Strategy 2
        print(
            f"INFO: bdate_range({test_start}, {test_end}) = {len(date_index)} days, "
            f"expected {n_days}. Falling back to label.csv trading calendar.",
            file=sys.stderr,
        )

    # --- Strategy 2: derive dates from label.csv actual trading calendar ---
    # The config [file] traffic key points to the flow.npz file; label.csv is in
    # the same directory.
    label_path = None
    if cfg.has_section("file") and cfg.has_option("file", "traffic"):
        traffic_path = cfg.get("file", "traffic").strip()
        data_dir = os.path.dirname(traffic_path)
        # Resolve relative paths relative to the config file location
        config_dir = os.path.dirname(os.path.abspath(config_path))
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(config_dir, data_dir)
        label_path = os.path.join(data_dir, "label.csv")

    if label_path is None or not os.path.isfile(label_path):
        # Try to find label.csv from the config directory structure
        config_dir = os.path.dirname(os.path.abspath(config_path))
        project_root = os.path.dirname(config_dir)
        candidate = os.path.join(
            project_root,
            "data",
            "Stock_SP500_2018-01-01_2024-01-01",
            "label.csv",
        )
        if os.path.isfile(candidate):
            label_path = candidate

    if label_path is None or not os.path.isfile(label_path):
        raise RuntimeError(
            f"Cannot derive test-period date index: neither config test_start/test_end "
            f"produced a matching bdate_range, nor label.csv was found at '{label_path}'. "
            f"Expected {n_days} trading days."
        )

    # Read split_indices.json to find where the test period starts
    split_path = os.path.join(os.path.dirname(label_path), "split_indices.json")
    if not os.path.isfile(split_path):
        raise RuntimeError(
            f"split_indices.json not found at {split_path}. "
            "Cannot determine test split boundary."
        )

    import json
    with open(split_path) as f:
        splits = json.load(f)
    val_end = splits["val_end"]  # first index of test set

    label_df = pd.read_csv(label_path, usecols=["Date"])
    all_dates = pd.to_datetime(label_df["Date"])

    # The prediction matrix has n_days rows. They correspond to the last n_days
    # rows of the test split in label.csv.
    test_dates = all_dates.iloc[val_end:]
    if len(test_dates) < n_days:
        raise RuntimeError(
            f"label.csv has only {len(test_dates)} test rows after val_end={val_end}, "
            f"but {n_days} prediction rows expected."
        )

    # The predictions are for the final n_days of the test split
    date_index = pd.DatetimeIndex(test_dates.iloc[-n_days:].values)

    assert len(date_index) == n_days, (
        f"Date index length {len(date_index)} != prediction rows {n_days}"
    )
    return date_index


def download_prices(tickers: list, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Download adjusted close prices for all tickers plus SPY.

    Downloads with a 10-day buffer before the first test date so that
    pct_change() has a valid prior-day price on day 0.

    Args:
        tickers:    List of ticker symbols (universe stocks).
        date_index: DatetimeIndex of test-period trading days.

    Returns:
        pd.DataFrame indexed by date_index (reindexed, ffill then fillna(0)),
        with columns = tickers + ["SPY"].
    """
    if not _YF_AVAILABLE:
        raise ImportError(
            "yfinance is required to download prices. "
            "Install it with: pip install yfinance"
        )

    download_tickers = list(tickers) + ["SPY"]
    start = date_index[0] - pd.Timedelta(days=10)
    end = date_index[-1] + pd.Timedelta(days=5)

    raw = yf.download(
        download_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # Handle MultiIndex columns from batch yfinance download
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker — wrap in DataFrame
        prices = raw[["Close"]]
        prices.columns = download_tickers

    # Reindex to test-period date_index, forward-fill gaps, then zero-fill remaining NaN
    prices = prices.reindex(date_index, method="ffill").fillna(0.0)

    return prices


def run_backtest_loop(
    pred_df: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: list,
    top_k_n: int,
) -> tuple:
    """Run the daily backtest loop over the test period.

    For each day in pred_df.index:
      1. Select top-K tickers by predicted return score.
      2. Build equal-weight portfolio weights.
      3. Compute realized price returns for that day.
      4. Deduct transaction costs (10 bps × turnover).

    Args:
        pred_df:  DataFrame of shape (n_days, n_stocks) with DatetimeIndex.
        prices:   DataFrame of adjusted close prices including all tickers + SPY.
        tickers:  Ordered list of stock universe tickers.
        top_k_n:  K — number of stocks to select per day.

    Returns:
        (portfolio_returns, spy_daily_returns): two lists of floats, length n_days.
    """
    FEE = 0.001  # 10 bps round-trip

    portfolio_returns = []
    spy_daily_returns = []

    weight_prev = pd.Series(0.0, index=tickers)

    for date in pred_df.index:
        # Daily price returns via pct_change (requires prior-day price)
        # prices is reindexed to date_index; use the prices DataFrame directly
        # compute pct_change on the slice up to this date
        date_loc = prices.index.get_loc(date)
        if date_loc == 0:
            # No prior day in the reindexed slice — use zero return
            daily_price_ret = pd.Series(0.0, index=prices.columns)
        else:
            prev_date = prices.index[date_loc - 1]
            curr_close = prices.loc[date]
            prev_close = prices.loc[prev_date]
            # Avoid division by zero for zero-padded prices
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(
                    prev_close != 0,
                    (curr_close - prev_close) / prev_close,
                    0.0,
                )
            daily_price_ret = pd.Series(ratio, index=prices.columns).fillna(0.0)

        # SPY return for the day
        spy_ret = float(daily_price_ret.get("SPY", 0.0))
        spy_daily_returns.append(spy_ret)

        # Portfolio construction
        scores = pred_df.loc[date]
        top_k_index = select_top_k(scores, top_k_n)
        weight_now = build_portfolio_weights(top_k_index, tickers, top_k_n)

        # Daily portfolio return
        stock_returns = daily_price_ret.reindex(tickers).fillna(0.0)
        port_ret = compute_daily_return(weight_now, weight_prev, stock_returns, fee=FEE)
        portfolio_returns.append(port_ret)

        weight_prev = weight_now

    return portfolio_returns, spy_daily_returns


def save_outputs(
    output_dir: str,
    date_index: pd.DatetimeIndex,
    portfolio_returns: list,
    spy_daily_returns: list,
    metrics: dict,
    top_k_n: int,
) -> None:
    """Save equity_curve.png, backtest_summary.csv, backtest_daily_returns.csv.

    Chart uses RESEARCH.md Pattern 5 exactly:
      - Portfolio line: red #CC2529, solid
      - SPY line: dark grey #333333, dashed
      - Both series start at 1.0 using (1+pd.Series(returns)).cumprod()
      - No plt.show(); fig.savefig() then plt.close(fig)

    Args:
        output_dir:       Directory where output files are written.
        date_index:       DatetimeIndex of test-period trading days.
        portfolio_returns: List of daily net portfolio returns.
        spy_daily_returns: List of daily SPY returns aligned to date_index.
        metrics:          Dict from compute_performance_metrics plus top_k and n_days.
        top_k_n:          K value (used for chart label).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Cumulative return series — both start at 1.0
    portfolio_cum = (1 + pd.Series(portfolio_returns)).cumprod()
    spy_cum = (1 + pd.Series(spy_daily_returns)).cumprod()

    # --- equity_curve.png ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        date_index,
        portfolio_cum.values,
        label=f"Stockformer Top-{top_k_n}",
        color="#CC2529",
        linewidth=1.5,
    )
    ax.plot(
        date_index,
        spy_cum.values,
        label="SPY",
        color="#333333",
        linewidth=1.5,
        linestyle="--",
    )
    ax.set_title("Portfolio vs SPY — Test Period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "equity_curve.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)  # no plt.show()

    # --- backtest_summary.csv (one row, eight columns) ---
    summary = {
        "annualized_return": metrics["annualized_return"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "alpha_annualized": metrics["alpha_annualized"],
        "beta": metrics["beta"],
        "top_k": top_k_n,
        "n_days": metrics["n_days"],
        "total_return": metrics["total_return"],
    }
    summary_path = os.path.join(output_dir, "backtest_summary.csv")
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    # --- backtest_daily_returns.csv (date / portfolio_return / spy_return) ---
    daily_df = pd.DataFrame({
        "date": [str(d.date()) for d in date_index],
        "portfolio_return": portfolio_returns,
        "spy_return": spy_daily_returns,
    })
    daily_path = os.path.join(output_dir, "backtest_daily_returns.csv")
    daily_df.to_csv(daily_path, index=False)

    print(f"\nOutputs saved to: {output_dir}")
    print(f"  equity_curve.png")
    print(f"  backtest_summary.csv")
    print(f"  backtest_daily_returns.csv")


def main(output_dir: str = None, top_k: int = None) -> None:
    """Run full end-to-end backtest and write output files.

    Can be called programmatically (output_dir, top_k as kwargs) or via CLI.
    When called programmatically, argparse is skipped (same pattern as compute_ic.py).

    Args:
        output_dir: Path to directory with regression/ subdirectory. If None,
                    parsed from --output_dir CLI argument.
        top_k:      K top stocks to hold per day. If None and called via CLI,
                    parsed from --top_k argument (default 10).
    """
    if output_dir is None:
        parser = argparse.ArgumentParser(
            description="Run Stockformer portfolio backtest against SPY benchmark."
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            required=True,
            help="Directory produced by run_inference.py (contains regression/)",
        )
        parser.add_argument(
            "--top_k",
            type=int,
            default=10,
            help="Number of stocks to hold per day (default: 10)",
        )
        parser.add_argument(
            "--tickers_file",
            type=str,
            default="data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt",
            help="Path to tickers file (one ticker per line, column-order matches CSV)",
        )
        parser.add_argument(
            "--config",
            type=str,
            default="config/Multitask_Stock_SP500.conf",
            help="Path to Stockformer config file (used to derive test-period dates)",
        )
        args = parser.parse_args()
        output_dir = args.output_dir
        top_k_n = args.top_k
        tickers_file = args.tickers_file
        config_path = args.config
    else:
        top_k_n = top_k if top_k is not None else 10
        tickers_file = "data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt"
        config_path = "config/Multitask_Stock_SP500.conf"

    if not os.path.isdir(output_dir):
        print(f"ERROR: output_dir does not exist: {output_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Load prediction matrix
    reg_pred, n_days, n_stocks = load_predictions(output_dir)
    print(f"Loaded predictions: {n_days} days × {n_stocks} stocks")

    # 2. Load tickers
    if not os.path.isfile(tickers_file):
        print(f"ERROR: Tickers file not found: {tickers_file}", file=sys.stderr)
        sys.exit(1)
    with open(tickers_file) as f:
        tickers = [t.strip() for t in f.readlines() if t.strip()]
    if len(tickers) != n_stocks:
        print(
            f"ERROR: Tickers file has {len(tickers)} entries but prediction has {n_stocks} columns.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Derive test-period date index
    if not os.path.isfile(config_path):
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    date_index = derive_date_index(config_path, n_days, output_dir)
    print(f"Test period: {date_index[0].date()} → {date_index[-1].date()} ({len(date_index)} days)")

    # 4. Build pred_df (n_days × n_stocks DataFrame with DatetimeIndex)
    pred_df = pd.DataFrame(reg_pred, index=date_index, columns=tickers)

    # 5. Download adjusted close prices (tickers + SPY)
    print(f"Downloading prices for {n_stocks} tickers + SPY via yfinance...")
    prices = download_prices(tickers, date_index)

    # 6. Run backtest loop
    print(f"Running backtest (top_k={top_k_n}, fee=0.001)...")
    portfolio_returns, spy_daily_returns = run_backtest_loop(
        pred_df, prices, tickers, top_k_n
    )

    # 7. Compute performance metrics
    metrics = compute_performance_metrics(portfolio_returns, spy_daily_returns)

    # 8. Save outputs
    save_outputs(
        output_dir, date_index, portfolio_returns, spy_daily_returns, metrics, top_k_n
    )

    # 9. Console summary block (same style as compute_ic.py)
    print("\n=== Backtest Summary ===")
    print(f"  Top-K               : {top_k_n}")
    print(f"  Test days           : {metrics['n_days']}")
    print(f"  Total return        : {metrics['total_return']:+.4f} ({metrics['total_return']*100:+.2f}%)")
    print(f"  Annualized return   : {metrics['annualized_return']:+.4f} ({metrics['annualized_return']*100:+.2f}%)")
    print(f"  Sharpe ratio        : {metrics['sharpe_ratio']:+.4f}")
    print(f"  Max drawdown        : {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']*100:.2f}%)")
    print(f"  Alpha (annualized)  : {metrics['alpha_annualized']:+.4f} ({metrics['alpha_annualized']*100:+.2f}%)")
    print(f"  Beta vs SPY         : {metrics['beta']:.4f}")


if __name__ == "__main__":
    main()
