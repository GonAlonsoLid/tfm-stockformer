#!/usr/bin/env python3
"""Pure-function backtest core for Stockformer portfolio simulation.

Usage (imported by run_backtest_cli.py in Plan 03):
    from scripts.run_backtest import (
        select_top_k,
        build_portfolio_weights,
        compute_daily_return,
        compute_performance_metrics,
    )

No CLI, no yfinance calls — those belong to Plan 03 (run_backtest_cli.py).

Functions:
    select_top_k            — top-K ticker selection by predicted return score
    build_portfolio_weights — equal-weight 1/k allocation over selected tickers
    compute_daily_return    — gross return minus turnover transaction cost
    compute_performance_metrics — annualized return, Sharpe, max drawdown, alpha, beta
"""
import math

import numpy as np
import pandas as pd
from scipy.stats import linregress


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
