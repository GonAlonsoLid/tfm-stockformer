"""Unified evaluation harness for the model-complexity ladder (TFM).

Single measurement instrument shared by every model — linear, trees, MLP,
StockMixer, Stockformer — so the comparison is homogeneous. Given two aligned
panels it returns the full metric set used in the thesis:

    - Cross-sectional Rank IC (Spearman) per day, ICIR, t-stat, % days IC>0
    - Bootstrap confidence interval on the mean IC
    - Paired test between two models' daily IC series
    - Dollar-neutral long-short decile portfolio (offline, on realized labels)
      with turnover and net-of-cost returns
    - Performance metrics: annualized return, Sharpe, max drawdown, alpha/beta

Convention (used everywhere):
    pred  : DataFrame [dates x tickers] — predicted score, higher = more bullish
    label : DataFrame [dates x tickers] — realized forward return

The long-short backtest is *factor-style*: it uses ``label`` itself as the
realized return of each stock, so it is fully reproducible and needs no price
download. ``run_backtest.py`` remains the tradable yfinance-based backtest.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS = 252


# ── IC computation ────────────────────────────────────────────────────────────

def daily_rank_ic(pred: pd.DataFrame, label: pd.DataFrame) -> pd.Series:
    """Cross-sectional Spearman rank IC for each date.

    Days with fewer than two valid pairs, or constant predictions/labels
    (no rank information), yield NaN.
    """
    pred, label = _align(pred, label)
    ics = {}
    for date in pred.index:
        p = pred.loc[date].to_numpy(dtype=float)
        l = label.loc[date].to_numpy(dtype=float)
        mask = ~(np.isnan(p) | np.isnan(l))
        if mask.sum() < 2 or np.std(p[mask]) == 0 or np.std(l[mask]) == 0:
            # constant input has no rank information -> IC undefined
            ics[date] = np.nan
            continue
        rho = stats.spearmanr(p[mask], l[mask]).correlation
        ics[date] = rho
    return pd.Series(ics, name="rank_ic")


def daily_pearson_ic(pred: pd.DataFrame, label: pd.DataFrame) -> pd.Series:
    """Cross-sectional Pearson IC for each date."""
    pred, label = _align(pred, label)
    ics = {}
    for date in pred.index:
        p = pred.loc[date].to_numpy(dtype=float)
        l = label.loc[date].to_numpy(dtype=float)
        mask = ~(np.isnan(p) | np.isnan(l))
        if mask.sum() < 2 or np.std(p[mask]) == 0 or np.std(l[mask]) == 0:
            ics[date] = np.nan
            continue
        ics[date] = float(np.corrcoef(p[mask], l[mask])[0, 1])
    return pd.Series(ics, name="pearson_ic")


def ic_stats(daily_ic: pd.Series) -> dict:
    """Summary statistics of a daily IC series (NaN days excluded).

    Returns ic_mean, ic_std (ddof=1), icir, tstat, pvalue, pct_positive, n.
    The t-stat tests H0: mean IC = 0 (one-sample t-test over trading days).
    """
    ic = pd.Series(daily_ic).dropna()
    n = int(len(ic))
    if n == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "icir": float("nan"), "tstat": float("nan"),
                "pvalue": float("nan"), "pct_positive": float("nan"), "n": 0}
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    icir = mean / std if std and std > 0 else float("nan")
    if n > 1 and std > 0:
        tstat, pvalue = stats.ttest_1samp(ic.to_numpy(), 0.0)
        tstat, pvalue = float(tstat), float(pvalue)
    else:
        tstat, pvalue = float("nan"), float("nan")
    pct_positive = float((ic > 0).mean() * 100)
    return {"ic_mean": mean, "ic_std": std, "icir": icir, "tstat": tstat,
            "pvalue": pvalue, "pct_positive": pct_positive, "n": n}


def bootstrap_ic_ci(daily_ic: pd.Series, n_boot: int = 1000,
                    ci: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """Bootstrap confidence interval for the mean daily IC.

    Resamples trading days with replacement. Deterministic for a fixed seed.
    """
    ic = pd.Series(daily_ic).dropna().to_numpy()
    if len(ic) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    n = len(ic)
    for b in range(n_boot):
        means[b] = ic[rng.integers(0, n, size=n)].mean()
    alpha = (1 - ci) / 2
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1 - alpha))
    return (lo, hi)


def paired_ic_ttest(ic_a: pd.Series, ic_b: pd.Series) -> dict:
    """Paired t-test on daily IC differences (model A minus model B).

    Aligns on common dates and drops days where either is NaN.
    """
    df = pd.concat([pd.Series(ic_a), pd.Series(ic_b)], axis=1, join="inner").dropna()
    diff = (df.iloc[:, 0] - df.iloc[:, 1]).to_numpy()
    if len(diff) < 2:
        return {"mean_diff": float("nan"), "tstat": float("nan"),
                "pvalue": float("nan"), "n": int(len(diff))}
    tstat, pvalue = stats.ttest_rel(df.iloc[:, 0].to_numpy(), df.iloc[:, 1].to_numpy())
    return {"mean_diff": float(diff.mean()), "tstat": float(tstat),
            "pvalue": float(pvalue), "n": int(len(diff))}


# ── Long-short factor backtest (offline, on realized labels) ───────────────────

def longshort_returns(pred: pd.DataFrame, label: pd.DataFrame,
                      quantile: float = 0.1, fee: float = 0.001) -> pd.DataFrame:
    """Dollar-neutral long-short decile portfolio returns from realized labels.

    Each day: rank stocks by ``pred``; long the top ``quantile`` (equal weight,
    summing to +1), short the bottom ``quantile`` (summing to -1). Realized
    return uses ``label``. Net return subtracts ``turnover * fee``.

    Returns a DataFrame indexed by date with columns gross, net, turnover.
    """
    pred, label = _align(pred, label)
    tickers = list(pred.columns)
    n_stocks = len(tickers)
    k = max(1, int(n_stocks * quantile))

    rows = {}
    w_prev = pd.Series(0.0, index=tickers)
    for date in pred.index:
        scores = pred.loc[date]
        rets = label.loc[date].reindex(tickers).fillna(0.0)
        w = pd.Series(0.0, index=tickers)
        valid = scores.dropna()
        # need enough names AND real dispersion (constant scores carry no rank
        # information -> a no-skill predictor must take no position)
        if len(valid) >= 2 * k and valid.nunique() >= 2:
            longs = valid.nlargest(k).index
            shorts = valid.nsmallest(k).index
            w[longs] = 1.0 / k
            w[shorts] = -1.0 / k
        gross = float((w * rets).sum())
        turnover = float((w - w_prev).abs().sum())
        net = gross - turnover * fee
        rows[date] = {"gross": gross, "net": net, "turnover": turnover}
        w_prev = w
    return pd.DataFrame.from_dict(rows, orient="index")[["gross", "net", "turnover"]]


# ── Performance metrics ────────────────────────────────────────────────────────

def perf_metrics(returns: pd.Series, market: pd.Series | None = None) -> dict:
    """Annualized return, Sharpe, max drawdown, total return, alpha/beta.

    Sharpe uses rf=0 and sample std (ddof=1). Alpha/beta are OLS of the
    portfolio (y) on the market (x); alpha is annualized (daily * 252).
    If ``market`` is None, alpha/beta are NaN.
    """
    r = pd.Series(returns, dtype=float).dropna()
    n = len(r)
    if n == 0:
        return {"total_return": float("nan"), "ann_return": float("nan"),
                "sharpe": float("nan"), "max_drawdown": float("nan"),
                "alpha": float("nan"), "beta": float("nan")}
    cum = (1 + r).cumprod()
    total_return = float(cum.iloc[-1] - 1)
    ann_return = float((1 + total_return) ** (TRADING_DAYS / n) - 1)
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(r.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else float("nan")
    max_drawdown = float((cum / cum.cummax() - 1).min())

    alpha = beta = float("nan")
    if market is not None:
        m = pd.Series(market, dtype=float)
        joined = pd.concat([r, m], axis=1, join="inner").dropna()
        if len(joined) > 1 and joined.iloc[:, 1].std() > 0:
            slope, intercept, _, _, _ = stats.linregress(
                joined.iloc[:, 1].to_numpy(), joined.iloc[:, 0].to_numpy())
            beta = float(slope)
            alpha = float(intercept) * TRADING_DAYS
    return {"total_return": total_return, "ann_return": ann_return,
            "sharpe": sharpe, "max_drawdown": max_drawdown,
            "alpha": alpha, "beta": beta}


# ── End-to-end ─────────────────────────────────────────────────────────────────

def evaluate(pred: pd.DataFrame, label: pd.DataFrame, quantile: float = 0.1,
             fee: float = 0.001, market: pd.Series | None = None,
             n_boot: int = 1000, seed: int = 0) -> dict:
    """Compute the full metric set for one model.

    ``market`` defaults to the equal-weight universe return (cross-sectional
    mean of ``label`` per day), a reproducible offline market proxy used for
    alpha/beta of the long-short book.
    """
    pred, label = _align(pred, label)
    rank_ic = daily_rank_ic(pred, label)
    pearson_ic = daily_pearson_ic(pred, label)
    stats_ic = ic_stats(rank_ic)
    lo, hi = bootstrap_ic_ci(rank_ic, n_boot=n_boot, seed=seed)

    ls = longshort_returns(pred, label, quantile=quantile, fee=fee)
    if market is None:
        market = label.mean(axis=1)
    perf = perf_metrics(ls["net"], market=market)

    return {
        "ic_mean": stats_ic["ic_mean"],
        "ic_std": stats_ic["ic_std"],
        "icir": stats_ic["icir"],
        "tstat": stats_ic["tstat"],
        "pvalue": stats_ic["pvalue"],
        "pct_positive": stats_ic["pct_positive"],
        "n_days": stats_ic["n"],
        "ic_pearson": float(pearson_ic.dropna().mean()) if pearson_ic.notna().any() else float("nan"),
        "ic_ci_low": lo,
        "ic_ci_high": hi,
        "ann_return": perf["ann_return"],
        "sharpe": perf["sharpe"],
        "max_drawdown": perf["max_drawdown"],
        "total_return": perf["total_return"],
        "alpha": perf["alpha"],
        "beta": perf["beta"],
        "turnover_mean": float(ls["turnover"].mean()),
    }


# ── Internals ──────────────────────────────────────────────────────────────────

def _align(pred: pd.DataFrame, label: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both panels to their common dates and tickers, same order."""
    common_dates = pred.index.intersection(label.index)
    common_cols = pred.columns.intersection(label.columns)
    if len(common_dates) == 0 or len(common_cols) == 0:
        raise ValueError(
            f"No overlap between pred and label "
            f"(dates: {len(common_dates)}, tickers: {len(common_cols)})"
        )
    p = pred.loc[common_dates, common_cols]
    l = label.loc[common_dates, common_cols]
    return p, l
