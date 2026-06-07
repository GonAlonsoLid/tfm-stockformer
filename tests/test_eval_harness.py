"""Tests for the unified evaluation harness (lib/eval_harness.py).

The harness is the single measurement instrument used by EVERY model in the
complexity ladder, so its correctness is verified against hand-computed values.

Convention used everywhere:
    pred  : DataFrame [dates x tickers], predicted score (higher = more bullish)
    label : DataFrame [dates x tickers], realized forward return
"""
import numpy as np
import pandas as pd
import pytest

from lib import eval_harness as eh


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _frame(rows, dates=None, tickers=None):
    """Build a [dates x tickers] DataFrame from a list of row lists."""
    arr = np.asarray(rows, dtype=float)
    n_days, n_stocks = arr.shape
    if dates is None:
        dates = pd.bdate_range("2024-01-01", periods=n_days)
    if tickers is None:
        tickers = [f"S{i}" for i in range(n_stocks)]
    return pd.DataFrame(arr, index=pd.DatetimeIndex(dates), columns=tickers)


# ── daily_rank_ic ─────────────────────────────────────────────────────────────

def test_perfect_positive_ranking_gives_ic_one():
    label = _frame([[0.05, 0.04, 0.03, -0.02, -0.05]])
    pred = label.copy()  # identical order -> Spearman 1.0
    ic = eh.daily_rank_ic(pred, label)
    assert ic.iloc[0] == pytest.approx(1.0)


def test_perfect_inverse_ranking_gives_ic_minus_one():
    label = _frame([[0.05, 0.04, 0.03, -0.02, -0.05]])
    pred = _frame([[-0.05, -0.04, -0.03, 0.02, 0.05]])  # reversed order
    ic = eh.daily_rank_ic(pred, label)
    assert ic.iloc[0] == pytest.approx(-1.0)


def test_daily_rank_ic_returns_one_value_per_date():
    label = _frame([[1, 2, 3], [3, 2, 1], [1, 3, 2]])
    pred = _frame([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    ic = eh.daily_rank_ic(pred, label)
    assert len(ic) == 3
    assert ic.iloc[0] == pytest.approx(1.0)
    assert ic.iloc[1] == pytest.approx(-1.0)


def test_constant_prediction_day_is_nan_and_excluded():
    label = _frame([[0.01, 0.02, 0.03]])
    pred = _frame([[5.0, 5.0, 5.0]])  # no rank information -> NaN
    ic = eh.daily_rank_ic(pred, label)
    assert np.isnan(ic.iloc[0])


# ── ic_stats ──────────────────────────────────────────────────────────────────

def test_ic_stats_mean_and_icir():
    daily_ic = pd.Series([0.1, 0.2, 0.3, 0.0, -0.1])
    stats = eh.ic_stats(daily_ic)
    assert stats["ic_mean"] == pytest.approx(0.1)
    assert stats["ic_std"] == pytest.approx(np.std([0.1, 0.2, 0.3, 0.0, -0.1], ddof=1))
    assert stats["icir"] == pytest.approx(stats["ic_mean"] / stats["ic_std"])
    assert stats["n"] == 5


def test_ic_stats_pct_positive():
    daily_ic = pd.Series([0.1, -0.2, 0.3, 0.0, 0.5])  # 3 of 5 strictly > 0
    stats = eh.ic_stats(daily_ic)
    assert stats["pct_positive"] == pytest.approx(60.0)


def test_ic_stats_tstat_sign_matches_mean():
    pos = eh.ic_stats(pd.Series([0.05, 0.06, 0.04, 0.05]))
    neg = eh.ic_stats(pd.Series([-0.05, -0.06, -0.04, -0.05]))
    assert pos["tstat"] > 0
    assert neg["tstat"] < 0


def test_ic_stats_ignores_nan_days():
    daily_ic = pd.Series([0.1, np.nan, 0.3])
    stats = eh.ic_stats(daily_ic)
    assert stats["n"] == 2
    assert stats["ic_mean"] == pytest.approx(0.2)


# ── bootstrap_ic_ci ───────────────────────────────────────────────────────────

def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    rng = np.random.default_rng(0)
    daily_ic = pd.Series(rng.normal(0.03, 0.1, size=500))
    lo1, hi1 = eh.bootstrap_ic_ci(daily_ic, n_boot=1000, seed=42)
    lo2, hi2 = eh.bootstrap_ic_ci(daily_ic, n_boot=1000, seed=42)
    assert (lo1, hi1) == (lo2, hi2)            # reproducible with seed
    assert lo1 < daily_ic.mean() < hi1          # CI brackets the sample mean


# ── paired_ic_ttest ───────────────────────────────────────────────────────────

def test_paired_ttest_detects_better_model():
    # model A consistently ~0.05 higher daily IC than model B (small jitter so
    # the difference is not perfectly constant, as in real data)
    rng = np.random.default_rng(7)
    ic_b = pd.Series(rng.normal(0.0, 0.01, size=30))
    ic_a = ic_b + 0.05 + rng.normal(0.0, 0.005, size=30)
    res = eh.paired_ic_ttest(ic_a, ic_b)
    assert res["mean_diff"] == pytest.approx(0.05, abs=0.01)
    assert res["tstat"] > 0
    assert res["pvalue"] < 0.05


# ── longshort_returns ─────────────────────────────────────────────────────────

def test_longshort_is_dollar_neutral_and_known_return():
    # 5 stocks: pred ranks them A>B>C>D>E; quantile 0.2 -> 1 long, 1 short.
    label = _frame([[0.05, 0.01, 0.0, -0.01, -0.05]],
                   tickers=["A", "B", "C", "D", "E"])
    pred = _frame([[5, 4, 3, 2, 1]], tickers=["A", "B", "C", "D", "E"])
    ls = eh.longshort_returns(pred, label, quantile=0.2, fee=0.0)
    # gross = 1.0*0.05 (long A) - 1.0*(-0.05) (short E) = 0.10
    assert ls["gross"].iloc[0] == pytest.approx(0.10)
    # day-0 turnover = |+1| + |-1| = 2.0
    assert ls["turnover"].iloc[0] == pytest.approx(2.0)


def test_longshort_fee_reduces_net_return():
    label = _frame([[0.05, 0.01, 0.0, -0.01, -0.05]],
                   tickers=["A", "B", "C", "D", "E"])
    pred = _frame([[5, 4, 3, 2, 1]], tickers=["A", "B", "C", "D", "E"])
    ls = eh.longshort_returns(pred, label, quantile=0.2, fee=0.001)
    # net = gross - turnover*fee = 0.10 - 2.0*0.001 = 0.098
    assert ls["net"].iloc[0] == pytest.approx(0.098)


# ── perf_metrics ──────────────────────────────────────────────────────────────

def test_perf_metrics_sharpe_and_total_return():
    returns = pd.Series([0.01, -0.005, 0.02, 0.0, 0.01])
    m = eh.perf_metrics(returns)
    expected_total = float((1 + returns).prod() - 1)
    assert m["total_return"] == pytest.approx(expected_total)
    expected_sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert m["sharpe"] == pytest.approx(expected_sharpe)


def test_perf_metrics_beta_zero_when_uncorrelated_with_market():
    # portfolio return constant -> zero covariance with any market -> beta 0
    returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
    market = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
    m = eh.perf_metrics(returns, market=market)
    assert m["beta"] == pytest.approx(0.0, abs=1e-9)


# ── evaluate (end-to-end) ─────────────────────────────────────────────────────

def test_evaluate_returns_all_metric_blocks():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2024-01-01", periods=40)
    tickers = [f"S{i}" for i in range(20)]
    label = pd.DataFrame(rng.normal(0, 0.02, (40, 20)), index=dates, columns=tickers)
    # pred = label + noise -> mild positive IC
    pred = label + rng.normal(0, 0.02, (40, 20))
    out = eh.evaluate(pred, label, quantile=0.2, fee=0.001, n_boot=200, seed=0)
    for key in ("ic_mean", "icir", "tstat", "pct_positive",
                "ic_ci_low", "ic_ci_high", "sharpe", "ann_return",
                "max_drawdown", "beta", "turnover_mean"):
        assert key in out, f"missing metric: {key}"
    assert out["ic_mean"] > 0  # pred is correlated with label by construction


def test_evaluate_aligns_mismatched_columns_and_dates():
    label = _frame([[0.01, 0.02, 0.03], [0.0, -0.01, 0.02]],
                   tickers=["A", "B", "C"])
    # pred missing column C and with a shuffled column order
    pred = _frame([[2, 1], [1, 2]], tickers=["B", "A"])
    out = eh.evaluate(pred, label, quantile=0.5, fee=0.0, n_boot=50, seed=0)
    assert np.isfinite(out["ic_mean"])
