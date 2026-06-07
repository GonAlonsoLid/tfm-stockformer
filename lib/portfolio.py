"""Cost-aware market-neutral portfolio construction (TFM returns pipeline).

This is the layer that turns a thin cross-sectional signal into a tradable,
cost-aware book — the documented binding constraint for net Sharpe on S&P 500.

costaware_weights solves, per rebalance date:

    max_w   alpha·w  -  gamma·wᵀΣw  -  (cost_bps/1e4)·|w - w_prev|₁
    s.t.    Σ w = 0            (dollar neutral)
            beta·w = 0         (beta neutral)
            ‖w‖₁ ≤ gross       (leverage cap, e.g. 2 = 1 long + 1 short)
            |w_i| ≤ name_cap   (per-name cap)

The L1 transaction-cost term implements "trade only when edge > cost", which is
what suppresses turnover and recovers net Sharpe at weekly frequency.
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def costaware_weights(alpha: np.ndarray, beta: np.ndarray,
                      Sigma: np.ndarray | None = None,
                      w_prev: np.ndarray | None = None,
                      gross: float = 2.0, name_cap: float = 0.02,
                      cost_bps: float = 8.0, risk_aversion: float = 0.0,
                      beta_neutral: bool = True,
                      solver: str | None = None) -> np.ndarray:
    """Solve the cost-aware market-neutral weight problem with cvxpy.

    Returns a weight vector (length N); NaN alphas are treated as 0 (no view).
    Falls back to a zero book if the solver fails.
    """
    import cvxpy as cp

    alpha = np.nan_to_num(np.asarray(alpha, dtype=float).reshape(-1), nan=0.0)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    n = alpha.shape[0]
    if w_prev is None:
        w_prev = np.zeros(n)

    w = cp.Variable(n)
    objective = alpha @ w
    if risk_aversion > 0 and Sigma is not None:
        objective = objective - risk_aversion * cp.quad_form(w, cp.psd_wrap(Sigma))
    if cost_bps > 0:
        objective = objective - (cost_bps / 1e4) * cp.norm1(w - w_prev)

    constraints = [cp.sum(w) == 0, cp.norm1(w) <= gross, cp.abs(w) <= name_cap]
    if beta_neutral:
        constraints.append(beta @ w == 0)

    prob = cp.Problem(cp.Maximize(objective), constraints)
    try:
        prob.solve(solver=solver) if solver else prob.solve()
    except Exception:
        return np.zeros(n)
    if w.value is None:
        return np.zeros(n)
    return np.asarray(w.value).reshape(-1)


def vol_target_scale(returns_window: np.ndarray, weights: np.ndarray,
                     target_ann_vol: float = 0.10,
                     max_leverage: float = 3.0) -> float:
    """Leverage multiplier so the book hits ``target_ann_vol`` annualized.

    ``returns_window`` is [days x assets] of recent asset returns; the book's
    realized daily vol is estimated from the weighted series.
    """
    R = np.asarray(returns_window, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    book = R @ w
    daily_vol = np.std(book, ddof=1)
    if daily_vol <= 1e-12:
        return 0.0
    scale = target_ann_vol / (daily_vol * np.sqrt(TRADING_DAYS))
    return float(min(scale, max_leverage))


def rolling_beta(stock_returns: np.ndarray, market_returns: np.ndarray) -> np.ndarray:
    """Per-stock beta vs the market over a window. [days x N], [days] -> [N]."""
    R = np.asarray(stock_returns, dtype=float)
    m = np.asarray(market_returns, dtype=float).reshape(-1)
    mc = m - m.mean()
    var = float((mc ** 2).mean())
    if var <= 1e-12:
        return np.ones(R.shape[1])
    Rc = R - R.mean(axis=0, keepdims=True)
    cov = (Rc * mc.reshape(-1, 1)).mean(axis=0)
    return cov / var
