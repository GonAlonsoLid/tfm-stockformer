#!/usr/bin/env python3
"""Tier-A weekly market-neutral strategy: the cost-aware returns pipeline.

End-to-end, fully offline on existing features:
    weekly panel -> ensemble (LightGBM + ElasticNet) -> per-week alpha
      -> beta-neutralize -> EWMA smooth -> cost-aware cvxpy weights
      -> vol-target -> VIX-style regime overlay -> weekly net-of-cost backtest

Reports gross/net Sharpe, ann. return, max drawdown, turnover, and IC, and saves
an equity curve. This is the core contribution: turning a thin signal into a
tradable market-neutral book whose NET Sharpe is the objective.

Usage:
    python scripts/run_weekly_strategy.py
    python scripts/run_weekly_strategy.py --cost_bps 8 --gross 2 --name_cap 0.04
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib import data_panel as dp  # noqa: E402
from lib import neutralize as nz  # noqa: E402
from lib import portfolio as pf  # noqa: E402
from lib import weekly_panel as wp  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
WEEKS_PER_YEAR = 52
BETA_WINDOW = 60        # daily rows for rolling beta / vol target
REGIME = {"low": 1.0, "mid": 0.7, "high": 0.4}


# ── Signal model ────────────────────────────────────────────────────────────────

def train_ensemble(Xw, yw, train_end_w, seed=0):
    """Train LightGBM + ElasticNet on the weekly panel; return per-week test alpha.

    Returns predictions for weeks [train_end_w:] as [W_test, N].
    """
    import lightgbm as lgb
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler

    W, N, F = Xw.shape
    # flatten train rows (weeks < train_end_w), drop NaN labels
    tr_w = np.arange(train_end_w)
    Xtr = Xw[tr_w].reshape(-1, F)
    ytr = yw[tr_w].reshape(-1)
    ok = ~np.isnan(ytr) & ~np.isnan(Xtr).any(axis=1)
    Xtr, ytr = Xtr[ok], ytr[ok]

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)

    gbm = lgb.LGBMRegressor(objective="huber", n_estimators=300, learning_rate=0.02,
                            num_leaves=31, subsample=0.7, colsample_bytree=0.5,
                            min_child_samples=100, reg_alpha=0.1, reg_lambda=1.0,
                            random_state=seed, n_jobs=-1, verbose=-1)
    enet = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=seed)
    gbm.fit(Xtr, ytr)
    enet.fit(Xtr_s, ytr)

    test_w = np.arange(train_end_w, W)
    preds = np.full((len(test_w), N), np.nan)
    for k, wk in enumerate(test_w):
        x = Xw[wk]                                  # [N, F]
        valid = ~np.isnan(x).any(axis=1)
        if valid.sum() == 0:
            continue
        p_g = gbm.predict(x[valid])
        p_e = enet.predict(scaler.transform(x[valid]))
        # average standardized predictions (rank-comparable ensemble)
        z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
        preds[k, valid] = 0.5 * z(p_g) + 0.5 * z(p_e)
    return preds, test_w


# ── Backtest ─────────────────────────────────────────────────────────────────

def regime_scale(market_daily: np.ndarray, d: int) -> float:
    """VIX-style overlay using trailing realized vol of the equal-weight market."""
    hist = market_daily[:d]
    if len(hist) < BETA_WINDOW + 20:
        return 1.0
    rv = pd.Series(hist).rolling(20).std().dropna()
    cur = rv.iloc[-1]
    lo, hi = rv.quantile(0.33), rv.quantile(0.67)
    return REGIME["low"] if cur < lo else (REGIME["high"] if cur >= hi else REGIME["mid"])


def run_backtest(week, preds, test_w, args):
    N = week.Xw.shape[1]
    market_daily = np.nanmean(week.daily_y, axis=1)  # equal-weight market proxy
    ewma_alpha = 1 - 0.5 ** (1 / args.smooth_halflife)

    w_prev = np.zeros(N)
    smoothed = None
    gross_rets, net_rets, turnovers, weekly_ic, dates = [], [], [], [], []

    for k, wk in enumerate(test_w):
        d = week.rebal_idx[wk]
        alpha_raw = preds[k]
        if np.all(np.isnan(alpha_raw)):
            continue
        # rolling beta vs market on trailing daily window
        win = slice(max(0, d - BETA_WINDOW), d)
        beta = pf.rolling_beta(week.daily_y[win], market_daily[win])
        # neutralize signal vs beta, then EWMA-smooth across weeks
        neutral = nz.neutralize(alpha_raw, beta.reshape(-1, 1))
        neutral = np.nan_to_num(neutral, nan=0.0)
        smoothed = neutral if smoothed is None else ewma_alpha * neutral + (1 - ewma_alpha) * smoothed

        w = pf.costaware_weights(smoothed, beta, w_prev=w_prev, gross=args.gross,
                                 name_cap=args.name_cap, cost_bps=args.cost_bps,
                                 beta_neutral=True)
        # vol target + regime overlay
        scale = pf.vol_target_scale(week.daily_y[win], w, target_ann_vol=args.target_vol)
        scale *= regime_scale(market_daily, d)
        w = w * scale

        realized = np.nan_to_num(week.yw[wk], nan=0.0)
        gross = float(w @ realized)
        turnover = float(np.abs(w - w_prev).sum())
        cost = turnover * (args.cost_bps / 1e4)
        net = gross - cost

        # information coefficient of the (neutral) signal this week
        from scipy.stats import spearmanr
        m = ~np.isnan(week.yw[wk])
        ic = spearmanr(smoothed[m], week.yw[wk][m]).correlation if m.sum() > 5 else np.nan

        gross_rets.append(gross); net_rets.append(net); turnovers.append(turnover)
        weekly_ic.append(ic); dates.append(week.dates_w[wk])
        w_prev = w

    return pd.DataFrame({"date": dates, "gross": gross_rets, "net": net_rets,
                         "turnover": turnovers, "ic": weekly_ic}).set_index("date")


def summarize(bt: pd.DataFrame) -> dict:
    def stats(r):
        r = r.dropna()
        ann = (1 + r).prod() ** (WEEKS_PER_YEAR / len(r)) - 1
        sharpe = r.mean() / r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR) if r.std() > 0 else np.nan
        cum = (1 + r).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        return ann, sharpe, mdd
    g_ann, g_sh, _ = stats(bt["gross"])
    n_ann, n_sh, n_mdd = stats(bt["net"])
    ic = bt["ic"].dropna()
    return {"n_weeks": len(bt), "gross_sharpe": g_sh, "gross_ann": g_ann,
            "net_sharpe": n_sh, "net_ann": n_ann, "net_maxdd": float(n_mdd),
            "ic_mean": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std(ddof=1)),
            "turnover_mean": float(bt["turnover"].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--cost_bps", type=float, default=8.0)
    ap.add_argument("--gross", type=float, default=2.0)
    ap.add_argument("--name_cap", type=float, default=0.04)
    ap.add_argument("--target_vol", type=float, default=0.10)
    ap.add_argument("--smooth_halflife", type=float, default=2.0)
    ap.add_argument("--with_fundamentals", action="store_true",
                    help="concatenate EDGAR fundamentals.npz to the feature set")
    args = ap.parse_args()

    print("Loading daily panel + building weekly panel ...")
    panel = dp.load_panel(args.data_dir)
    if args.with_fundamentals:
        panel = dp.attach_fundamentals(panel, os.path.join(args.data_dir, "fundamentals.npz"))
        print(f"  + fundamentals -> {panel.X.shape[2]} features")
    week = wp.build_weekly(panel)
    print(f"  weekly: {week.Xw.shape[0]} weeks x {week.Xw.shape[1]} stocks; "
          f"train_end_w={week.train_end_w} val_end_w={week.val_end_w}")

    print("Training ensemble (LightGBM + ElasticNet) ...")
    preds, test_w = train_ensemble(week.Xw, week.yw, week.val_end_w)

    print(f"Running weekly market-neutral backtest "
          f"(cost={args.cost_bps}bps, gross={args.gross}, cap={args.name_cap}) ...")
    bt = run_backtest(week, preds, test_w, args)
    s = summarize(bt)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    bt.to_csv(os.path.join(RESULTS_DIR, "weekly_strategy_returns.csv"))
    pd.DataFrame([s]).to_csv(os.path.join(RESULTS_DIR, "weekly_strategy_summary.csv"), index=False)

    print("\n" + "=" * 56)
    print("  WEEKLY MARKET-NEUTRAL STRATEGY — Tier A")
    print("=" * 56)
    print(f"  Test weeks         : {s['n_weeks']}")
    print(f"  Signal IC (mean)   : {s['ic_mean']:+.4f}  (IR {s['ic_ir']:+.2f})")
    print(f"  GROSS Sharpe       : {s['gross_sharpe']:+.2f}  (ann {s['gross_ann']:+.1%})")
    print(f"  NET Sharpe         : {s['net_sharpe']:+.2f}  (ann {s['net_ann']:+.1%})")
    print(f"  NET max drawdown   : {s['net_maxdd']:+.1%}")
    print(f"  Avg weekly turnover: {s['turnover_mean']:.2f}")
    print("=" * 56)

    # equity curve
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bt.index, (1 + bt["gross"]).cumprod(), label="Bruto", color="#7f8c8d", lw=1.3)
    ax.plot(bt.index, (1 + bt["net"]).cumprod(), label="Neto (8 bps/lado)", color="#c0392b", lw=2)
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_title(f"Estrategia semanal market-neutral — Sharpe neto {s['net_sharpe']:+.2f}", fontsize=13)
    ax.set_xlabel("Fecha"); ax.set_ylabel("Retorno acumulado")
    ax.legend(fontsize=10); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "weekly_strategy_equity.png"), dpi=300)
    plt.close(fig)
    print(f"Saved results/weekly_strategy_summary.csv and figures/weekly_strategy_equity.png")


if __name__ == "__main__":
    main()
