#!/usr/bin/env python3
"""Robustness + attribution for the weekly market-neutral strategy (Tier A).

Two analyses that blindan the result before adding data:

  1. WALK-FORWARD: retrain the ensemble periodically and concatenate ~4 years of
     out-of-sample weeks into one equity curve, reporting net Sharpe over the
     full OOS span and per-segment, so the single-window result is stress-tested.

  2. CONSTRUCTION ATTRIBUTION: run the SAME walk-forward signal through four
     construction variants of increasing sophistication, isolating how much net
     Sharpe comes from portfolio construction vs the raw signal:
        raw_decile -> +neutralize -> +cost_aware -> full (+vol_target+regime)

Outputs:
    results/weekly_robustness_summary.csv
    results/weekly_attribution.csv
    results/figures/weekly_oos_equity.png
    results/figures/weekly_attribution.png

Usage:
    python scripts/run_weekly_robustness.py --init_train 200 --step 26
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
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import data_panel as dp  # noqa: E402
from lib import neutralize as nz  # noqa: E402
from lib import portfolio as pf  # noqa: E402
from lib import weekly_panel as wp  # noqa: E402
from run_weekly_strategy import train_ensemble, regime_scale  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
WEEKS_PER_YEAR = 52
BETA_WINDOW = 60
COST_BPS = 8.0
GROSS = 2.0
NAME_CAP = 0.04
TARGET_VOL = 0.10


# ── Walk-forward signal generation ──────────────────────────────────────────────

def walkforward_signal(week, init_train_w, step_w):
    """Retrain every step_w weeks; return {week_idx: alpha[N]} over the OOS span."""
    W = week.Xw.shape[0]
    preds = {}
    start = init_train_w
    n_retrains = 0
    while start < W:
        end = min(start + step_w, W)
        p, tw = train_ensemble(week.Xw, week.yw, start)   # train <start, predict >=start
        n_retrains += 1
        for k, wk in enumerate(tw):
            if start <= wk < end:
                preds[wk] = p[k]
        start = end
    print(f"  walk-forward: {len(preds)} OOS weeks, {n_retrains} retrains")
    return preds


# ── Construction variants (same signal, increasing sophistication) ──────────────

def _decile_weights(sig, gross):
    n = len(sig); k = max(1, int(n * 0.1))
    w = np.zeros(n)
    valid = ~np.isnan(sig)
    order = np.argsort(np.where(valid, sig, -np.inf))
    w[order[-k:]] = 1.0 / k
    w[order[:k]] = -1.0 / k
    return w * (gross / 2.0)


CONFIGS = {
    "raw_decile":   dict(neutral=False, smooth=False, costaware=False, voltarget=False, regime=False),
    "+neutralize":  dict(neutral=True,  smooth=False, costaware=False, voltarget=False, regime=False),
    "+cost_aware":  dict(neutral=True,  smooth=True,  costaware=True,  voltarget=False, regime=False),
    "full":         dict(neutral=True,  smooth=True,  costaware=True,  voltarget=True,  regime=True),
}


def backtest_variant(week, preds, cfg, smooth_halflife=2.0):
    market_daily = np.nanmean(week.daily_y, axis=1)
    ewma_a = 1 - 0.5 ** (1 / smooth_halflife)
    weeks = sorted(preds)
    w_prev = np.zeros(week.Xw.shape[1])
    smoothed = None
    rows = []
    for wk in weeks:
        d = week.rebal_idx[wk]
        alpha = preds[wk]
        if np.all(np.isnan(alpha)):
            continue
        win = slice(max(0, d - BETA_WINDOW), d)
        beta = pf.rolling_beta(week.daily_y[win], market_daily[win])
        sig = nz.neutralize(alpha, beta.reshape(-1, 1)) if cfg["neutral"] else _zscore(alpha)
        sig = np.nan_to_num(sig, nan=0.0)
        if cfg["smooth"]:
            smoothed = sig if smoothed is None else ewma_a * sig + (1 - ewma_a) * smoothed
            use = smoothed
        else:
            use = sig
        if cfg["costaware"]:
            w = pf.costaware_weights(use, beta, w_prev=w_prev, gross=GROSS,
                                     name_cap=NAME_CAP, cost_bps=COST_BPS,
                                     beta_neutral=cfg["neutral"])
        else:
            w = _decile_weights(use, GROSS)
        if cfg["voltarget"]:
            w = w * pf.vol_target_scale(week.daily_y[win], w, target_ann_vol=TARGET_VOL)
        if cfg["regime"]:
            w = w * regime_scale(market_daily, d)

        realized = np.nan_to_num(week.yw[wk], nan=0.0)
        gross = float(w @ realized)
        turnover = float(np.abs(w - w_prev).sum())
        net = gross - turnover * (COST_BPS / 1e4)
        m = ~np.isnan(week.yw[wk])
        ic = spearmanr(use[m], week.yw[wk][m]).correlation if m.sum() > 5 else np.nan
        rows.append({"date": week.dates_w[wk], "gross": gross, "net": net,
                     "turnover": turnover, "ic": ic})
        w_prev = w
    return pd.DataFrame(rows).set_index("date")


def _zscore(v):
    valid = ~np.isnan(v)
    out = np.full_like(v, np.nan, dtype=float)
    if valid.sum() > 1:
        s = v[valid]
        out[valid] = (s - s.mean()) / (s.std() + 1e-12)
    return out


# ── Stats ───────────────────────────────────────────────────────────────────────

def sharpe(r):
    r = pd.Series(r).dropna()
    return float(r.mean() / r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)) if r.std() > 0 else float("nan")


def ann(r):
    r = pd.Series(r).dropna()
    return float((1 + r).prod() ** (WEEKS_PER_YEAR / len(r)) - 1)


def maxdd(r):
    cum = (1 + pd.Series(r).dropna()).cumprod()
    return float((cum / cum.cummax() - 1).min())


def sharpe_se(sr, n):
    """Approx standard error of an annualized Sharpe over n weekly obs."""
    sr_w = sr / np.sqrt(WEEKS_PER_YEAR)
    se_w = np.sqrt((1 + 0.5 * sr_w ** 2) / n)
    return se_w * np.sqrt(WEEKS_PER_YEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--init_train", type=int, default=200, help="initial train weeks (~4y)")
    ap.add_argument("--step", type=int, default=26, help="retrain step in weeks (~semiannual)")
    ap.add_argument("--with_fundamentals", action="store_true",
                    help="concatenate EDGAR fundamentals.npz to the feature set")
    ap.add_argument("--with_realized", action="store_true",
                    help="concatenate realized_vol.npz (Tier C risk features)")
    args = ap.parse_args()

    print("Loading panel + weekly resampling ...")
    panel = dp.load_panel(args.data_dir)
    if args.with_fundamentals:
        panel = dp.attach_fundamentals(panel, os.path.join(args.data_dir, "fundamentals.npz"))
        print(f"  + fundamentals -> {panel.X.shape[2]} features")
    if args.with_realized:
        panel = dp.attach_fundamentals(panel, os.path.join(args.data_dir, "realized_vol.npz"))
        print(f"  + realized-vol -> {panel.X.shape[2]} features")
    week = wp.build_weekly(panel)
    print(f"  {week.Xw.shape[0]} weeks total")

    print("Walk-forward signal (retraining) ...")
    preds = walkforward_signal(week, args.init_train, args.step)

    # 1. Attribution across construction variants
    print("\nConstruction attribution (same OOS signal):")
    attr_rows, full_bt = [], None
    for name, cfg in CONFIGS.items():
        bt = backtest_variant(week, preds, cfg)
        n = len(bt)
        ns = sharpe(bt["net"])
        attr_rows.append({"variant": name, "n_weeks": n,
                          "net_sharpe": ns, "net_sharpe_se": sharpe_se(ns, n),
                          "gross_sharpe": sharpe(bt["gross"]),
                          "net_ann": ann(bt["net"]), "net_maxdd": maxdd(bt["net"]),
                          "turnover": float(bt["turnover"].mean()),
                          "ic_mean": float(bt["ic"].dropna().mean())})
        print(f"  {name:13s} net Sharpe={ns:+.2f}±{sharpe_se(ns,n):.2f}  "
              f"ann={ann(bt['net']):+.1%}  maxDD={maxdd(bt['net']):+.1%}  turn={bt['turnover'].mean():.2f}")
        if name == "full":
            full_bt = bt
    attr = pd.DataFrame(attr_rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    attr.to_csv(os.path.join(RESULTS_DIR, "weekly_attribution.csv"), index=False)

    # 2. Walk-forward robustness of the full strategy (per-year segments)
    full_bt = full_bt.copy()
    full_bt["year"] = pd.DatetimeIndex(full_bt.index).year
    seg = full_bt.groupby("year")["net"].apply(lambda r: pd.Series(
        {"net_sharpe": sharpe(r), "net_ann": ann(r), "weeks": len(r)}))
    seg = seg.unstack()
    full_sharpe = sharpe(full_bt["net"])
    summary = {"oos_weeks": len(full_bt), "net_sharpe": full_sharpe,
               "net_sharpe_se": sharpe_se(full_sharpe, len(full_bt)),
               "net_ann": ann(full_bt["net"]), "net_maxdd": maxdd(full_bt["net"]),
               "gross_sharpe": sharpe(full_bt["gross"]),
               "turnover": float(full_bt["turnover"].mean()),
               "ic_mean": float(full_bt["ic"].dropna().mean())}
    pd.DataFrame([summary]).to_csv(os.path.join(RESULTS_DIR, "weekly_robustness_summary.csv"), index=False)

    print("\n" + "=" * 60)
    print("  WALK-FORWARD ROBUSTNESS — full strategy")
    print("=" * 60)
    print(f"  OOS weeks          : {summary['oos_weeks']}")
    print(f"  NET Sharpe (full)  : {full_sharpe:+.2f} ± {summary['net_sharpe_se']:.2f}")
    print(f"  NET ann return     : {summary['net_ann']:+.1%}")
    print(f"  NET max drawdown   : {summary['net_maxdd']:+.1%}")
    print(f"  Mean weekly IC     : {summary['ic_mean']:+.4f}")
    print("  Per-year net Sharpe:")
    for yr, row in seg.iterrows():
        print(f"    {int(yr)}: Sharpe={row['net_sharpe']:+.2f}  ann={row['net_ann']:+.1%}  ({int(row['weeks'])} wks)")
    print("=" * 60)

    # Figures
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(full_bt.index, (1 + full_bt["gross"]).cumprod(), color="#7f8c8d", lw=1.2, label="Bruto")
    ax.plot(full_bt.index, (1 + full_bt["net"]).cumprod(), color="#c0392b", lw=2, label="Neto (8 bps/lado)")
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_title(f"Walk-forward OOS: retorno acumulado bruto y neto ({summary['oos_weeks']} semanas)", fontsize=13)
    ax.set_xlabel("Fecha"); ax.set_ylabel("Retorno acumulado")
    ax.legend(fontsize=10); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "weekly_oos_equity.png"), dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    x = range(len(attr))
    ax.bar(x, attr["net_sharpe"], yerr=attr["net_sharpe_se"], capsize=5,
           color=["#bdc3c7", "#85c1e9", "#5dade2", "#c0392b"], edgecolor="black")
    ax.set_xticks(list(x)); ax.set_xticklabels(attr["variant"], rotation=15)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Sharpe neto"); ax.set_title("Atribución: aporte de cada paso de construcción", fontsize=13)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout(); fig.savefig(os.path.join(FIGURES_DIR, "weekly_attribution.png"), dpi=300); plt.close(fig)
    print("Saved robustness/attribution CSVs and figures.")


if __name__ == "__main__":
    main()
