#!/usr/bin/env python3
"""Lote 1 modular-transfer signals through the cost-aware walk-forward.

Evaluates: ensemble (base), peer, trend, ens+peer, ens+trend, ens+peer+trend.
Writes results/transplant_search.csv with [signal, sharpe, t, hold, turnover, ic].
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))  # so run_* scripts import as modules

from lib import data_panel as dp          # noqa: E402
from lib import weekly_panel as wp         # noqa: E402
from lib import transplant_signals as ts   # noqa: E402
from run_signal_search3 import get_long_ensemble          # noqa: E402
from run_weekly_robustness import backtest_variant, sharpe, CONFIGS  # noqa: E402
from run_signal_search import nw_tstat                    # noqa: E402

DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
INIT_TRAIN = 104
STEP = 26
HOLD_WEEKS = 104


SIGNALS = {
    "peer":     lambda dy, d, dates: ts.peer_graph_signal(dy, d),
    "trend":    lambda dy, d, dates: ts.filtered_trend_signal(dy, d),
    "low_ivol": lambda dy, d, dates: ts.low_ivol_signal(dy, d),
    "bab":      lambda dy, d, dates: ts.bab_signal(dy, d),
    "volmom":   lambda dy, d, dates: ts.volmanaged_momentum_signal(dy, d),
    "high52":   lambda dy, d, dates: ts.fiftytwo_week_high_signal(dy, d),
    "tsmom":    lambda dy, d, dates: ts.ts_momentum_signal(dy, d),
    "season":   lambda dy, d, dates: ts.seasonality_signal(dy, d, dates),
    "highfreq_rev": lambda dy, d, dates: ts.high_freq_reversal_signal(dy, d),
    "leadlag":  lambda dy, d, dates: ts.lead_lag_signal(dy, d),
}


def _load_sector_ids(tickers):
    """Load data/<dir>/sector_map.json (if present) -> [N] int sector labels aligned to
    `tickers`; returns None if the file is absent or no ticker has a known sector."""
    import json
    path = os.path.join(os.path.dirname(__file__), "..", DATA_DIR, "sector_map.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        smap = json.load(f)
    names = [smap.get(t, "Unknown") for t in tickers]
    if all(n == "Unknown" for n in names):
        return None
    uniq = {s: i for i, s in enumerate(sorted(set(names)))}
    return np.array([uniq[n] for n in names])


def _ev(net: pd.Series) -> dict:
    net = pd.Series(net).dropna()
    return {"sharpe": sharpe(net), "t": nw_tstat(net)}


def main():
    panel = dp.load_panel(DATA_DIR)
    week = wp.build_weekly(panel)
    dates = pd.DatetimeIndex(panel.dates)
    ens = get_long_ensemble(week, INIT_TRAIN, STEP)
    weeks = sorted(ens)
    dy = week.daily_y
    hold_dates = {week.dates_w[w] for w in weeks[-HOLD_WEEKS:]}

    active = dict(SIGNALS)
    sector_ids = _load_sector_ids(week.tickers)
    if sector_ids is not None:
        active["secpeer"] = lambda dy, d, dates: ts.sector_peer_momentum_signal(dy, d, sector_ids)
        print(f"  sector-peer ACTIVADO ({len(set(sector_ids.tolist()))} sectores)")
    else:
        print("  sector-peer DESACTIVADO (sin sector_map.json)")

    names = ["ensemble"] + list(active) + [f"ens_{k}" for k in active]
    cand = {k: {} for k in names}
    for wk in weeks:
        d = int(week.rebal_idx[wk])
        e = ts.xz(ens[wk])
        e0 = np.nan_to_num(e)
        cand["ensemble"][wk] = e
        for k, fn in active.items():
            s = ts.xz(fn(dy, d, dates))
            cand[k][wk] = s
            cand[f"ens_{k}"][wk] = e0 + np.nan_to_num(s)

    rows = []
    for name in names:
        bt = backtest_variant(week, cand[name], CONFIGS["full"]).copy()
        ish = np.array([dt in hold_dates for dt in bt.index])
        full, hold = _ev(bt["net"]), _ev(bt["net"][ish])
        rows.append({"signal": name, "sharpe": full["sharpe"], "t": full["t"],
                     "hold": hold["sharpe"], "turnover": float(bt["turnover"].mean()),
                     "ic": float(bt["ic"].dropna().mean())})
        print(f"  {name:18s} Sharpe={full['sharpe']:+.2f} t={full['t']:+.2f} "
              f"hold={hold['sharpe']:+.2f} ic={rows[-1]['ic']:+.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "transplant_search.csv"), index=False)
    print(f"\nSaved results/transplant_search.csv ({len(rows)} signals)")


if __name__ == "__main__":
    main()
