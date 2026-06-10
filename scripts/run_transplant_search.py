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


def _ev(net: pd.Series) -> dict:
    net = pd.Series(net).dropna()
    return {"sharpe": sharpe(net), "t": nw_tstat(net)}


def main():
    panel = dp.load_panel(DATA_DIR)
    week = wp.build_weekly(panel)
    ens = get_long_ensemble(week, INIT_TRAIN, STEP)
    weeks = sorted(ens)
    dy = week.daily_y
    hold_dates = {week.dates_w[w] for w in weeks[-HOLD_WEEKS:]}

    cand = {k: {} for k in
            ["ensemble", "peer", "trend", "ens_peer", "ens_trend", "ens_peer_trend"]}
    for wk in weeks:
        d = int(week.rebal_idx[wk])
        e = ts.xz(ens[wk])
        peer = ts.xz(ts.peer_graph_signal(dy, d))
        trend = ts.xz(ts.filtered_trend_signal(dy, d))
        e0, p0, t0 = np.nan_to_num(e), np.nan_to_num(peer), np.nan_to_num(trend)
        cand["ensemble"][wk] = e
        cand["peer"][wk] = peer
        cand["trend"][wk] = trend
        cand["ens_peer"][wk] = e0 + p0
        cand["ens_trend"][wk] = e0 + t0
        cand["ens_peer_trend"][wk] = e0 + p0 + t0

    rows = []
    for name, preds in cand.items():
        bt = backtest_variant(week, preds, CONFIGS["full"]).copy()
        ish = np.array([dt in hold_dates for dt in bt.index])
        full, hold = _ev(bt["net"]), _ev(bt["net"][ish])
        rows.append({"signal": name, "sharpe": full["sharpe"], "t": full["t"],
                     "hold": hold["sharpe"], "turnover": float(bt["turnover"].mean()),
                     "ic": float(bt["ic"].dropna().mean())})
        print(f"  {name:16s} Sharpe={full['sharpe']:+.2f} t={full['t']:+.2f} "
              f"hold={hold['sharpe']:+.2f} ic={rows[-1]['ic']:+.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "transplant_search.csv"), index=False)
    print(f"\nSaved results/transplant_search.csv ({len(rows)} signals)")


if __name__ == "__main__":
    main()
