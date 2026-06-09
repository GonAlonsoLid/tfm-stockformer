#!/usr/bin/env python3
"""Genera la tabla LaTeX de la búsqueda de señal de RQ2, todas las señales sobre la
MISMA ventana justa de 311 semanas (2020-2026, incluye el momentum-crash de 2021;
holdout = últimas 104 semanas). Garantiza que las cifras del capítulo son
consistentes (una sola ventana, un solo protocolo).

Salida: MEMORIA/tfm/tablas/rq2_signal_search.tex  (+ stdout con cifras exactas)
Uso: python scripts/build_rq2_table.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import data_panel as dp  # noqa: E402
from lib import weekly_panel as wp  # noqa: E402
from run_weekly_robustness import CONFIGS, backtest_variant, sharpe  # noqa: E402
from run_signal_search import xz, cum_ret, idio_vol  # noqa: E402
from run_signal_search2 import resid_mom, ev  # noqa: E402
from run_signal_search3 import get_long_ensemble  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
TABLES_DIR = os.path.join(PROJECT_ROOT, "MEMORIA", "tfm", "tablas")
DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
HOLD_WEEKS = 104


def Z(v):
    return np.nan_to_num(xz(v), nan=0.0)


def main():
    panel = dp.load_panel(DATA_DIR)
    panel_f = dp.attach_fundamentals(panel, os.path.join(DATA_DIR, "fundamentals.npz"))
    week = wp.build_weekly(panel)
    week_f = wp.build_weekly(panel_f)
    dy = week.daily_y
    ens = get_long_ensemble(week, 104, 26)
    weeks = sorted(ens)

    sig = {k: {} for k in ["ensemble", "reversal", "quality", "momresid", "ens_momresid"]}
    for wk in weeks:
        d = int(week.rebal_idx[wk])
        iv = idio_vol(dy, d, 60)
        rev_raw = week.yw[wk - 1] if wk - 1 >= 0 else np.full(week.Xw.shape[1], np.nan)
        revvol = -xz(np.where(np.isnan(iv) | (iv < 1e-6), np.nan, rev_raw / iv))
        F = week_f.Xw[wk]
        quality = Z(F[:, -7]) + Z(F[:, -6]) - Z(F[:, -5]) - Z(F[:, -4])
        mrs = Z(resid_mom(dy, d))
        sig["ensemble"][wk] = ens[wk]
        sig["reversal"][wk] = revvol
        sig["quality"][wk] = quality
        sig["momresid"][wk] = mrs
        sig["ens_momresid"][wk] = Z(ens[wk]) + mrs

    label = {"ensemble": r"Ensemble ML (base)", "reversal": r"Reversal 1 sem.\ (vol-esc.)",
             "quality": r"Calidad (fundamentales)", "momresid": r"Momentum residual",
             "ens_momresid": r"\textbf{Ensemble + mom.\ residual}"}
    order = ["ensemble", "reversal", "quality", "momresid", "ens_momresid"]
    hold_set = set(weeks[-HOLD_WEEKS:])
    rows = []
    for k in order:
        bt = backtest_variant(week, sig[k], CONFIGS["full"]).copy()
        wk_of = {week.dates_w[wk]: wk for wk in weeks}
        is_h = np.array([wk_of.get(dt, -1) in hold_set for dt in bt.index])
        full, hold = ev(bt["net"]), ev(bt["net"][is_h])
        rows.append(dict(k=k, sharpe=full["sharpe"], t=full["t"], hold=hold["sharpe"],
                         turn=float(bt["turnover"].mean()), ic=float(bt["ic"].dropna().mean()),
                         gross=sharpe(bt["gross"])))
        print(f"{k:14s} S={full['sharpe']:+.2f} t={full['t']:+.2f} hold={hold['sharpe']:+.2f} "
              f"turn={bt['turnover'].mean():.2f} gross={sharpe(bt['gross']):+.2f} IC={bt['ic'].dropna().mean():+.4f}")

    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Búsqueda de señal para RQ2 sobre la misma ventana \textit{walk-forward} "
        r"de 311 semanas (2020--2026, incluye el \textit{momentum-crash} de 2021; "
        r"\textit{holdout}: últimas 104 semanas). Misma construcción \textit{cost-aware} "
        r"(coste 8 bps/lado) para todas las señales. El reversal pierde incluso en bruto y "
        r"los fundamentales no aportan; el momentum residual es la única señal con IC e "
        r"\textit{holdout} positivos, y su combinación con el \textit{ensemble} da el mejor "
        r"Sharpe neto.}",
        r"\label{tab:rq2-signal-search}", r"\small",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Señal & Sharpe neto & $t_{\text{NW}}$ & \textit{Holdout} & Turnover & IC \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f"{label[r['k']]} & ${r['sharpe']:+.2f}$ & ${r['t']:+.2f}$ & "
                     f"${r['hold']:+.2f}$ & {r['turn']:.2f} & ${r['ic']:+.4f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "rq2_signal_search.tex"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nGuardado: {os.path.join(TABLES_DIR, 'rq2_signal_search.tex')}")


if __name__ == "__main__":
    main()
