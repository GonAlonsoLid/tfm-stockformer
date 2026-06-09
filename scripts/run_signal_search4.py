#!/usr/bin/env python3
"""Ronda 4 (final de la búsqueda): multi-factor calidad + valor + momentum residual.

Calidad y momentum son los dos factores más robustos y complementarios del equity;
el blend a partes iguales (sin ajustar pesos -> sin overfitting) suele subir Sharpe
y bajar drawdown. Aquí se añade un factor de CALIDAD y otro de VALOR construidos a
partir de los fundamentales EDGAR point-in-time, en blend con el momentum residual
de la ronda 3 y con el ensemble, sobre la MISMA ventana larga y justa (311 sem con
2021 dentro), con holdout final intacto.

Señales de factor (z-scores cross-seccionales, missing=neutro):
  calidad = roa + gross_profitability - asset_growth - net_issuance
  valor   = earnings_yield + sales_to_price + book_to_market

PRE-REGISTRO: éxito = t Newey-West > 2 en periodo completo Y holdout > 0. Todo se
reporta, gane o no. Pesos de blend fijos a partes iguales (no se ajustan al OOS).

Salida: results/signal_search4.csv
Uso: python scripts/run_signal_search4.py --init_train 104 --step 26
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import data_panel as dp  # noqa: E402
from lib import weekly_panel as wp  # noqa: E402
from run_weekly_robustness import CONFIGS, backtest_variant, sharpe  # noqa: E402
from run_signal_search import xz  # noqa: E402
from run_signal_search2 import resid_mom, per_year, ev  # noqa: E402
from run_signal_search3 import get_long_ensemble  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
FUND = "fundamentals.npz"
HOLD_WEEKS = 104


def Z(v):
    return np.nan_to_num(xz(v), nan=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--init_train", type=int, default=104)
    ap.add_argument("--step", type=int, default=26)
    args = ap.parse_args()

    print("Cargando panel + fundamentales + semanal ...")
    panel = dp.load_panel(args.data_dir)
    panel_f = dp.attach_fundamentals(panel, os.path.join(args.data_dir, FUND))
    week = wp.build_weekly(panel)
    week_f = wp.build_weekly(panel_f)
    # índices de los 7 fundamentales (orden de fundamentals.npz)
    # roa,-7 gross_prof,-6 asset_growth,-5 net_issuance,-4 earnings_yield,-3 s2p,-2 b2m,-1
    print(f"  {week.Xw.shape[0]} sem x {week.Xw.shape[1]} acc; F+fund={week_f.Xw.shape[2]}")

    ens = get_long_ensemble(week, args.init_train, args.step)
    weeks = sorted(ens)
    print(f"  ventana OOS: {len(weeks)} sem ({week.dates_w[weeks[0]].date()} a "
          f"{week.dates_w[weeks[-1]].date()}); holdout={HOLD_WEEKS}")

    dy = week.daily_y
    names = ["quality", "value", "momresid_quality", "ens_momresid_quality",
             "momresid_qual_val", "ens_momresid_qual_val"]
    sig = {n: {} for n in names}
    for wk in weeks:
        d = int(week.rebal_idx[wk])
        F = week_f.Xw[wk]                      # [N, F+7]
        roa, gp, ag, ni = F[:, -7], F[:, -6], F[:, -5], F[:, -4]
        ey, s2p, b2m = F[:, -3], F[:, -2], F[:, -1]
        quality = Z(roa) + Z(gp) - Z(ag) - Z(ni)
        value = Z(ey) + Z(s2p) + Z(b2m)
        mrs = Z(resid_mom(dy, d))
        e = Z(ens[wk])
        sig["quality"][wk] = quality
        sig["value"][wk] = value
        sig["momresid_quality"][wk] = mrs + quality
        sig["ens_momresid_quality"][wk] = e + mrs + quality
        sig["momresid_qual_val"][wk] = mrs + quality + value
        sig["ens_momresid_qual_val"][wk] = e + mrs + quality + value

    hold_set = set(weeks[-HOLD_WEEKS:])
    rows = []
    for n, preds in sig.items():
        bt = backtest_variant(week, preds, CONFIGS["full"]).copy()
        wk_of = {week.dates_w[wk]: wk for wk in weeks}
        is_hold = np.array([wk_of.get(dt, -1) in hold_set for dt in bt.index])
        full, sel, hold = ev(bt["net"]), ev(bt["net"][~is_hold]), ev(bt["net"][is_hold])
        rows.append(dict(signal=n, full_sharpe=full["sharpe"], full_t=full["t"],
                         sel_sharpe=sel["sharpe"], hold_sharpe=hold["sharpe"], hold_t=hold["t"],
                         full_ann=full["ann"], full_mdd=full["mdd"],
                         turnover=float(bt["turnover"].mean()),
                         gross=sharpe(bt["gross"]), ic=float(bt["ic"].dropna().mean())))
        print(f"  {n:24s} full S={full['sharpe']:+.2f}(t={full['t']:+.2f}) "
              f"sel={sel['sharpe']:+.2f} hold={hold['sharpe']:+.2f}(t={hold['t']:+.2f}) "
              f"turn={bt['turnover'].mean():.2f} gross={sharpe(bt['gross']):+.2f} IC={bt['ic'].dropna().mean():+.4f}")
        print(f"      por año: {per_year(bt)}")

    df = pd.DataFrame(rows).sort_values("full_sharpe", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "signal_search4.csv"), index=False)
    print("\n" + "=" * 88)
    print(f"  RONDA 4 — multi-factor sobre ventana justa ({len(weeks)} sem, incluye 2021)")
    print("=" * 88)
    print(df.to_string(index=False))
    print("=" * 88)
    b = df.iloc[0]
    ok = (b["full_t"] > 2) and (b["hold_sharpe"] > 0)
    print(f"\n  Mejor: {b['signal']} full S={b['full_sharpe']:+.2f} t={b['full_t']:+.2f} "
          f"hold={b['hold_sharpe']:+.2f} -> {'CRUZA EL LISTÓN' if ok else 'no cruza (t<2 o holdout<=0)'}")


if __name__ == "__main__":
    main()
