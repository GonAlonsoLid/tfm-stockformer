#!/usr/bin/env python3
"""Ronda 3 (decisiva): ensemble + momentum residual sobre una ventana larga y JUSTA.

La ronda 2 sugirió que (ensemble + momentum residual) da t>2, pero el ensemble
cacheado solo cubre 2022-2026 (excluye el momentum-crash de 2021): ventana
favorable. Aquí se regenera el ensemble walk-forward empezando antes (init_train
más corto) para evaluar TODAS las señales sobre la MISMA ventana de ~6 años que
SÍ incluye 2021, con holdout final intacto y desglose por año.

Comparación apples-to-apples sobre las mismas semanas OOS:
    ensemble | mom_resid | blend_mom_ra_resid | ens+momresid | ens+momresid+ra

PRE-REGISTRO: éxito = t Newey-West > 2 en el periodo completo Y holdout > 0,
sobre la ventana larga (con 2021). Se reporta todo.

Salida: results/signal_search3.csv  (+ cache results/_wf_preds_long.npz)

Uso: python scripts/run_signal_search3.py --init_train 104 --step 26
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
from run_weekly_robustness import CONFIGS, backtest_variant, sharpe, ann, maxdd, walkforward_signal  # noqa: E402
from run_signal_search import xz, nw_tstat  # noqa: E402
from run_signal_search2 import resid_mom, risk_adj_mom, per_year, ev  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
HOLD_WEEKS = 104


def get_long_ensemble(week, init_train, step):
    cache = os.path.join(RESULTS_DIR, f"_wf_preds_long_{init_train}_{step}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"  ensemble largo cacheado: {cache}")
        return {int(w): d["mat"][i] for i, w in enumerate(d["weeks"])}
    print(f"  generando ensemble walk-forward (init_train={init_train}, step={step}) ...")
    preds = walkforward_signal(week, init_train, step)
    weeks = np.array(sorted(preds))
    mat = np.stack([preds[int(w)] for w in weeks])
    np.savez(cache, weeks=weeks, mat=mat)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--init_train", type=int, default=104)
    ap.add_argument("--step", type=int, default=26)
    args = ap.parse_args()

    print("Cargando panel + semanal ...")
    panel = dp.load_panel(args.data_dir)
    week = wp.build_weekly(panel)
    print(f"  {week.Xw.shape[0]} semanas x {week.Xw.shape[1]} acciones")

    ens = get_long_ensemble(week, args.init_train, args.step)
    weeks = sorted(ens)
    print(f"  ventana OOS: {len(weeks)} sem ({week.dates_w[weeks[0]].date()} "
          f"a {week.dates_w[weeks[-1]].date()}); holdout={HOLD_WEEKS}")

    dy = week.daily_y
    sig = {"ensemble": {}, "mom_resid": {}, "blend_mom_ra_resid": {},
           "ens_plus_momresid": {}, "ens_momresid_ra": {}}
    for wk in weeks:
        d = int(week.rebal_idx[wk])
        e = ens[wk]
        mrs = xz(resid_mom(dy, d))
        mra = xz(risk_adj_mom(dy, d))
        sig["ensemble"][wk] = e
        sig["mom_resid"][wk] = mrs
        sig["blend_mom_ra_resid"][wk] = np.nan_to_num(mrs, nan=0.0) + np.nan_to_num(mra, nan=0.0)
        sig["ens_plus_momresid"][wk] = np.nan_to_num(xz(e), nan=0.0) + np.nan_to_num(mrs, nan=0.0)
        sig["ens_momresid_ra"][wk] = (np.nan_to_num(xz(e), nan=0.0)
                                      + np.nan_to_num(mrs, nan=0.0)
                                      + np.nan_to_num(mra, nan=0.0))

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
        print(f"  {n:20s} full S={full['sharpe']:+.2f}(t={full['t']:+.2f}) "
              f"sel={sel['sharpe']:+.2f} hold={hold['sharpe']:+.2f}(t={hold['t']:+.2f}) "
              f"turn={bt['turnover'].mean():.2f} gross={sharpe(bt['gross']):+.2f} IC={bt['ic'].dropna().mean():+.4f}")
        print(f"      por año: {per_year(bt)}")

    df = pd.DataFrame(rows).sort_values("full_sharpe", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "signal_search3.csv"), index=False)
    print("\n" + "=" * 84)
    print(f"  RONDA 3 — ventana larga JUSTA ({len(weeks)} sem, incluye 2021)")
    print("=" * 84)
    print(df.to_string(index=False))
    print("=" * 84)
    b = df.iloc[0]
    ok = (b["full_t"] > 2) and (b["hold_sharpe"] > 0)
    print(f"\n  Mejor: {b['signal']} full S={b['full_sharpe']:+.2f} t={b['full_t']:+.2f} "
          f"hold={b['hold_sharpe']:+.2f} -> {'CRUZA EL LISTÓN' if ok else 'no cruza (t<2 o holdout<=0)'}")


if __name__ == "__main__":
    main()
