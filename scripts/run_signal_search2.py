#!/usr/bin/env python3
"""Ronda 2 de la búsqueda de señal: familia MOMENTUM bien hecha.

Motivación (de la ronda 1, results/signal_search.csv):
  - El reversal pierde incluso en bruto en large-cap semanal (resultado negativo).
  - El momentum 12-1 directo tiene IC 0,026 (8x el ensemble entrenado) y rotación
    bajísima (0,18), pero el 12-1 crudo no es robusto (holdout negativo: el
    "momentum crash"). La literatura corrige esto con momentum ajustado por riesgo
    y momentum residual (Blitz-Huij-Martens), de menor riesgo de cola.

Los factores NO se entrenan, así que la ventana honesta de evaluación es toda la
historia disponible (~7 años, ~360 semanas), no solo las 215 OOS del modelo de ML.
Se reserva un HOLDOUT final de 2 años (104 semanas) intacto y se desglosa por año.

PRE-REGISTRO ronda 2 (fijado antes de mirar resultados):
  * Hipótesis primaria: momentum residual ajustado por riesgo (mom_resid).
  * Selección: Sharpe neto en weeks[:-HOLD]; confirmación en holdout weeks[-HOLD:].
  * Éxito: t-stat Newey-West > 2 en el periodo completo Y holdout positivo.
  * Se reportan todas las variantes (gane o no).

Salida: results/signal_search2.csv

Uso:
    python scripts/run_signal_search2.py
    python scripts/run_signal_search2.py --smoke 30
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
from run_weekly_robustness import CONFIGS, backtest_variant, sharpe, ann, maxdd, sharpe_se  # noqa: E402
from run_signal_search import xz, cum_ret, nw_tstat  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
CACHE = os.path.join(RESULTS_DIR, "_wf_preds_base.npz")
MIN_HIST = 260      # días mínimos de historia para el formation de 12 meses
HOLD_WEEKS = 104    # holdout final (~2 años)


def risk_adj_mom(daily_y, d, lo=252, hi=21):
    """Momentum 12-1 ajustado por riesgo: cumret(formation) / vol(formation)."""
    a, b = d - lo, d - hi
    if a < 0:
        return np.full(daily_y.shape[1], np.nan)
    cr = cum_ret(daily_y, a, b)
    vol = np.nanstd(daily_y[a:b], axis=0)
    return np.where(vol > 1e-6, cr / vol, np.nan)


def resid_mom(daily_y, d, lo=252, hi=21):
    """Momentum residual: Sharpe de los retornos residuales (vs mercado) en el
    periodo de formación. Menor riesgo de crash (Blitz-Huij-Martens)."""
    a, b = d - lo, d - hi
    if a < 0:
        return np.full(daily_y.shape[1], np.nan)
    R = np.nan_to_num(daily_y[a:b], nan=0.0)          # [days, N]
    mkt = R.mean(axis=1)                               # equal-weight market
    mc = mkt - mkt.mean()
    var = float((mc ** 2).mean()) + 1e-12
    Rc = R - R.mean(axis=0, keepdims=True)
    beta = (Rc * mc.reshape(-1, 1)).mean(axis=0) / var # [N]
    resid = R - np.outer(mkt, beta)                    # [days, N]
    mu = resid.mean(axis=0)
    sd = resid.std(axis=0)
    return np.where(sd > 1e-9, mu / sd, np.nan)


def build(week, weeks, ensemble_preds, names):
    dy = week.daily_y
    N = week.Xw.shape[1]
    out = {n: {} for n in names}
    for wk in weeks:
        d = int(week.rebal_idx[wk])
        m121 = xz(cum_ret(dy, d - 252, d - 21))
        m61 = xz(cum_ret(dy, d - 126, d - 21))
        mra = xz(risk_adj_mom(dy, d))
        mrs = xz(resid_mom(dy, d))
        cand = {
            "mom_12_1": m121,
            "mom_6_1": m61,
            "mom_riskadj": mra,
            "mom_resid": mrs,
            "blend_mom_ra_resid": _avg([mra, mrs]),
            "ens_plus_momresid": _avg([xz(ensemble_preds.get(wk)), mrs])
            if ensemble_preds.get(wk) is not None else None,
        }
        for n in names:
            out[n][wk] = cand[n]
    return out


def _avg(arrs):
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return None
    return np.vstack([np.nan_to_num(a, nan=0.0) for a in arrs]).mean(axis=0)


def per_year(bt):
    by = bt.copy()
    by["year"] = pd.DatetimeIndex(by.index).year
    return {int(y): round(sharpe(g["net"]), 2) for y, g in by.groupby("year")}


def ev(net):
    n = int(net.notna().sum())
    if n < 5:
        return dict(n=n, sharpe=float("nan"), se=float("nan"),
                    t=float("nan"), ann=float("nan"), mdd=float("nan"))
    return dict(n=n, sharpe=sharpe(net), se=sharpe_se(sharpe(net), n),
                t=nw_tstat(net), ann=ann(net), mdd=maxdd(net))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    print("Cargando panel + semanal ...")
    panel = dp.load_panel(args.data_dir)
    week = wp.build_weekly(panel)
    W = week.Xw.shape[0]
    print(f"  {W} semanas x {week.Xw.shape[1]} acciones")

    ensemble_preds = {}
    if os.path.exists(CACHE):
        dd = np.load(CACHE)
        ensemble_preds = {int(w): dd["mat"][i] for i, w in enumerate(dd["weeks"])}

    weeks = [wk for wk in range(1, W) if week.rebal_idx[wk] >= MIN_HIST]
    if args.smoke:
        weeks = weeks[:args.smoke]
    names = ["mom_12_1", "mom_6_1", "mom_riskadj", "mom_resid",
             "blend_mom_ra_resid", "ens_plus_momresid"]
    if args.smoke:
        names = ["mom_12_1", "mom_resid"]
    print(f"  evaluación: {len(weeks)} semanas (desde {week.dates_w[weeks[0]].date()} "
          f"a {week.dates_w[weeks[-1]].date()}); holdout={HOLD_WEEKS} sem")

    sig = build(week, weeks, ensemble_preds, names)
    hold_set = set(weeks[-HOLD_WEEKS:])

    rows = []
    for n in names:
        preds = {wk: sig[n][wk] for wk in weeks if sig[n][wk] is not None}
        if not preds:
            continue
        bt = backtest_variant(week, preds, CONFIGS["full"]).copy()
        wk_of = {week.dates_w[wk]: wk for wk in preds}
        is_hold = np.array([wk_of.get(dt, -1) in hold_set for dt in bt.index])
        full, sel, hold = ev(bt["net"]), ev(bt["net"][~is_hold]), ev(bt["net"][is_hold])
        rows.append(dict(signal=n, full_sharpe=full["sharpe"], full_t=full["t"],
                         sel_sharpe=sel["sharpe"], hold_sharpe=hold["sharpe"],
                         hold_t=hold["t"], full_ann=full["ann"], full_mdd=full["mdd"],
                         turnover=float(bt["turnover"].mean()),
                         gross=sharpe(bt["gross"]), ic=float(bt["ic"].dropna().mean())))
        print(f"  {n:20s} full S={full['sharpe']:+.2f} (t={full['t']:+.2f}) "
              f"sel={sel['sharpe']:+.2f} hold={hold['sharpe']:+.2f}(t={hold['t']:+.2f}) "
              f"turn={bt['turnover'].mean():.2f} gross={sharpe(bt['gross']):+.2f} "
              f"IC={bt['ic'].dropna().mean():+.4f}")
        print(f"      por año: {per_year(bt)}")

    df = pd.DataFrame(rows).sort_values("full_sharpe", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, "signal_search2.csv"), index=False)
    print("\n" + "=" * 80)
    print("  LEADERBOARD ronda 2 (momentum) — base TFM 215sem = +0.58")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    if len(df):
        b = df.iloc[0]
        ok = (b["full_t"] > 2) and (b["hold_sharpe"] > 0)
        print(f"\n  Mejor: {b['signal']}  full S={b['full_sharpe']:+.2f} t={b['full_t']:+.2f} "
              f"holdout={b['hold_sharpe']:+.2f}  -> {'CRUZA el listón' if ok else 'NO cruza el listón'}")


if __name__ == "__main__":
    main()
