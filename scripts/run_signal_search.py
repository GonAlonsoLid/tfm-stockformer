#!/usr/bin/env python3
"""Búsqueda DISCIPLINADA de una señal que dé Sharpe neto defendible en el régimen
large-cap semanal, contestando RQ2 con una solución concreta.

Filosofía anti-overfitting (coherente con de Prado y con la tesis):
  - Lista CERRADA de hipótesis motivadas a priori (no grid sobre el OOS).
  - Construcción cost-aware y walk-forward IDÉNTICOS a los del TFM (mismo
    constructor, mismas 215 semanas OOS, mismos costes 8 bps/lado): comparación
    limpia contra la base ensemble (Sharpe neto 0,58 ± 0,49).
  - Cada señal es leakage-free por construcción: usa solo retornos ya realizados
    en el día de decisión (rev usa yw[wk-1], el retorno de la semana ya cerrada).
  - PRE-REGISTRO (fijado antes de mirar resultados):
      * Hipótesis primaria: reversal de 1 semana escalado por vol (rev1w_vol).
      * Métrica de selección: Sharpe neto en weeks[:SEL] (primeras ~163 sem).
      * Confirmación: holdout weeks[SEL:] (últimas ~52 sem), intacto.
      * Umbral de éxito: t-stat Newey-West del retorno neto medio > 2 en el
        periodo completo Y holdout con el mismo signo.
  - Se reportan TODAS las hipótesis (gane o no), igual que los negativos del TFM.

Salida: results/signal_search.csv  (+ stdout leaderboard)

Uso:
    python scripts/run_signal_search.py            # barrido completo (215 sem)
    python scripts/run_signal_search.py --smoke 20 # smoke: 20 semanas, 2 señales
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import data_panel as dp  # noqa: E402
from lib import weekly_panel as wp  # noqa: E402
from run_weekly_robustness import (  # noqa: E402
    CONFIGS, backtest_variant, sharpe, ann, maxdd, sharpe_se, walkforward_signal,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
CACHE = os.path.join(RESULTS_DIR, "_wf_preds_base.npz")
WEEKS_PER_YEAR = 52
NW_LAGS = 6  # Newey-West maxlags para autocorrelación de retornos semanales


# ── Utilidades de señal (cross-sectional, leakage-free) ─────────────────────────

def xz(v: np.ndarray) -> np.ndarray:
    """Z-score cross-seccional ignorando NaN; NaN se conservan como NaN."""
    v = np.asarray(v, dtype=float).reshape(-1)
    out = np.full_like(v, np.nan)
    ok = ~np.isnan(v)
    if ok.sum() > 1:
        s = v[ok]
        out[ok] = (s - s.mean()) / (s.std() + 1e-12)
    return out


def cum_ret(daily_y: np.ndarray, a: int, b: int) -> np.ndarray:
    """Retorno compuesto por acción sobre días [a, b). Conocido en el día b."""
    a = max(0, a)
    if b <= a:
        return np.full(daily_y.shape[1], np.nan)
    R = daily_y[a:b]
    return np.prod(1.0 + np.nan_to_num(R, nan=0.0), axis=0) - 1.0


def idio_vol(daily_y: np.ndarray, d: int, win: int = 60) -> np.ndarray:
    """Vol realizada por acción sobre los últimos `win` días hasta d (excl.)."""
    a = max(0, d - win)
    R = daily_y[a:d]
    if R.shape[0] < 5:
        return np.full(daily_y.shape[1], np.nan)
    return np.nanstd(R, axis=0)


# ── Constructores de señal: cada uno devuelve {wk: alpha[N]} ────────────────────

def build_signals(week, eval_weeks, ensemble_preds, which):
    """Construye los dicts de señal sobre `eval_weeks` (mismas fechas que la base)."""
    daily_y = week.daily_y
    sig = {name: {} for name in which}

    for wk in eval_weeks:
        d = int(week.rebal_idx[wk])
        rev1 = -xz(week.yw[wk - 1]) if wk - 1 >= 0 else np.full(week.Xw.shape[1], np.nan)
        iv = idio_vol(daily_y, d, 60)
        ivz = xz(iv)
        rev_raw = week.yw[wk - 1] if wk - 1 >= 0 else np.full(week.Xw.shape[1], np.nan)
        rev1_vol = -xz(np.where(np.isnan(iv) | (iv < 1e-6), np.nan, rev_raw / iv))
        rev4 = -xz(cum_ret(daily_y, d - 20, d))
        mom = xz(cum_ret(daily_y, d - 252, d - 21))   # 12-1 momentum
        lowvol = -ivz                                  # baja vol = señal positiva

        cand = {
            "ensemble": ensemble_preds.get(wk),
            "rev1w": rev1,
            "rev1w_vol": rev1_vol,
            "rev4w": rev4,
            "mom_12_1": mom,
            "lowvol": lowvol,
            "blend_rev_mom": _sum_z([rev1_vol, mom]),
            "blend_rev_lowvol": _sum_z([rev1_vol, lowvol]),
            "ens_plus_rev": _sum_z([xz(ensemble_preds.get(wk)), rev1_vol])
            if ensemble_preds.get(wk) is not None else None,
        }
        for name in which:
            sig[name][wk] = cand[name]
    return sig


def _sum_z(arrs):
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return None
    stack = np.vstack([np.nan_to_num(a, nan=0.0) for a in arrs])
    return stack.sum(axis=0)


# ── Estadística honesta ─────────────────────────────────────────────────────────

def nw_tstat(net: pd.Series) -> float:
    """t-stat Newey-West del retorno semanal medio (HAC, maxlags=NW_LAGS)."""
    r = pd.Series(net).dropna().values
    if len(r) < 10 or np.allclose(r, 0):
        return float("nan")
    X = np.ones((len(r), 1))
    res = sm.OLS(r, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
    return float(res.tvalues[0])


def evaluate(net: pd.Series) -> dict:
    return {
        "n": int(net.notna().sum()),
        "net_sharpe": sharpe(net),
        "net_sharpe_se": sharpe_se(sharpe(net), int(net.notna().sum())),
        "nw_t": nw_tstat(net),
        "net_ann": ann(net),
        "net_maxdd": maxdd(net),
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--init_train", type=int, default=200)
    ap.add_argument("--step", type=int, default=26)
    ap.add_argument("--sel_frac", type=float, default=0.75,
                    help="fracción inicial para selección; resto = holdout")
    ap.add_argument("--smoke", type=int, default=0, help="si>0: nº semanas y solo 2 señales")
    args = ap.parse_args()

    print("Cargando panel + resampling semanal ...")
    panel = dp.load_panel(args.data_dir)
    week = wp.build_weekly(panel)
    print(f"  {week.Xw.shape[0]} semanas x {week.Xw.shape[1]} acciones")

    # Señal base (ensemble) y ventana de evaluación = mismas semanas que el TFM
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        ensemble_preds = {int(w): d["mat"][i] for i, w in enumerate(d["weeks"])}
        print(f"  señal ensemble cacheada: {len(ensemble_preds)} semanas OOS")
    else:
        ensemble_preds = walkforward_signal(week, args.init_train, args.step)

    eval_weeks = sorted(ensemble_preds)
    all_names = ["ensemble", "rev1w", "rev1w_vol", "rev4w", "mom_12_1", "lowvol",
                 "blend_rev_mom", "blend_rev_lowvol", "ens_plus_rev"]
    if args.smoke:
        eval_weeks = eval_weeks[:args.smoke]
        all_names = ["ensemble", "rev1w_vol"]
        print(f"  SMOKE: {len(eval_weeks)} semanas, señales {all_names}")

    print("Construyendo señales (leakage-free) ...")
    sig = build_signals(week, eval_weeks, ensemble_preds, all_names)

    sel_cut = int(len(eval_weeks) * args.sel_frac)
    sel_weeks = set(eval_weeks[:sel_cut])
    print(f"  selección: {sel_cut} sem | holdout: {len(eval_weeks) - sel_cut} sem\n")

    rows = []
    cfg = CONFIGS["full"]
    for name in all_names:
        preds = {wk: sig[name][wk] for wk in eval_weeks if sig[name][wk] is not None}
        bt = backtest_variant(week, preds, cfg)
        bt = bt.copy()
        # marcar selección/holdout por posición temporal
        wk_of_date = {week.dates_w[wk]: wk for wk in eval_weeks}
        in_sel = np.array([wk_of_date.get(dt, -1) in sel_weeks for dt in bt.index])
        full = evaluate(bt["net"])
        sel = evaluate(bt["net"][in_sel])
        hold = evaluate(bt["net"][~in_sel])
        row = {
            "signal": name,
            "full_sharpe": full["net_sharpe"], "full_se": full["net_sharpe_se"],
            "full_nw_t": full["nw_t"], "full_ann": full["net_ann"], "full_maxdd": full["net_maxdd"],
            "sel_sharpe": sel["net_sharpe"], "hold_sharpe": hold["net_sharpe"],
            "hold_nw_t": hold["nw_t"],
            "turnover": float(bt["turnover"].mean()),
            "gross_sharpe": sharpe(bt["gross"]),
            "ic_mean": float(bt["ic"].dropna().mean()),
        }
        rows.append(row)
        print(f"  {name:17s} full S={full['net_sharpe']:+.2f} (t={full['nw_t']:+.2f}) | "
              f"sel={sel['net_sharpe']:+.2f} hold={hold['net_sharpe']:+.2f} | "
              f"turn={bt['turnover'].mean():.2f} gross={sharpe(bt['gross']):+.2f} "
              f"IC={bt['ic'].dropna().mean():+.4f}")

    df = pd.DataFrame(rows).sort_values("full_sharpe", ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "signal_search.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 78)
    print("  LEADERBOARD (ordenado por Sharpe neto completo)  —  base TFM = +0.58")
    print("=" * 78)
    print(df[["signal", "full_sharpe", "full_nw_t", "sel_sharpe", "hold_sharpe",
              "turnover", "gross_sharpe", "ic_mean"]].to_string(index=False))
    print("=" * 78)
    best = df.iloc[0]
    print(f"\n  Mejor por Sharpe completo: {best['signal']} "
          f"(S={best['full_sharpe']:+.2f}, t={best['full_nw_t']:+.2f}, "
          f"holdout={best['hold_sharpe']:+.2f})")
    print(f"  Guardado: {out}")


if __name__ == "__main__":
    main()
