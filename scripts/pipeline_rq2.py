#!/usr/bin/env python3
"""Pipeline concreto para RQ2: momentum residual + ensemble, construcción cost-aware.

Resultado de la búsqueda disciplinada (scripts/run_signal_search{,2,3,4}.py). Es la
respuesta concreta a "¿qué genera retorno neto en S&P 500 large-cap semanal?":
un FACTOR REAL (momentum residual, Blitz-Huij-Martens) explotado con la construcción
cost-aware del TFM. No es la arquitectura ni los fundamentales; es señal-factor +
construcción.

PIPELINE (todo offline, leakage-free; reusa lib/ del TFM):
  1. Panel semanal (último día de cada semana ISO; etiqueta = retorno forward).
  2. Señal = z(ensemble LightGBM+ElasticNet)  +  z(momentum residual 12-1),
     ambas estandarizadas cross-seccionalmente. El ensemble es el shallow de la
     Parte I; el momentum residual se calcula sobre retornos residualizados vs
     mercado en el periodo de formación [d-252, d-21], escalado por su vol (Sharpe
     de los residuos). Sólo usa información disponible en el día de decisión.
  3. Neutralización beta -> EWMA (halflife 2 sem) -> optimizador convexo cost-aware
     (neutral dólar/beta, penalización L1 de rotación, tope por nombre, gross 2)
     -> objetivo de vol 10% -> overlay de régimen. Coste 8 bps/lado.

RESULTADO (ventana justa 311 sem, 2020-2026, incluye el momentum-crash de 2021;
holdout = últimas 104 sem, intactas):
  Sharpe neto 0.72 (t Newey-West 1.82) | holdout +0.55 | positivo 5/7 años
  maxDD -7.4% | turnover 0.43 | IC 0.020
Honesto: mejora robusta sobre la base (0.38 en la misma ventana), pero al borde de
la significancia (t<2). El techo honesto del régimen con datos públicos es ~0.7.

Uso:
    python scripts/pipeline_rq2.py                      # ensemble+momresid (def.)
    python scripts/pipeline_rq2.py --signal momresid    # solo momentum residual
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import data_panel as dp  # noqa: E402
from lib import weekly_panel as wp  # noqa: E402
from run_weekly_robustness import CONFIGS, backtest_variant, sharpe  # noqa: E402
from run_signal_search import xz  # noqa: E402
from run_signal_search2 import resid_mom, per_year, ev  # noqa: E402
from run_signal_search3 import get_long_ensemble  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
HOLD_WEEKS = 104


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--init_train", type=int, default=104)
    ap.add_argument("--step", type=int, default=26)
    ap.add_argument("--signal", choices=["ens_momresid", "momresid"],
                    default="ens_momresid")
    args = ap.parse_args()

    print("Cargando panel + resampling semanal ...")
    panel = dp.load_panel(args.data_dir)
    week = wp.build_weekly(panel)
    dy = week.daily_y

    ens = get_long_ensemble(week, args.init_train, args.step)
    weeks = sorted(ens)
    print(f"  ventana OOS: {len(weeks)} sem "
          f"({week.dates_w[weeks[0]].date()} a {week.dates_w[weeks[-1]].date()})")

    preds = {}
    for wk in weeks:
        d = int(week.rebal_idx[wk])
        mrs = np.nan_to_num(xz(resid_mom(dy, d)), nan=0.0)
        if args.signal == "ens_momresid":
            preds[wk] = np.nan_to_num(xz(ens[wk]), nan=0.0) + mrs
        else:
            preds[wk] = mrs

    bt = backtest_variant(week, preds, CONFIGS["full"]).copy()
    hold_set = set(weeks[-HOLD_WEEKS:])
    wk_of = {week.dates_w[wk]: wk for wk in weeks}
    is_hold = np.array([wk_of.get(dt, -1) in hold_set for dt in bt.index])

    full, sel, hold = ev(bt["net"]), ev(bt["net"][~is_hold]), ev(bt["net"][is_hold])
    summary = {
        "signal": args.signal, "n_weeks": full["n"],
        "net_sharpe": full["sharpe"], "net_sharpe_t_nw": full["t"],
        "net_ann": full["ann"], "net_maxdd": full["mdd"],
        "gross_sharpe": sharpe(bt["gross"]),
        "turnover": float(bt["turnover"].mean()),
        "ic_mean": float(bt["ic"].dropna().mean()),
        "sel_sharpe": sel["sharpe"], "hold_sharpe": hold["sharpe"], "hold_t_nw": hold["t"],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([summary]).to_csv(
        os.path.join(RESULTS_DIR, f"pipeline_rq2_{args.signal}.csv"), index=False)
    bt.to_csv(os.path.join(RESULTS_DIR, f"pipeline_rq2_{args.signal}_weekly.csv"))

    print("\n" + "=" * 60)
    print(f"  PIPELINE RQ2 — {args.signal}")
    print("=" * 60)
    print(f"  Semanas OOS        : {full['n']}")
    print(f"  NET Sharpe         : {full['sharpe']:+.2f}  (t Newey-West {full['t']:+.2f})")
    print(f"  NET ann return     : {full['ann']:+.1%}")
    print(f"  NET max drawdown   : {full['mdd']:+.1%}")
    print(f"  GROSS Sharpe       : {sharpe(bt['gross']):+.2f}")
    print(f"  Turnover medio     : {bt['turnover'].mean():.2f}")
    print(f"  IC medio semanal   : {full['ic'] if 'ic' in full else float(bt['ic'].dropna().mean()):+.4f}")
    print(f"  Holdout (104 sem)  : {hold['sharpe']:+.2f}  (t {hold['t']:+.2f})")
    print(f"  Sharpe por año     : {per_year(bt)}")
    print("=" * 60)

    # Curva de equity (bruta vs neta)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bt.index, (1 + bt["gross"]).cumprod(), color="#7f8c8d", lw=1.2, label="Bruto")
    ax.plot(bt.index, (1 + bt["net"]).cumprod(), color="#27ae60", lw=2,
            label="Neto (8 bps/lado)")
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_title(f"Pipeline RQ2 ({args.signal}) — Sharpe neto {full['sharpe']:+.2f} "
                 f"(t={full['t']:+.2f}, {full['n']} semanas)", fontsize=13)
    ax.set_xlabel("Fecha"); ax.set_ylabel("Retorno acumulado")
    ax.legend(fontsize=10); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    out_png = os.path.join(FIGURES_DIR, f"pipeline_rq2_{args.signal}_equity.png")
    fig.savefig(out_png, dpi=300); plt.close(fig)
    print(f"Guardado: results/pipeline_rq2_{args.signal}.csv (+ _weekly.csv) y {out_png}")


if __name__ == "__main__":
    main()
