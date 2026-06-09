#!/usr/bin/env python3
"""Estudio de la capa de construccion: frontera coste-Sharpe (A) + termino de riesgo (B).

La atribucion del TFM localiza el valor en la construccion cost-aware, no en la senal.
Este script profundiza esa capa sobre el MISMO walk-forward de 215 semanas y la MISMA
senal (ensemble constante 0,5/0,5, cacheada), variando solo parametros de construccion.

A. Mapa del espacio de construccion (descriptivo): sensibilidad del Sharpe neto a
   cost_bps, gross, name_cap, smooth_halflife y target_vol, con break-even de coste.

B. Termino de riesgo: activa `risk_aversion * w' Sigma w` en el optimizador, con Sigma
   = covarianza Ledoit-Wolf de los retornos diarios de la ventana de 60 dias. Barre
   risk_aversion y reporta Sharpe neto / MaxDD.

Pre-registro: docs/tesis/PREREGISTRO_PARS_US.md (Parte II).

Salidas:
    results/construction_frontier.csv, results/construction_risk.csv
    MEMORIA/tfm/tablas/construction_cost.tex, construction_risk.tex
    results/figures/construction_frontier.png, construction_risk.png

Uso:
    python scripts/run_construction.py --init_train 200 --step 26
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
from lib import neutralize as nz  # noqa: E402
from lib import portfolio as pf  # noqa: E402
from lib import weekly_panel as wp  # noqa: E402
from run_weekly_robustness import walkforward_signal, sharpe, ann, maxdd, sharpe_se  # noqa: E402
from run_weekly_strategy import regime_scale  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "MEMORIA", "tfm", "tablas")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
BETA_WINDOW = 60

DEFAULTS = dict(cost_bps=8.0, gross=2.0, name_cap=0.04, target_vol=0.10, halflife=2.0)
SWEEPS = {
    "cost_bps": [0.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0],
    "gross": [1.0, 1.5, 2.0, 2.5, 3.0],
    "name_cap": [0.02, 0.03, 0.04, 0.06, 0.10],
    "halflife": [1.0, 2.0, 3.0, 4.0],
    "target_vol": [0.06, 0.08, 0.10, 0.12, 0.15],
}
RISK_GRID = [0.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]


# ── Construccion parametrizada (config `full`, con termino de riesgo opcional) ────

def construct(week, preds, *, cost_bps, gross, name_cap, target_vol, halflife,
              risk_aversion=0.0, sigma_by_week=None):
    market_daily = np.nanmean(week.daily_y, axis=1)
    ewma_a = 1 - 0.5 ** (1 / halflife)
    w_prev = np.zeros(week.Xw.shape[1])
    smoothed = None
    rows = []
    zero_books = 0
    for wk in sorted(preds):
        d = week.rebal_idx[wk]
        alpha = preds[wk]
        if np.all(np.isnan(alpha)):
            continue
        win = slice(max(0, d - BETA_WINDOW), d)
        beta = pf.rolling_beta(week.daily_y[win], market_daily[win])
        sig = np.nan_to_num(nz.neutralize(alpha, beta.reshape(-1, 1)), nan=0.0)
        smoothed = sig if smoothed is None else ewma_a * sig + (1 - ewma_a) * smoothed
        Sigma = None
        if risk_aversion > 0 and sigma_by_week is not None:
            Sigma = sigma_by_week.get(wk)
        w = pf.costaware_weights(smoothed, beta, Sigma=Sigma, w_prev=w_prev, gross=gross,
                                 name_cap=name_cap, cost_bps=cost_bps,
                                 risk_aversion=risk_aversion, beta_neutral=True)
        if not np.any(np.abs(w) > 1e-9):
            zero_books += 1
        w = w * pf.vol_target_scale(week.daily_y[win], w, target_ann_vol=target_vol)
        w = w * regime_scale(market_daily, d)
        realized = np.nan_to_num(week.yw[wk], nan=0.0)
        gross_r = float(w @ realized)
        turnover = float(np.abs(w - w_prev).sum())
        net = gross_r - turnover * (cost_bps / 1e4)
        rows.append({"date": week.dates_w[wk], "gross": gross_r, "net": net, "turnover": turnover})
        w_prev = w
    return pd.DataFrame(rows).set_index("date"), zero_books


def metrics(df: pd.DataFrame, zero_books: int = 0) -> dict:
    n = len(df)
    ns = sharpe(df["net"])
    return {
        "n_weeks": n, "net_sharpe": ns, "net_sharpe_se": sharpe_se(ns, n),
        "gross_sharpe": sharpe(df["gross"]), "net_ann": ann(df["net"]),
        "net_maxdd": maxdd(df["net"]), "turnover": float(df["turnover"].mean()),
        "zero_books": zero_books,
    }


# ── Senal walk-forward (cacheada) y covarianzas (Ledoit-Wolf) ─────────────────────

def get_preds(week, init_train: int, step: int, cache: str) -> dict:
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"  senal walk-forward cacheada: {cache}")
        return {int(w): d["mat"][i] for i, w in enumerate(d["weeks"])}
    preds = walkforward_signal(week, init_train, step)
    weeks = np.array(sorted(preds))
    mat = np.stack([preds[int(w)] for w in weeks])
    np.savez(cache, weeks=weeks, mat=mat)
    return preds


def precompute_sigma(week, preds) -> dict:
    from sklearn.covariance import LedoitWolf
    out = {}
    for wk in sorted(preds):
        d = week.rebal_idx[wk]
        R = np.nan_to_num(week.daily_y[slice(max(0, d - BETA_WINDOW), d)], nan=0.0)
        if R.shape[0] < 10:
            out[wk] = None
            continue
        out[wk] = LedoitWolf(assume_centered=False).fit(R).covariance_
    return out


# ── Tablas LaTeX ──────────────────────────────────────────────────────────────────

def write_cost_table(rows: list[dict]) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    be = next((r["value"] for r in rows if r["net_sharpe"] <= 0), None)
    be_txt = (f"El Sharpe neto cruza cero en torno a {be:.0f} bps/lado."
              if be is not None else "El Sharpe neto se mantiene positivo en todo el rango.")
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Sensibilidad del Sharpe neto (walk-forward, 215 semanas) al coste de "
        r"transacci\'on, manteniendo el resto de la construcci\'on en su valor por defecto. "
        + be_txt + r"}",
        r"\label{tab:construction-cost}", r"\small",
        r"\begin{tabular}{rrrr}", r"\toprule",
        r"Coste (bps/lado) & Sharpe neto & Sharpe bruto & Turnover \\", r"\midrule",
    ]
    for r in rows:
        lines.append(f"{r['value']:.0f} & ${r['net_sharpe']:+.2f} \\pm "
                     f"{r['net_sharpe_se']:.2f}$ & {r['gross_sharpe']:+.2f} & {r['turnover']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(TABLES_DIR, "construction_cost.tex"), "w") as f:
        f.write("\n".join(lines))


def write_risk_table(rows: list[dict]) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    base = rows[0]
    best = max(rows, key=lambda r: r["net_sharpe"])
    delta = best["net_sharpe"] - base["net_sharpe"]
    verdict = ("mejora $>1$\\,SE" if delta > base["net_sharpe_se"]
               else "no distinguible de cero ($<1$\\,SE), como se pre-registr\\'o")
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{T\'ermino de riesgo (covarianza Ledoit-Wolf) en el optimizador "
        r"cost-aware: Sharpe neto y MaxDD por aversi\'on al riesgo $\lambda$ "
        r"(walk-forward, 215 semanas). $\lambda=0$ es el \emph{baseline}. Resultado: "
        + verdict + r".}",
        r"\label{tab:construction-risk}", r"\small",
        r"\begin{tabular}{rrrr}", r"\toprule",
        r"$\lambda$ & Sharpe neto & MaxDD & Turnover \\", r"\midrule",
    ]
    for r in rows:
        lines.append(f"{r['risk_aversion']:.0f} & ${r['net_sharpe']:+.2f} \\pm "
                     f"{r['net_sharpe_se']:.2f}$ & {r['net_maxdd']:+.1%} & "
                     f"{r['turnover']:.2f} \\\\".replace("%", r"\%"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(os.path.join(TABLES_DIR, "construction_risk.tex"), "w") as f:
        f.write("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--init_train", type=int, default=200)
    ap.add_argument("--step", type=int, default=26)
    args = ap.parse_args()

    print("Cargando panel + resampling semanal ...")
    panel = dp.load_panel(args.data_dir)
    week = wp.build_weekly(panel)
    print(f"  {week.Xw.shape[0]} semanas x {week.Xw.shape[1]} acciones")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    cache = os.path.join(RESULTS_DIR, "_wf_preds_base.npz")
    print("Senal walk-forward (ensemble constante 0,5/0,5) ...")
    preds = get_preds(week, args.init_train, args.step, cache)

    # ── A. Frontera / sensibilidad de construccion ──
    print("\nA. Mapa del espacio de construccion:")
    frontier = []
    for knob, values in SWEEPS.items():
        for v in values:
            params = dict(DEFAULTS)
            params[knob] = v
            df, _ = construct(week, preds, **params)
            m = metrics(df)
            frontier.append({"knob": knob, "value": v, **m})
            print(f"  {knob:11s}={v:<6} net Sharpe={m['net_sharpe']:+.2f}±{m['net_sharpe_se']:.2f} "
                  f"gross={m['gross_sharpe']:+.2f} turn={m['turnover']:.2f}")
    pd.DataFrame(frontier).to_csv(os.path.join(RESULTS_DIR, "construction_frontier.csv"), index=False)
    cost_rows = [r for r in frontier if r["knob"] == "cost_bps"]
    write_cost_table(cost_rows)

    # ── B. Termino de riesgo (Ledoit-Wolf) ──
    print("\nB. Termino de riesgo (Ledoit-Wolf), precomputando covarianzas ...")
    sigma_by_week = precompute_sigma(week, preds)
    risk_rows = []
    for lam in RISK_GRID:
        params = dict(DEFAULTS)
        df, zb = construct(week, preds, risk_aversion=lam, sigma_by_week=sigma_by_week, **params)
        m = metrics(df, zb)
        risk_rows.append({"risk_aversion": lam, **m})
        print(f"  lambda={lam:<7} net Sharpe={m['net_sharpe']:+.2f}±{m['net_sharpe_se']:.2f} "
              f"MaxDD={m['net_maxdd']:+.1%} gross={m['gross_sharpe']:+.2f} zero_books={zb}")
    pd.DataFrame(risk_rows).to_csv(os.path.join(RESULTS_DIR, "construction_risk.csv"), index=False)
    write_risk_table(risk_rows)

    base_sh = risk_rows[0]["net_sharpe"]
    best = max(risk_rows, key=lambda r: r["net_sharpe"])
    se = risk_rows[0]["net_sharpe_se"]
    print(f"\n  baseline (lambda=0) Sharpe={base_sh:+.2f}; mejor lambda={best['risk_aversion']:.0f} "
          f"Sharpe={best['net_sharpe']:+.2f} (delta={best['net_sharpe']-base_sh:+.2f}, 1 SE={se:.2f})")

    # ── Figuras ──
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, knob in zip(axes.flat, SWEEPS):
        rs = [r for r in frontier if r["knob"] == knob]
        xs = [r["value"] for r in rs]
        ys = [r["net_sharpe"] for r in rs]
        es = [r["net_sharpe_se"] for r in rs]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=4, color="#c0392b")
        ax.axhline(0, color="black", lw=0.8, ls=":")
        ax.axvline(DEFAULTS[knob], color="#7f8c8d", lw=1, ls="--")
        ax.set_title(knob); ax.set_ylabel("Sharpe neto"); ax.grid(True, ls="--", alpha=0.4)
    # 6th panel: termino de riesgo
    ax = axes.flat[5]
    xs = [r["risk_aversion"] for r in risk_rows]
    ys = [r["net_sharpe"] for r in risk_rows]
    es = [r["net_sharpe_se"] for r in risk_rows]
    ax.errorbar(range(len(xs)), ys, yerr=es, marker="s", capsize=4, color="#2c3e50")
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:.0f}" for x in xs], rotation=30)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_title("risk_aversion (Ledoit-Wolf)"); ax.set_ylabel("Sharpe neto")
    ax.grid(True, ls="--", alpha=0.4)
    fig.suptitle("Espacio de construccion: sensibilidad del Sharpe neto (215 semanas)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "construction_frontier.png"), dpi=300)
    plt.close(fig)
    print("Guardado: construction_frontier.csv/.tex, construction_risk.csv/.tex, figura.")


if __name__ == "__main__":
    main()
