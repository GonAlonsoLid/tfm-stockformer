#!/usr/bin/env python3
"""PARS-US: predict-shallow, construct-smart, condition-on-regime.

Contribucion constructiva del TFM, inversa a Stockformer: el unico objeto aprendido
nuevo es la COMBINACION de los dos aprendices base (LightGBM + ElasticNet),
condicionada por el regimen de volatilidad (VIX), en lugar del blend constante
0,5/0,5. Tres grados de libertad (un peso por regimen), CONGELADOS tras estimarse
una sola vez sobre el OOF del tramo de entrenamiento inicial.

Compara, sobre el MISMO walk-forward y la MISMA construccion `full` cost-aware:
    base   = blend constante 0,5/0,5            (reproduce el pipeline actual)
    parsus = blend por regimen (low/normal/high)

Criterio pre-registrado (docs/tesis/PREREGISTRO_PARS_US.md): se declara mejora solo
si Sharpe_neto(parsus) - Sharpe_neto(base) > 1 SE (~0,49). El resultado esperado y
aceptado es el NULO.

Salidas:
    results/pars_us_summary.csv          (base vs parsus)
    results/pars_us_regime_blend.csv     (pesos por regimen + ICIR por regimen)
    MEMORIA/tfm/tablas/pars_us.tex
    results/figures/pars_us_equity.png

Uso:
    python scripts/run_pars_us.py --init_train 200 --step 26
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import data_panel as dp  # noqa: E402
from lib import weekly_panel as wp  # noqa: E402
from run_weekly_robustness import (  # noqa: E402
    CONFIGS, backtest_variant, sharpe, ann, maxdd, sharpe_se,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "MEMORIA", "tfm", "tablas")
DEFAULT_DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
REGIMES = ("low", "normal", "high")
BLEND_GRID = tuple(round(0.1 * i, 1) for i in range(11))  # 0.0 ... 1.0


@dataclass(frozen=True)
class ParsConfig:
    """Hiperparametros del blend por regimen (pre-registrados, no se ajustan a OOS)."""
    inner_frac: float = 0.3
    embargo: int = 1
    min_weeks: int = 8
    seed: int = 0


# ── Aprendices base (identicos a train_ensemble, pero exponiendo componentes) ─────

def _z(v: np.ndarray) -> np.ndarray:
    m = ~np.isnan(v)
    out = np.full_like(v, np.nan, dtype=float)
    if m.sum() > 1:
        s = v[m]
        out[m] = (s - s.mean()) / (s.std() + 1e-12)
    return out


def _train_learners(Xw: np.ndarray, yw: np.ndarray, train_end_w: int, seed: int):
    import lightgbm as lgb
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler

    F = Xw.shape[2]
    tr = np.arange(train_end_w)
    Xtr = Xw[tr].reshape(-1, F)
    ytr = yw[tr].reshape(-1)
    ok = ~np.isnan(ytr) & ~np.isnan(Xtr).any(axis=1)
    Xtr, ytr = Xtr[ok], ytr[ok]
    scaler = StandardScaler().fit(Xtr)
    gbm = lgb.LGBMRegressor(
        objective="huber", n_estimators=300, learning_rate=0.02, num_leaves=31,
        subsample=0.7, colsample_bytree=0.5, min_child_samples=100,
        reg_alpha=0.1, reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1,
    ).fit(Xtr, ytr)
    enet = ElasticNet(alpha=1e-4, l1_ratio=0.5, max_iter=5000, random_state=seed).fit(
        scaler.transform(Xtr), ytr
    )
    return gbm, enet, scaler


def _components(models, Xw: np.ndarray, test_w: np.ndarray):
    """Devuelve (Zg, Ze): predicciones estandarizadas por semana de cada aprendiz."""
    gbm, enet, scaler = models
    N = Xw.shape[1]
    Zg = np.full((len(test_w), N), np.nan)
    Ze = np.full((len(test_w), N), np.nan)
    for k, wk in enumerate(test_w):
        x = Xw[wk]
        valid = ~np.isnan(x).any(axis=1)
        if valid.sum() == 0:
            continue
        Zg[k, valid] = _z(gbm.predict(x[valid]))
        Ze[k, valid] = _z(enet.predict(scaler.transform(x[valid])))
    return Zg, Ze


# ── Regimen por semana (VIX, terciles congelados en train, sin look-ahead) ────────

def regime_labels(week, data_dir: str, init_train_w: int):
    """Clasifica cada semana en low/normal/high por la serie VIX estandarizada.

    El fichero MACRO_VIX_level.csv difunde el VIX (estandarizado) a todas las
    columnas; tomamos una. Como esta en z-score, no aplican los umbrales 15/25 en
    niveles: usamos terciles CONGELADOS en el tramo de entrenamiento inicial
    (sin look-ahead). El valor de cada semana es el VIX conocido <= su rebalanceo.
    """
    csv = os.path.join(data_dir, "features", "MACRO_VIX_level.csv")
    df = pd.read_csv(csv, index_col=0, parse_dates=True)
    vix = df.iloc[:, 0].sort_index()  # difundido: cualquier columna es el VIX
    vals = []
    for dt in pd.DatetimeIndex(week.dates_w):
        prior = vix.loc[:dt]
        vals.append(prior.iloc[-1] if len(prior) else np.nan)
    vals = np.asarray(vals, dtype=float)
    train = vals[:init_train_w]
    train = train[~np.isnan(train)]
    q33, q67 = np.quantile(train, [1 / 3, 2 / 3])
    labels = np.where(vals < q33, "low", np.where(vals > q67, "high", "normal")).astype(object)
    return labels, (float(q33), float(q67))


# ── Estimacion del blend por regimen (OOF interno, congelado) ─────────────────────

def fit_regime_blend(week, regime_w: np.ndarray, init_train_w: int, cfg: ParsConfig):
    inner_end = int(round(init_train_w * (1 - cfg.inner_frac)))
    val_w = np.arange(inner_end + cfg.embargo, init_train_w)
    models = _train_learners(week.Xw, week.yw, inner_end, cfg.seed)
    Zg, Ze = _components(models, week.Xw, val_w)

    blend, counts = {}, {}
    for r in REGIMES:
        idx = [k for k, wk in enumerate(val_w) if regime_w[wk] == r]
        counts[r] = len(idx)
        if len(idx) < cfg.min_weeks:
            blend[r] = 0.5
            continue
        best_w, best_ic = 0.5, -np.inf
        for w in BLEND_GRID:
            sigs, ys = [], []
            for k in idx:
                s = w * Zg[k] + (1 - w) * Ze[k]
                y = week.yw[val_w[k]]
                m = ~np.isnan(s) & ~np.isnan(y)
                sigs.append(s[m])
                ys.append(y[m])
            sig = np.concatenate(sigs)
            y = np.concatenate(ys)
            ic = spearmanr(sig, y).correlation if len(sig) > 5 else np.nan
            if np.isfinite(ic) and ic > best_ic:
                best_ic, best_w = ic, w
        blend[r] = best_w
    return blend, counts


# ── Walk-forward (blend constante o por regimen, congelado) ───────────────────────

def walkforward_both(week, regime_w: np.ndarray, init_train_w: int, step_w: int,
                     blend: dict, seed: int):
    """Un solo entrenamiento por reentreno; aplica blend constante y por regimen.

    Devuelve (preds_base, preds_pars). Entrenar GBM/ENet domina el coste, asi que
    se comparten los componentes z(GBM)/z(ENet) entre ambas variantes.
    """
    W = week.Xw.shape[0]
    preds_base: dict[int, np.ndarray] = {}
    preds_pars: dict[int, np.ndarray] = {}
    start = init_train_w
    n_retrains = 0
    while start < W:
        end = min(start + step_w, W)
        models = _train_learners(week.Xw, week.yw, start, seed)
        test_w = np.arange(start, W)
        Zg, Ze = _components(models, week.Xw, test_w)
        n_retrains += 1
        for k, wk in enumerate(test_w):
            if not (start <= wk < end):
                continue
            preds_base[wk] = 0.5 * Zg[k] + 0.5 * Ze[k]
            w = float(blend.get(regime_w[wk], 0.5))
            preds_pars[wk] = w * Zg[k] + (1 - w) * Ze[k]
        start = end
    print(f"  walk-forward: {len(preds_base)} semanas OOS, {n_retrains} reentrenos")
    return preds_base, preds_pars


# ── Metricas ──────────────────────────────────────────────────────────────────────

def summarize(name: str, bt: pd.DataFrame) -> dict:
    n = len(bt)
    ns = sharpe(bt["net"])
    return {
        "variant": name, "n_weeks": n,
        "net_sharpe": ns, "net_sharpe_se": sharpe_se(ns, n),
        "gross_sharpe": sharpe(bt["gross"]),
        "net_ann": ann(bt["net"]), "net_maxdd": maxdd(bt["net"]),
        "turnover": float(bt["turnover"].mean()),
        "ic_mean": float(bt["ic"].dropna().mean()),
    }


def icir_by_regime(bt: pd.DataFrame, week, regime_w: np.ndarray) -> dict:
    date_to_regime = {pd.Timestamp(d): regime_w[i] for i, d in enumerate(week.dates_w)}
    reg = pd.Series([date_to_regime.get(pd.Timestamp(d), "normal") for d in bt.index],
                    index=bt.index)
    out = {}
    for r in REGIMES:
        ic = bt.loc[reg == r, "ic"].dropna()
        out[r] = {
            "weeks": int(len(ic)),
            "ic_mean": float(ic.mean()) if len(ic) else float("nan"),
            "icir": float(ic.mean() / ic.std(ddof=1)) if len(ic) > 1 and ic.std() > 0 else float("nan"),
        }
    return out


# ── Tabla LaTeX ─────────────────────────────────────────────────────────────────

def write_table(base: dict, pars: dict, blend: dict) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    blend_txt = ", ".join(f"{r}\\,$={blend[r]:.1f}$" for r in REGIMES)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{PARS-US: blend constante frente a blend condicionado por r\'egimen, "
        r"sobre el mismo walk-forward de " + f"{base['n_weeks']}" + r" semanas y la misma "
        r"construcci\'on \texttt{full} cost-aware. Peso del LightGBM por r\'egimen (resto, "
        r"ElasticNet): " + blend_txt + r". La diferencia de Sharpe es inferior a 1\,SE: el "
        r"resultado es nulo, como se pre-registr\'o.}",
        r"\label{tab:pars-us}",
        r"\small",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Variante & IC medio & Sharpe neto & MaxDD & Turnover \\",
        r"\midrule",
        f"Blend constante (0{{,}}5/0{{,}}5) & {base['ic_mean']:+.4f} & "
        f"${base['net_sharpe']:+.2f} \\pm {base['net_sharpe_se']:.2f}$ & "
        f"{base['net_maxdd']:+.1%} & {base['turnover']:.2f} \\\\".replace("%", r"\%"),
        f"PARS-US (blend por r\\'egimen) & {pars['ic_mean']:+.4f} & "
        f"${pars['net_sharpe']:+.2f} \\pm {pars['net_sharpe_se']:.2f}$ & "
        f"{pars['net_maxdd']:+.1%} & {pars['turnover']:.2f} \\\\".replace("%", r"\%"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    with open(os.path.join(TABLES_DIR, "pars_us.tex"), "w") as f:
        f.write("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--init_train", type=int, default=200)
    ap.add_argument("--step", type=int, default=26)
    args = ap.parse_args()
    cfg = ParsConfig()

    print("Cargando panel + resampling semanal ...")
    panel = dp.load_panel(args.data_dir)
    week = wp.build_weekly(panel)
    print(f"  {week.Xw.shape[0]} semanas x {week.Xw.shape[1]} acciones")

    regime_w, (q33, q67) = regime_labels(week, args.data_dir, args.init_train)
    uniq, cnt = np.unique(regime_w, return_counts=True)
    print(f"  regimenes (terciles VIX train q33={q33:+.2f} q67={q67:+.2f}): "
          f"{dict(zip(uniq.tolist(), cnt.tolist()))}")

    print("Estimando blend por regimen sobre OOF del train inicial (congelado) ...")
    blend, inner_counts = fit_regime_blend(week, regime_w, args.init_train, cfg)
    print(f"  pesos LightGBM por regimen: {blend}  (semanas internas: {inner_counts})")

    print("Walk-forward (un entrenamiento por reentreno, ambos blends) ...")
    preds_base, preds_pars = walkforward_both(
        week, regime_w, args.init_train, args.step, blend, cfg.seed)

    bt_base = backtest_variant(week, preds_base, CONFIGS["full"])
    bt_pars = backtest_variant(week, preds_pars, CONFIGS["full"])
    base = summarize("base_const_blend", bt_base)
    pars = summarize("parsus_regime_blend", bt_pars)

    delta = pars["net_sharpe"] - base["net_sharpe"]
    se = base["net_sharpe_se"]
    verdict = "MEJORA" if delta > se else "NULO (dentro de 1 SE, como se pre-registro)"

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame([base, pars]).to_csv(os.path.join(RESULTS_DIR, "pars_us_summary.csv"), index=False)

    reg_rows = []
    for variant, bt in [("base", bt_base), ("parsus", bt_pars)]:
        by = icir_by_regime(bt, week, regime_w)
        for r in REGIMES:
            reg_rows.append({"variant": variant, "regime": r, "blend_w_gbm": blend[r], **by[r]})
    pd.DataFrame(reg_rows).to_csv(os.path.join(RESULTS_DIR, "pars_us_regime_blend.csv"), index=False)

    write_table(base, pars, blend)

    print("\n" + "=" * 64)
    print("  PARS-US — blend constante vs blend por regimen")
    print("=" * 64)
    for d in (base, pars):
        print(f"  {d['variant']:22s} IC={d['ic_mean']:+.4f}  "
              f"Sharpe={d['net_sharpe']:+.2f}±{d['net_sharpe_se']:.2f}  "
              f"MaxDD={d['net_maxdd']:+.1%}  turn={d['turnover']:.2f}")
    print(f"  delta Sharpe = {delta:+.3f}  (1 SE = {se:.2f})  ->  {verdict}")
    print("=" * 64)

    # Figura: curvas de equity neto
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(bt_base.index, (1 + bt_base["net"]).cumprod(), color="#7f8c8d", lw=1.6,
            label=f"Blend constante (Sharpe {base['net_sharpe']:+.2f})")
    ax.plot(bt_pars.index, (1 + bt_pars["net"]).cumprod(), color="#c0392b", lw=2,
            label=f"PARS-US blend por regimen (Sharpe {pars['net_sharpe']:+.2f})")
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_title("PARS-US: blend por regimen vs constante (walk-forward neto)", fontsize=13)
    ax.set_xlabel("Fecha"); ax.set_ylabel("Retorno acumulado")
    ax.legend(fontsize=10); ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "pars_us_equity.png"), dpi=300)
    plt.close(fig)
    print("Guardado: results/pars_us_summary.csv, pars_us_regime_blend.csv, "
          "tablas/pars_us.tex, figures/pars_us_equity.png")


if __name__ == "__main__":
    main()
