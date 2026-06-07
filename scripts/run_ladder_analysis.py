#!/usr/bin/env python3
"""Assemble the full complexity-ladder comparison and generate thesis figures.

Steps:
    1. Load CPU ladder results (results/ladder_results.csv).
    2. Evaluate the trained Stockformer from its saved inference output, with the
       SAME harness and real test dates, and append it as the top rung.
    3. Paired IC significance tests: best simple models vs Stockformer.
    4. Generate publishable figures (complexity vs IC, IC with bootstrap CI,
       long-short equity curves, metrics heatmap) under results/figures/.

Outputs:
    results/ladder_full_results.csv
    results/significance_tests.csv
    results/figures/*.png

Usage:
    python scripts/run_ladder_analysis.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib import data_panel as dp  # noqa: E402
from lib import eval_harness as eh  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DAILY_IC_DIR = os.path.join(RESULTS_DIR, "ladder_daily_ic")
LS_RETURNS_DIR = os.path.join(RESULTS_DIR, "ladder_ls_returns")

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "Stock_SP500_2018-01-01_2026-03-16")
STOCKFORMER_OUT = os.path.join(
    PROJECT_ROOT, "output", "Multitask_output_SP500_2018-01-01_2026-03-16", "regression")
STOCKFORMER_PARAMS = 1_041_755  # counted from the saved checkpoint

QUANTILE, FEE, N_BOOT, SEED = 0.1, 0.001, 2000, 0

# Display labels and complexity tiers for the ladder
PRETTY = {
    "zero": "Zero", "momentum": "Momentum", "reversal": "Reversal",
    "ridge": "Ridge", "lasso": "Lasso", "elasticnet": "ElasticNet",
    "lightgbm": "LightGBM", "xgboost": "XGBoost", "stockformer": "Stockformer",
}
FAMILY_COLOR = {
    "sanity": "#95a5a6", "linear": "#2980b9", "trees": "#27ae60",
    "transformer": "#c0392b",
}


# ── Stockformer integration ─────────────────────────────────────────────────────

def stockformer_canonical() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load saved Stockformer pred/label and attach real test dates + tickers."""
    pred = pd.read_csv(os.path.join(STOCKFORMER_OUT, "regression_pred_last_step.csv"),
                       header=None).values.astype(float)
    label = pd.read_csv(os.path.join(STOCKFORMER_OUT, "regression_label_last_step.csv"),
                        header=None).values.astype(float)
    n_days, n_stocks = pred.shape

    tickers = dp._load_tickers(DATA_DIR, n_stocks)
    all_dates = pd.to_datetime(pd.read_csv(os.path.join(DATA_DIR, "label.csv"),
                                           usecols=[0]).iloc[:, 0])
    aligned = all_dates.iloc[dp.ALPHA360_LAG:]
    test_dates = pd.DatetimeIndex(aligned.iloc[-n_days:].values)
    return (dp.to_canonical(pred, test_dates, tickers),
            dp.to_canonical(label, test_dates, tickers))


def evaluate_stockformer() -> dict:
    pred, label = stockformer_canonical()
    metrics = eh.evaluate(pred, label, quantile=QUANTILE, fee=FEE,
                          n_boot=N_BOOT, seed=SEED)
    eh.daily_rank_ic(pred, label).rename("rank_ic").to_csv(
        os.path.join(DAILY_IC_DIR, "stockformer.csv"), header=True)
    eh.longshort_returns(pred, label, quantile=QUANTILE, fee=FEE).to_csv(
        os.path.join(LS_RETURNS_DIR, "stockformer.csv"))
    return {"model": "stockformer", "level": "L5", "family": "transformer",
            "n_params": STOCKFORMER_PARAMS, **metrics, "seconds": 0.0}


# ── Significance tests ──────────────────────────────────────────────────────────

def significance_tests(challengers: list[str]) -> pd.DataFrame:
    """Paired daily-IC t-test of each challenger vs the Stockformer."""
    sf = pd.read_csv(os.path.join(DAILY_IC_DIR, "stockformer.csv"),
                     index_col=0)["rank_ic"]
    sf.index = pd.to_datetime(sf.index)
    rows = []
    for name in challengers:
        ic = pd.read_csv(os.path.join(DAILY_IC_DIR, f"{name}.csv"),
                         index_col=0)["rank_ic"]
        ic.index = pd.to_datetime(ic.index)
        res = eh.paired_ic_ttest(ic, sf)
        rows.append({"model": name, "vs": "stockformer", **res})
        print(f"  {PRETTY[name]:11s} vs Stockformer: "
              f"ΔIC={res['mean_diff']:+.5f}  t={res['tstat']:+.2f}  "
              f"p={res['pvalue']:.4f}  n={res['n']}")
    return pd.DataFrame(rows)


# ── Figures ─────────────────────────────────────────────────────────────────────

def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values("n_params").reset_index(drop=True)


def fig_complexity_vs_ic(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["ic_mean"]).copy()
    d["x"] = d["n_params"].clip(lower=1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for fam, sub in d.groupby("family"):
        ax.scatter(sub["x"], sub["ic_mean"], s=120, color=FAMILY_COLOR.get(fam, "#777"),
                   label=fam, edgecolor="black", linewidth=0.6, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(PRETTY.get(r["model"], r["model"]),
                    (r["x"], r["ic_mean"]), textcoords="offset points",
                    xytext=(7, 5), fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.02, color="gray", ls="--", lw=0.8, label="IC útil mínimo (0.02)")
    ax.set_xscale("log")
    ax.set_xlabel("Complejidad del modelo (nº de parámetros, escala log)", fontsize=12)
    ax.set_ylabel("IC medio (Spearman)", fontsize=12)
    ax.set_title("Complejidad del modelo vs. poder predictivo (S&P 500)", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, ls="--", alpha=0.4)
    _save(fig, "complexity_vs_ic.png")


def fig_ic_with_ci(df: pd.DataFrame) -> None:
    d = _ordered(df.dropna(subset=["ic_mean"]))
    x = np.arange(len(d))
    lower = d["ic_mean"] - d["ic_ci_low"]
    upper = d["ic_ci_high"] - d["ic_mean"]
    colors = [FAMILY_COLOR.get(f, "#777") for f in d["family"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, d["ic_mean"], color=colors, edgecolor="black", linewidth=0.5,
           yerr=[lower, upper], capsize=4, error_kw=dict(ecolor="black", lw=1))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(m, m) for m in d["model"]], rotation=30, ha="right")
    ax.set_ylabel("IC medio con IC bootstrap 95%", fontsize=12)
    ax.set_title("IC por modelo con intervalos de confianza bootstrap (S&P 500)", fontsize=13)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    _save(fig, "ic_with_ci.png")


def fig_longshort_equity(df: pd.DataFrame) -> None:
    highlight = {"lasso": "#2980b9", "lightgbm": "#27ae60", "stockformer": "#c0392b"}
    fig, ax = plt.subplots(figsize=(11, 6))
    for name in df["model"]:
        path = os.path.join(LS_RETURNS_DIR, f"{name}.csv")
        if not os.path.isfile(path):
            continue
        ls = pd.read_csv(path, index_col=0, parse_dates=True)
        if "net" not in ls or ls["net"].abs().sum() == 0:
            continue
        equity = (1 + ls["net"]).cumprod()
        if name in highlight:
            ax.plot(equity.index, equity.values, label=PRETTY[name],
                    color=highlight[name], lw=2, zorder=3)
        else:
            ax.plot(equity.index, equity.values, color="#cccccc", lw=1, zorder=1)
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Fecha", fontsize=12)
    ax.set_ylabel("Retorno acumulado (long-short neto, deciles)", fontsize=12)
    ax.set_title("Curvas de equity long-short neto de costes (10 bps/lado)", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, ls="--", alpha=0.4)
    _save(fig, "longshort_equity.png")


def fig_metrics_heatmap(df: pd.DataFrame) -> None:
    d = _ordered(df.dropna(subset=["ic_mean"]))
    metrics = ["ic_mean", "icir", "tstat", "pct_positive", "sharpe"]
    labels = ["IC", "ICIR", "t-stat", "% días +", "Sharpe"]
    mat = d[metrics].astype(float).to_numpy()
    # column-wise z-score for comparable color scale
    z = (mat - np.nanmean(mat, axis=0)) / (np.nanstd(mat, axis=0) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(z, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([PRETTY.get(m, m) for m in d["model"]])
    for i in range(len(d)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Métricas por modelo (color = z-score por columna)", fontsize=13)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, "metrics_heatmap.png")


def _save(fig, name: str) -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, name), dpi=300)
    plt.close(fig)
    print(f"  saved figures/{name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    cpu = pd.read_csv(os.path.join(RESULTS_DIR, "ladder_results.csv"))

    print("Evaluating Stockformer with the unified harness ...")
    sf_row = evaluate_stockformer()
    print(f"  Stockformer: IC={sf_row['ic_mean']:+.5f}  ICIR={sf_row['icir']:+.3f}  "
          f"CI=[{sf_row['ic_ci_low']:+.4f},{sf_row['ic_ci_high']:+.4f}]  n={sf_row['n_days']}")

    full = pd.concat([cpu, pd.DataFrame([sf_row])], ignore_index=True)
    full = full[cpu.columns.tolist()]
    full_path = os.path.join(RESULTS_DIR, "ladder_full_results.csv")
    full.to_csv(full_path, index=False)
    print(f"\nSaved full ladder ({len(full)} models) to {full_path}")
    print(full[["model", "level", "n_params", "ic_mean", "icir", "tstat", "sharpe"]].to_string(index=False))

    print("\nPaired IC significance tests (challenger vs Stockformer):")
    sig = significance_tests(["lasso", "ridge", "lightgbm", "xgboost", "momentum"])
    sig.to_csv(os.path.join(RESULTS_DIR, "significance_tests.csv"), index=False)

    print("\nGenerating figures:")
    fig_complexity_vs_ic(full)
    fig_ic_with_ci(full)
    fig_longshort_equity(full)
    fig_metrics_heatmap(full)
    print(f"\nDone. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
