"""Genera fragmentos LaTeX de tablas desde results/*.csv para la memoria del TFM."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "MEMORIA" / "tfm" / "tablas"
OUT.mkdir(parents=True, exist_ok=True)


def emit(df, fname, caption, label, cols=None, floatfmt="%.4f"):
    if cols:
        df = df[cols]
    tex = df.to_latex(index=False, escape=True,
                      float_format=lambda x: (floatfmt % x) if pd.notna(x) else "",
                      na_rep="",
                      caption=caption, label=label, position="htbp")
    (OUT / fname).write_text(tex, encoding="utf-8")
    print("wrote", fname)


def main():
    ladder = pd.read_csv(RES / "ladder_results.csv")
    emit(ladder, "ladder.tex",
         "Complexity ladder: IC, parámetros y Sharpe por modelo (split único, 250 días).",
         "tab:ladder",
         cols=["model", "family", "n_params", "ic_mean", "icir", "tstat", "pvalue", "sharpe"])

    sig = pd.read_csv(RES / "significance_tests.csv")
    emit(sig, "significancia.tex",
         "Tests de significancia del IC frente a Stockformer (diferencia de medias, t, p).",
         "tab:significancia")

    ws = pd.read_csv(RES / "weekly_strategy_summary.csv")
    emit(ws, "weekly_summary.tex",
         "Estrategia semanal market-neutral cost-aware (ventana de 52 semanas).",
         "tab:weekly-summary")

    wr = pd.read_csv(RES / "weekly_robustness_summary.csv")
    emit(wr, "weekly_robustness.tex",
         "Robustez walk-forward de la estrategia semanal (215 semanas OOS).",
         "tab:weekly-robustness")

    wa = pd.read_csv(RES / "weekly_attribution.csv")
    emit(wa, "weekly_attribution.tex",
         "Atribución de construcción: aporte incremental de cada etapa al Sharpe neto.",
         "tab:weekly-attribution")

    print("Tablas generadas en", OUT)


if __name__ == "__main__":
    main()
