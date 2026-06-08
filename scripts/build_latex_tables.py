"""Genera fragmentos LaTeX de tablas desde results/*.csv para la memoria del TFM."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "MEMORIA" / "tfm" / "tablas"
OUT.mkdir(parents=True, exist_ok=True)


def emit(df, fname, caption, label, cols=None, floatfmt="%.4f", body_only=False):
    if cols:
        df = df[cols]
    if body_only:
        # Just the tabular environment, so the caller can wrap it in
        # \resizebox / \begin{table} from the .tex (used for very wide tables).
        tex = df.to_latex(index=False, escape=True,
                          float_format=lambda x: (floatfmt % x) if pd.notna(x) else "",
                          na_rep="")
    else:
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

    # ── Tablas completas (Apéndice B) ───────────────────────────────────────────
    ladder_full = pd.read_csv(RES / "ladder_full_results.csv")
    # 21 columns: emit only the tabular body so the .tex wraps it in \resizebox.
    emit(ladder_full, "ladder_full_body.tex",
         "Resultados completos de la complexity ladder",
         "tab:ladder-full", body_only=True)

    rob_fund = pd.read_csv(RES / "weekly_robustness_summary_fund.csv")
    emit(rob_fund, "robustez_fund.tex",
         "Robustez walk-forward con fundamentales (Tier B)",
         "tab:rob-fund")

    rob_realized = pd.read_csv(RES / "weekly_robustness_summary_realized.csv")
    emit(rob_realized, "robustez_realized.tex",
         "Robustez walk-forward con realized-vol (Tier C)",
         "tab:rob-realized")

    ablation = pd.read_csv(RES / "ablation_results.csv")
    emit(ablation, "ablation.tex",
         "Ablación de variantes de entrada y modelo",
         "tab:ablation")

    print("Tablas generadas en", OUT)


if __name__ == "__main__":
    main()
