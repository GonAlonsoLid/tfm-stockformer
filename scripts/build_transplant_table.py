#!/usr/bin/env python3
"""results/transplant_search.csv -> MEMORIA/tfm/tablas/rq2_modular_transfer.tex."""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "transplant_search.csv")
OUT = os.path.join(ROOT, "MEMORIA", "tfm", "tablas", "rq2_modular_transfer.tex")

LABEL = {
    "ensemble": "Ensemble ML (base)",
    "peer": r"Se\~nal de pares (grafo) sola",
    "trend": "Tendencia filtrada sola",
    "ens_peer": "Ensemble $+$ pares",
    "ens_trend": "Ensemble $+$ tendencia filtrada",
    "ens_peer_trend": "Ensemble $+$ pares $+$ tendencia",
}
ORDER = ["ensemble", "peer", "trend", "ens_peer", "ens_trend", "ens_peer_trend"]


def main():
    df = pd.read_csv(CSV).set_index("signal")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Transferencia modular: se\~nales destiladas de la arquitectura del "
        r"Stockformer (grafo de pares causal, tendencia filtrada causal), solas y combinadas "
        r"con el \textit{ensemble} de \textit{features}. Misma construcci\'on cost-aware y "
        r"\textit{walk-forward} de 311 semanas (\textit{holdout}: 104 semanas) que la "
        r"Tabla~\ref{tab:rq2-signal-search}.}",
        r"\label{tab:rq2-modular}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Se\~nal & Sharpe neto & $t_{\text{NW}}$ & \textit{Holdout} & Turnover & IC \\",
        r"\midrule",
    ]
    for k in ORDER:
        if k not in df.index:
            continue
        r = df.loc[k]
        lines.append(
            f"{LABEL[k]} & ${r['sharpe']:+.2f}$ & ${r['t']:+.2f}$ & "
            f"${r['hold']:+.2f}$ & {r['turnover']:.2f} & ${r['ic']:+.4f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
