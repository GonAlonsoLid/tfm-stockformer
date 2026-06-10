#!/usr/bin/env python3
"""results/transplant_search.csv -> MEMORIA/tfm/tablas/rq2_modular_transfer.tex.

One row per standalone signal (grouped: base / architecture-distilled / established
bench), with its solo metrics plus the Sharpe and holdout of the ensemble+signal
combination, so complementarity with the feature predictor is visible.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "transplant_search.csv")
OUT = os.path.join(ROOT, "MEMORIA", "tfm", "tablas", "rq2_modular_transfer.tex")

LABEL = {
    "ensemble": "Ensemble ML (base)",
    "peer": "Pares / grafo (causal)",
    "trend": "Tendencia filtrada (causal)",
    "low_ivol": r"Baja vol.\ idiosincr\'atica (IVOL)",
    "bab": "Betting-against-beta",
    "volmom": r"Momentum gestionado por vol.",
    "high52": r"Proximidad m\'aximo 52 sem.",
    "tsmom": "Trend / TS-momentum",
    "season": "Estacionalidad",
}
GROUPS = [
    ("Predictor base", ["ensemble"]),
    ("Destiladas de la arquitectura del Stockformer", ["peer", "trend"]),
    ("Banco de factores establecidos (pre-registrado)",
     ["low_ivol", "bab", "volmom", "high52", "tsmom", "season"]),
]


def main():
    df = pd.read_csv(CSV).set_index("signal")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{B\'usqueda de se\~nal ampliada: se\~nales destiladas de la arquitectura del "
        r"Stockformer (grafo de pares y tendencia filtrada, ambas causales) y banco "
        r"pre-registrado de factores establecidos, cada una \emph{sola} y \emph{combinada} con el "
        r"\textit{ensemble} de \textit{features} (columnas $+$ens). Misma construcci\'on "
        r"cost-aware y \textit{walk-forward} de 311 semanas (\textit{holdout}: 104) que la "
        r"Tabla~\ref{tab:rq2-signal-search}. List\'on pre-registrado: $t_{\text{NW}}>2$ y "
        r"\textit{holdout}$>0$.}",
        r"\label{tab:rq2-modular}",
        r"\small",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Se\~nal & Sharpe & $t_{\text{NW}}$ & \textit{Hold.} & IC & "
        r"Sharpe$_{+\text{ens}}$ & \textit{Hold.}$_{+\text{ens}}$ \\",
        r"\midrule",
    ]
    for gi, (group, keys) in enumerate(GROUPS):
        if gi > 0:
            lines.append(r"\addlinespace")
        lines.append(rf"\multicolumn{{7}}{{l}}{{\emph{{{group}}}}} \\")
        for k in keys:
            if k not in df.index:
                continue
            r = df.loc[k]
            ek = f"ens_{k}"
            if ek in df.index:
                er = df.loc[ek]
                ens_sh, ens_hold = f"${er['sharpe']:+.2f}$", f"${er['hold']:+.2f}$"
            else:
                ens_sh, ens_hold = "---", "---"
            lines.append(
                f"{LABEL[k]} & ${r['sharpe']:+.2f}$ & ${r['t']:+.2f}$ & ${r['hold']:+.2f}$ & "
                f"${r['ic']:+.4f}$ & {ens_sh} & {ens_hold} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
