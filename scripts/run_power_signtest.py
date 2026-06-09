"""Potencia, MDE y tests de signo/Wilcoxon del rank-IC diario frente a Stockformer.

Lee results/ladder_daily_ic/{model}.csv (columnas: fecha, rank_ic), alinea cada
modelo a las fechas del Stockformer (comparacion pareada por dia) y calcula, para
cada modelo simple:

  - dIC_t = IC_modelo(t) - IC_stockformer(t)  (diferencia pareada por dia)
  - media, sd, SE, t y p (t pareado, dos colas)
  - test de SIGNOS temporal: #dias con dIC>0, binomial exacto (una y dos colas)
  - Wilcoxon signed-rank (dos colas)
  - IC 95% de la media de dIC
  - MDE al 80% de potencia, alpha 0.05 dos colas: (z_.975 + z_.80) * SE

Salidas:
  results/power_signtest.csv
  MEMORIA/tfm/tablas/power_signtest.tex

Honestidad: este script NO elige cola ni umbral; reporta ambas colas y deja que
los numeros hablen. Reproducible: solo depende de los CSV de IC diario ya volcados.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IC_DIR = os.path.join(REPO, "results", "ladder_daily_ic")
OUT_CSV = os.path.join(REPO, "results", "power_signtest.csv")
OUT_TEX = os.path.join(REPO, "MEMORIA", "tfm", "tablas", "power_signtest.tex")

# Modelos predictivos (no-sanity), en orden de la escalera.
MODELS = ["ridge", "lasso", "elasticnet", "lightgbm", "xgboost"]
BENCH = "stockformer"

Z_975 = stats.norm.ppf(0.975)   # 1.95996
Z_80 = stats.norm.ppf(0.80)     # 0.84162
MDE_K = Z_975 + Z_80            # ~2.80158


def load_ic(name: str) -> pd.Series:
    df = pd.read_csv(os.path.join(IC_DIR, f"{name}.csv"), index_col=0)
    s = df["rank_ic"].copy()
    s.index = pd.to_datetime(s.index)
    return s.dropna()


def main() -> None:
    bench = load_ic(BENCH)
    rows = []
    for m in MODELS:
        ic = load_ic(m)
        # Comparacion pareada por dia: interseccion de fechas con el benchmark.
        joined = pd.concat([ic.rename("m"), bench.rename("b")], axis=1, join="inner").dropna()
        d = (joined["m"] - joined["b"]).to_numpy()
        n = d.size
        mean = float(d.mean())
        sd = float(d.std(ddof=1))
        se = sd / np.sqrt(n)
        tstat = mean / se
        p_t = float(stats.t.sf(abs(tstat), df=n - 1) * 2)

        # Test de signos temporal (binomial exacto sobre dias con dIC != 0).
        pos = int((d > 0).sum())
        nz = int((d != 0).sum())
        bt = stats.binomtest(pos, nz, 0.5)
        p_sign_two = float(bt.pvalue)
        p_sign_one = float(stats.binom.sf(pos - 1, nz, 0.5))  # P(X >= pos)

        # Wilcoxon signed-rank (dos colas).
        try:
            p_wilcoxon = float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
        except ValueError:
            p_wilcoxon = float("nan")

        # IC 95% de la media (t de Student) y MDE al 80% de potencia.
        tcrit = stats.t.ppf(0.975, df=n - 1)
        ci_low = mean - tcrit * se
        ci_high = mean + tcrit * se
        mde = MDE_K * se

        rows.append(
            dict(
                model=m, n=n, mean_dIC=mean, sd_dIC=sd, se=se, tstat=tstat, p_t=p_t,
                days_pos=pos, pct_pos=100.0 * pos / nz,
                sign_p_one=p_sign_one, sign_p_two=p_sign_two,
                wilcoxon_p=p_wilcoxon, ci_low=ci_low, ci_high=ci_high, mde_80=mde,
            )
        )

    res = pd.DataFrame(rows)
    res.to_csv(OUT_CSV, index=False)
    print(res.to_string(index=False))

    # Tabla LaTeX (solo columnas defendibles).
    disp = {
        "ridge": "Ridge", "lasso": "Lasso", "elasticnet": "ElasticNet",
        "lightgbm": "LightGBM", "xgboost": "XGBoost",
    }
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Contraste direccional del rank-IC diario frente a Stockformer (comparaci\'on "
        r"pareada por d\'ia, $n$ d\'ias comunes). Test de signos temporal (binomial exacto, dos "
        r"colas), Wilcoxon, intervalo de confianza al 95\% de la diferencia media de IC ($\Delta$IC) "
        r"y efecto m\'inimo detectable (MDE) al 80\% de potencia. Ning\'un modelo cruza $0{,}05$: "
        r"la superioridad es direccional, no concluyente, y todos los $\Delta$IC observados caen por "
        r"debajo de su MDE.}",
        r"\label{tab:power-signtest}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Modelo & \% d\'ias $>$SF & $p$ signos & $p$ Wilcoxon & $\Delta$IC [IC 95\%] & MDE$_{80}$ \\",
        r"\midrule",
    ]
    for _, r in res.iterrows():
        lines.append(
            f"{disp[r['model']]} & {r['pct_pos']:.1f} & {r['sign_p_two']:.3f} & "
            f"{r['wilcoxon_p']:.3f} & "
            f"${r['mean_dIC']:+.4f}$ [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] & "
            f"{r['mde_80']:.4f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines))
    print(f"\nEscrito: {OUT_CSV}\nEscrito: {OUT_TEX}")


if __name__ == "__main__":
    main()
