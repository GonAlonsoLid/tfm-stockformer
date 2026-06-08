# Redacción de la Memoria del TFM — Plan de Implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Producir la memoria del TFM en LaTeX (español, plantilla Comillas), narrativa de síntesis "fracaso de Stockformer → pipeline simple cost-aware", ~70-80 páginas que compilan a PDF sin errores.

**Architecture:** Esqueleto LaTeX compilable desde el día 1 (`MEMORIA/tfm/`), tablas generadas automáticamente desde los CSV de `results/`, y redacción capítulo a capítulo en orden de disponibilidad fáctica (4→5→6→3→7→8→2→1), con compilación + commit tras cada capítulo. Bibliografía con biblatex/backend=bibtex.

**Tech Stack:** LaTeX (clase `book`, `babel` español, `biblatex` backend=bibtex), `latexmk`/`pdflatex`/`bibtex`, Python+pandas (`df.to_latex()`) para tablas, figuras PNG ya existentes en `results/figures/`.

**Spec de referencia:** `docs/superpowers/specs/2026-06-08-tfm-memoria-redaccion-design.md` (cifras canónicas en su sección 4 — NO usar las del plan de marzo).

---

## Convenciones de este plan

- **"Verificación" de cada capítulo = compila + cobertura del spec.** No hay tests unitarios de prosa. La prueba objetiva es: `latexmk -pdf main.tex` termina sin error y el PDF incluye el capítulo.
- **Cifras:** únicamente las de la sección 4 del spec. Si una cifra no está ahí ni en `results/`, NO inventarla — marcarla y preguntar.
- **Comando de compilación canónico** (desde `MEMORIA/tfm/`):
  ```bash
  cd "MEMORIA/tfm" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  ```
  Esperado: `Output written on main.pdf`. latexmk corre bibtex automáticamente.
- **Idioma de la prosa:** español. Términos técnicos en inglés en *cursiva* la primera vez (ej. *walk-forward*, *cost-aware*).

---

## Estructura de ficheros

```
MEMORIA/tfm/
├── main.tex                  # Documento maestro (portada, índices, \include de capítulos)
├── main.bib                  # Bibliografía (biblatex)
├── LogoUniversidadBN.pdf     # Logo Comillas (copiado de la plantilla)
├── preambulo.tex             # Paquetes + config (extraído para mantener main.tex limpio)
├── Capitulos/
│   ├── 01_introduccion.tex
│   ├── 02_estado_del_arte.tex
│   ├── 03_stockformer.tex
│   ├── 04_metodologia.tex
│   ├── 05_parte1_transferencia.tex
│   ├── 06_parte2_que_funciona.tex
│   ├── 07_discusion.tex
│   └── 08_conclusiones.tex
├── Apendices/
│   ├── A_reproducibilidad.tex
│   ├── B_tablas_completas.tex
│   └── C_estructura_repo.tex
├── tablas/                   # Fragmentos .tex generados desde results/*.csv
│   ├── ladder.tex
│   ├── significancia.tex
│   ├── weekly_summary.tex
│   ├── weekly_robustness.tex
│   └── weekly_attribution.tex
└── figuras/                  # Copias de results/figures/*.png usadas en la memoria
scripts/
└── build_latex_tables.py     # CSV (results/) → tablas/*.tex
```

---

## Task 1: Esqueleto LaTeX compilable (español, plantilla Comillas)

**Files:**
- Create: `MEMORIA/tfm/main.tex`
- Create: `MEMORIA/tfm/preambulo.tex`
- Create: `MEMORIA/tfm/main.bib`
- Create: `MEMORIA/tfm/Capitulos/0{1..8}_*.tex` (8 stubs con secciones)
- Create: `MEMORIA/tfm/Apendices/{A,B,C}_*.tex` (3 stubs)
- Copy: `MEMORIA/Posible plantilla para la memoria Latex - 2/LogoUniversidadBN.pdf` → `MEMORIA/tfm/LogoUniversidadBN.pdf`

- [ ] **Step 1: Crear `preambulo.tex`** con los paquetes de la plantilla adaptados a español y `backend=bibtex` (biber no está instalado):

```latex
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-tabla]{babel}
\usepackage{amsmath, amsfonts, amssymb}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{graphicx}
\graphicspath{{figuras/}}
\usepackage{geometry}
\geometry{a4paper, margin=2.8cm}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true, linkcolor=black, citecolor=blue, urlcolor=blue}
\usepackage[backend=bibtex, style=numeric, citestyle=numeric, sorting=none]{biblatex}
\addbibresource{main.bib}
% Cabeceras
\fancyhf{}
\fancyhead[L]{\slshape \leftmark}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
% Metadatos de portada (POR COMPLETAR por el autor donde se indica)
\newcommand{\tituloTFM}{De la transferencia fallida a una estrategia rentable: por qué los modelos simples baten a Stockformer en el S\&P 500}
\newcommand{\autorTFM}{Gonzalo Alonso Lidón}
\newcommand{\tutorTFM}{[POR COMPLETAR: nombre del tutor/a]}
\newcommand{\masterTFM}{[POR COMPLETAR: nombre exacto del Máster] --- ICAI}
\newcommand{\fechaTFM}{Madrid, junio de 2026}
```

- [ ] **Step 2: Crear `main.tex`** con portada Comillas, abstract, índices y `\include` de los 8 capítulos + 3 apéndices:

```latex
\documentclass[12pt, a4paper]{book}
\input{preambulo}
\begin{document}
  \frontmatter
  \pagestyle{empty}
  \begin{titlepage}
    \centering
    \includegraphics[width=0.3\linewidth]{LogoUniversidadBN}\\[2em]
    {\Large Universidad Pontificia Comillas --- ICAI}\\[0.5em]
    {\large \masterTFM}\\[2.5em]
    {\large Trabajo Fin de Máster}\\[1em]
    {\LARGE \bfseries \tituloTFM}\\[3em]
    {\large Autor: \autorTFM}\\[1em]
    {\large Dirigido por: \tutorTFM}\\
    \vfill
    {\large \fechaTFM}
  \end{titlepage}
  \cleardoublepage
  \chapter*{Resumen}
  \input{Capitulos/00_resumen} % abstract ES (y EN opcional)
  \cleardoublepage
  \tableofcontents
  \listoffigures
  \listoftables
  \cleardoublepage
  \mainmatter
  \pagestyle{fancy}
  \include{Capitulos/01_introduccion}
  \include{Capitulos/02_estado_del_arte}
  \include{Capitulos/03_stockformer}
  \include{Capitulos/04_metodologia}
  \include{Capitulos/05_parte1_transferencia}
  \include{Capitulos/06_parte2_que_funciona}
  \include{Capitulos/07_discusion}
  \include{Capitulos/08_conclusiones}
  \appendix
  \include{Apendices/A_reproducibilidad}
  \include{Apendices/B_tablas_completas}
  \include{Apendices/C_estructura_repo}
  \cleardoublepage
  \printbibliography[heading=bibintoc]
  \backmatter
\end{document}
```

- [ ] **Step 3: Crear los stubs de capítulo** con su esqueleto de secciones (sin prosa todavía, solo `\chapter`/`\section` + un `% TODO redactar` por sección). Ejemplo para `Capitulos/04_metodologia.tex`:

```latex
\chapter{Metodología}
\label{cap:metodologia}
\section{Datos: universo S\&P 500 y panel}      % TODO redactar
\section{Ingeniería de variables}                % TODO redactar
\section{Harness de evaluación}                  % TODO redactar
\section{La \textit{complexity ladder}}          % TODO redactar
\section{Función de pérdida de \textit{ranking}} % TODO redactar
\section{Auditoría anti-\textit{leakage}}        % TODO redactar
```

Crear análogamente los 7 capítulos restantes y `00_resumen.tex` (con `\lipsum` temporal de `\usepackage{lipsum}` NO — usar texto mínimo "Resumen pendiente de redacción.") y los 3 apéndices. Las secciones exactas de cada capítulo están en las Tasks 4-11.

- [ ] **Step 4: Crear `main.bib`** con 2 entradas semilla para que biblatex compile:

```bibtex
@article{gu2020empirical,
  title={Empirical asset pricing via machine learning},
  author={Gu, Shihao and Kelly, Bryan and Xiu, Dacheng},
  journal={The Review of Financial Studies}, volume={33}, number={5},
  pages={2223--2273}, year={2020}
}
@article{ma2024stockformer,
  title={Stockformer: A price-volume factor stock selection model},
  author={Ma, Bo and others}, journal={arXiv preprint}, year={2024}
}
```

- [ ] **Step 5: Copiar el logo**:

```bash
cp "MEMORIA/Posible plantilla para la memoria Latex - 2/LogoUniversidadBN.pdf" "MEMORIA/tfm/LogoUniversidadBN.pdf"
```

- [ ] **Step 6: Compilar el esqueleto**:

Run:
```bash
cd "MEMORIA/tfm" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```
Expected: `Output written on main.pdf`. (Un `\cite` semilla en algún stub puede dar warning de "no citado"; aceptable.)

- [ ] **Step 7: Commit**:

```bash
git add "MEMORIA/tfm" && git commit -m "feat(tfm): esqueleto LaTeX compilable de la memoria (es, plantilla Comillas)"
```

---

## Task 2: Generación automática de tablas desde `results/`

**Files:**
- Create: `scripts/build_latex_tables.py`
- Create (output): `MEMORIA/tfm/tablas/{ladder,significancia,weekly_summary,weekly_robustness,weekly_attribution}.tex`

- [ ] **Step 1: Escribir `scripts/build_latex_tables.py`** que lee los CSV canónicos y emite fragmentos `.tex` con `df.to_latex()`:

```python
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
    tex = df.to_latex(index=False, escape=True, float_format=lambda x: floatfmt % x,
                      caption=caption, label=label, position="htbp")
    (OUT / fname).write_text(tex, encoding="utf-8")
    print("wrote", fname)

# Complexity ladder
ladder = pd.read_csv(RES / "ladder_results.csv")
emit(ladder, "ladder.tex",
     "Complexity ladder: IC, parámetros y Sharpe por modelo (split único, 250 días).",
     "tab:ladder",
     cols=["model", "family", "n_params", "ic_mean", "icir", "tstat", "pvalue", "sharpe"])

# Significancia
sig = pd.read_csv(RES / "significance_tests.csv")
emit(sig, "significancia.tex",
     "Tests de significancia del IC frente a Stockformer (diferencia de medias, t, p).",
     "tab:significancia")

# Weekly summary
ws = pd.read_csv(RES / "weekly_strategy_summary.csv")
emit(ws, "weekly_summary.tex",
     "Estrategia semanal market-neutral cost-aware (ventana de 52 semanas).",
     "tab:weekly-summary")

# Weekly robustness (walk-forward)
wr = pd.read_csv(RES / "weekly_robustness_summary.csv")
emit(wr, "weekly_robustness.tex",
     "Robustez walk-forward de la estrategia semanal (215 semanas OOS).",
     "tab:weekly-robustness")

# Atribución de construcción
wa = pd.read_csv(RES / "weekly_attribution.csv")
emit(wa, "weekly_attribution.tex",
     "Atribución de construcción: aporte incremental de cada etapa al Sharpe neto.",
     "tab:weekly-attribution")

if __name__ == "__main__":
    print("Tablas generadas en", OUT)
```

- [ ] **Step 2: Ejecutar el script**:

Run: `python3 scripts/build_latex_tables.py`
Expected: imprime `wrote ladder.tex` … y `Tablas generadas en …/tablas`. 5 ficheros creados.

- [ ] **Step 3: Verificar que las tablas compilan** añadiendo temporalmente `\input{tablas/ladder}` en `Capitulos/04_metodologia.tex` y compilando:

Run: `cd "MEMORIA/tfm" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
Expected: `Output written on main.pdf` con la tabla renderizada. (Quitar el `\input` temporal después; las tablas se ubican definitivamente en las Tasks 7-8.)

- [ ] **Step 4: Commit**:

```bash
git add scripts/build_latex_tables.py "MEMORIA/tfm/tablas" && git commit -m "feat(tfm): generador de tablas LaTeX desde results/*.csv"
```

---

## Task 3: Preparar figuras

**Files:**
- Copy: `results/figures/*.png` → `MEMORIA/tfm/figuras/`

- [ ] **Step 1: Copiar las figuras usadas en la memoria**:

```bash
mkdir -p "MEMORIA/tfm/figuras"
cp results/figures/complexity_vs_ic.png results/figures/ic_comparison.png \
   results/figures/ic_with_ci.png results/figures/weekly_oos_equity.png \
   results/figures/weekly_strategy_equity.png results/figures/weekly_attribution.png \
   results/figures/improvement_waterfall.png results/figures/metrics_heatmap.png \
   results/figures/longshort_equity.png "MEMORIA/tfm/figuras/"
```

- [ ] **Step 2: Verificar inserción** añadiendo temporalmente a `04_metodologia.tex` una figura y compilando:

```latex
\begin{figure}[htbp]\centering
\includegraphics[width=0.8\linewidth]{complexity_vs_ic}
\caption{Prueba de inserción.}\label{fig:test}
\end{figure}
```
Run: `cd "MEMORIA/tfm" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
Expected: PDF con la figura. Quitar el bloque de prueba después.

- [ ] **Step 3: Commit**:

```bash
git add "MEMORIA/tfm/figuras" && git commit -m "feat(tfm): figuras de la memoria desde results/figures"
```

---

## Task 4: Capítulo 4 — Metodología (primero, máxima base fáctica)

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/04_metodologia.tex`

Secciones y contenido a redactar (≈12 pp). Cada sección: prosa en español, citas con `\cite{}`, referencias a figuras/tablas con `\ref{}`.

- [ ] **Step 1: Redactar el capítulo** con esta estructura y estos contenidos concretos:
  - `\section{Datos: universo S\&P 500 y panel}` — universo ~477 acciones, OHLCV vía yfinance, rango temporal, panel cross-seccional, splits fijos (offset 60, F-base 375), protocolo *walk-forward* purgado/embargado. Fuente: `lib/data_panel.py`, `lib/weekly_panel.py`.
  - `\section{Ingeniería de variables}` — Alpha360 (6 OHLCV × 60 lags), Alpha158, macro (VIX, term spread). Higiene por fecha: winsorize → Gaussian-rank → neutralización. Fuente: `scripts/build_alpha360.py`, `build_alpha158.py`, `build_macro_features.py`.
  - `\section{Harness de evaluación}` — rank IC, ICIR, bootstrap CI, backtest long-short, t-test IC≠0. Fuente: `lib/eval_harness.py`. Citar `\cite{gu2020empirical}`.
  - `\section{La \textit{complexity ladder}}` — niveles L0 sanity (zero/momentum/reversal), L1 lineal (Lasso/Ridge/ElasticNet), L2 trees (LightGBM/XGBoost), y el Stockformer como cúspide. Diseño "simpler-is-better" como hipótesis. Fuente: `scripts/run_cpu_ladder.py`, `run_ladder_analysis.py`.
  - `\section{Función de pérdida de \textit{ranking}}` — ListNet+IC+MAE vs MSE; optimizar para cartera, no para IC.
  - `\section{Auditoría anti-\textit{leakage}}` — las 6 comprobaciones (6/6 limpias); el IC ≈ 0 del Stockformer es genuino, no un bug. Referenciar memoria del proyecto `project_leakage_audit_clean`.
- [ ] **Step 2: Compilar** — `cd "MEMORIA/tfm" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Expected: `Output written on main.pdf`.
- [ ] **Step 3: Commit** — `git add "MEMORIA/tfm/Capitulos/04_metodologia.tex" && git commit -m "docs(tfm): capítulo 4 Metodología"`

---

## Task 5: Capítulo 5 — Parte I · ¿Transfiere Stockformer?

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/05_parte1_transferencia.tex`

- [ ] **Step 1: Redactar el capítulo** (≈12 pp):
  - `\section{Resultados de la complexity ladder}` — `\input{tablas/ladder}` (Tabla \ref{tab:ladder}) y figura `\includegraphics{complexity_vs_ic}`. Cifras canónicas (spec §4): Stockformer IC ≈ −0.003; **Lasso +0.0238 (5 params) > LightGBM +0.0136 > XGBoost +0.0141 > Ridge +0.0173**; momentum/reversal ≈ 0. Mensaje: el IC no crece (más bien decrece) con la complejidad.
  - `\section{Significancia estadística}` — `\input{tablas/significancia}`; figura `ic_with_ci`. Lasso vs Stockformer: diff +0.0319, t=1.86, **p=0.064**; LightGBM vs SF p=0.105. Honestidad: marginal, no fuerte; pero la *dirección* es consistente y robusta.
  - `\section{Ablaciones (E1--E7)}` — resumen de `results/ablation_results.csv`; figura `ic_comparison`. Loss de ranking, grafos dinámicos, features ricas: ninguno rescata el IC.
  - `\section{Modos de fallo de la transferencia}` — mapear cada resultado a un modo de fallo de la literatura (eficiencia US vs China, insuficiencia de señal OHLCV, microestructura). Citar comparativa con StockMixer/MASTER (China-only).
- [ ] **Step 2: Compilar** — `cd "MEMORIA/tfm" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Expected: PDF OK, Tablas \ref{tab:ladder}/\ref{tab:significancia} renderizadas.
- [ ] **Step 3: Commit** — `git commit -am "docs(tfm): capítulo 5 Parte I (transferencia/fallo)"`

---

## Task 6: Capítulo 6 — Parte II · ¿Qué sí funciona?

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/06_parte2_que_funciona.tex`

- [ ] **Step 1: Redactar el capítulo** (≈14 pp) — usar `docs/tesis/PROPUESTA_PIPELINE_RETORNOS.md` como fuente:
  - `\section{De la señal a las posiciones}` — pipeline: resampling semanal → neutralización FF6+sector → constructor cost-aware cvxpy → overlay VIX. Fuente: `lib/portfolio.py`, `lib/neutralize.py`, `scripts/run_weekly_strategy.py`.
  - `\section{Construcción de cartera cost-aware}` — formulación cvxpy (market-neutral, coste L1, no-trade bands, vol-target 10\%, caps), coste 8 bps/lado, lag 1d.
  - `\section{Resultados}` — `\input{tablas/weekly_summary}` y figura `weekly_oos_equity`. **net Sharpe 0.948 (52 sem), ann 7.0\%, maxDD −4.75\%, turnover 0.81**.
  - `\section{Robustez \textit{walk-forward}}` — `\input{tablas/weekly_robustness}`. **net Sharpe 0.579 ± 0.493 (215 sem OOS)**. ⚠️ Aplicar la restricción de honestidad del spec §4: presentar ambas cifras; el estimador robusto no es distinguible de cero.
  - `\section{Atribución de construcción}` — `\input{tablas/weekly_attribution}` y figura `weekly_attribution`/`improvement_waterfall`. raw 0.045 → neutralize −0.40 → cost\_aware 0.47 → full 0.58. **Mensaje central: la palanca es la construcción cost-aware, no la señal (IC semanal ≈ 0.003).**
  - `\section{Ablaciones de datos (Tier B/C)}` — fundamentales EDGAR (`weekly_robustness_summary_fund.csv`) y realized-vol/IVOL (`_realized.csv`): resultados negativos honestos, no mejoran el Sharpe neto. Refuerza "simpler-is-better".
- [ ] **Step 2: Compilar** — `cd "MEMORIA/tfm" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Expected: PDF OK con las 3 tablas weekly.
- [ ] **Step 3: Commit** — `git commit -am "docs(tfm): capítulo 6 Parte II (pipeline cost-aware)"`

---

## Task 7: Capítulo 3 — Stockformer: arquitectura original

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/03_stockformer.tex`

- [ ] **Step 1: Redactar** (≈8 pp), citando `\cite{ma2024stockformer}` y la implementación en `Stockformermodel/` / `cluster_bundle/models.py`:
  - `\section{Conjunto de variables Alpha360}` — 6 OHLCV × 60 lags.
  - `\section{Descomposición wavelet (DWT Sym2)}` — alta/baja frecuencia.
  - `\section{Encoder espacio-temporal de doble canal}` — self-attention temporal + espacial.
  - `\section{Embeddings de grafo (struc2vec)}` — grafo estático.
  - `\section{Cabezas multitarea}` — regresión (retornos) + clasificación (dirección).
  - `\section{Resultados originales y supuestos de mercado}` — CSI 300/500; supuestos atados a microestructura china (retail, límites de precio).
- [ ] **Step 2: Compilar** — Expected: PDF OK.
- [ ] **Step 3: Commit** — `git commit -am "docs(tfm): capítulo 3 arquitectura Stockformer"`

---

## Task 8: Capítulo 7 — Discusión

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/07_discusion.tex`

- [ ] **Step 1: Redactar** (≈8 pp):
  - `\section{Por qué los modelos simples ganan}` — mapeo a *shallow-beats-deep* (GKX, Rahimikia); ~400 obs cross-seccionales efectivas → sobreajuste de modelos complejos.
  - `\section{El efecto compuesto de los modos de fallo}` — cada "arreglo" corrige uno pero quedan otros.
  - `\section{Honestidad sobre el Sharpe}` — ventana favorable (0.95) vs estimador robusto (0.58 ± 0.49 ≈ 0); el retorno viene de construcción, no de predicción. Figura `metrics_heatmap`.
  - `\section{Comparación con benchmarks reportados}` — StockMixer IC 0.041 (NASDAQ), MASTER IC 0.064 (CSI300): no transfieren.
  - `\section{Limitaciones}` — datos públicos gratuitos, large-cap, semanal; sin IV histórica.
- [ ] **Step 2: Compilar** — Expected: PDF OK.
- [ ] **Step 3: Commit** — `git commit -am "docs(tfm): capítulo 7 Discusión"`

---

## Task 9: Capítulo 8 — Conclusiones y trabajo futuro

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/08_conclusiones.tex`

- [ ] **Step 1: Redactar** (≈4 pp):
  - `\section{Conclusiones}` — contribución doble (no-transferibilidad + pipeline cost-aware reproducible).
  - `\section{Trabajo futuro}` — workflow de inferencia *live* (Phase 5: predicción de posiciones reales, no solo backtest — ref. memoria `project_phase5_inference_for_trading`); sentiment GDELT+FinBERT; optimización E2E.
  - `\section{Implicaciones}` — para el ML financiero cross-market.
- [ ] **Step 2: Compilar** — Expected: PDF OK.
- [ ] **Step 3: Commit** — `git commit -am "docs(tfm): capítulo 8 Conclusiones"`

---

## Task 10: Capítulo 2 — Estado del arte

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/02_estado_del_arte.tex`
- Modify: `MEMORIA/tfm/main.bib` (añadir todas las fuentes citadas)

- [ ] **Step 1: Redactar** (≈18-22 pp) desde `docs/RESEARCH PRESS`, 8 secciones (alineadas con el spec §3, cap. 2):
  1. `\section{Eficiencia de mercado: China vs EE.UU.}`
  2. `\section{Ingeniería de variables para predicción bursátil}`
  3. `\section{Arquitecturas de aprendizaje profundo}` (transformers, MLPs, SSM, foundation models; *shallow-beats-deep*)
  4. `\section{Descomposición de señal y wavelets}`
  5. `\section{Modelado de relaciones entre acciones con grafos}`
  6. `\section{Funciones de pérdida para \textit{ranking}}`
  7. `\section{Metodología de entrenamiento}`
  8. `\section{Construcción y evaluación de cartera}`
  Cada sección: papers clave con resultados cuantitativos + implicación para la tesis.
- [ ] **Step 2: Poblar `main.bib`** con TODAS las fuentes de la sección "Fuentes principales" de `PROPUESTA_PIPELINE_RETORNOS.md` y de RESEARCH PRESS (GKX 2020, Rahimikia 2025, Ma 2024, Novy-Marx 2013, Fama-French 2015, Daniel-Moskowitz 2016, Kirtac-Germano 2024, RegimeFolio 2025, López de Prado 2018, StockMixer, MASTER, loss-functions-ranking 2025). Reemplazar las 2 entradas semilla por las reales.
- [ ] **Step 3: Compilar y verificar que TODAS las `\cite` resuelven** — `cd "MEMORIA/tfm" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Expected: PDF OK, sin warnings `Citation undefined` en el log (`grep -i "undefined" main.log` vacío).
- [ ] **Step 4: Commit** — `git add "MEMORIA/tfm/Capitulos/02_estado_del_arte.tex" "MEMORIA/tfm/main.bib" && git commit -m "docs(tfm): capítulo 2 Estado del arte + bibliografía"`

---

## Task 11: Capítulo 1 — Introducción + Resumen

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/01_introduccion.tex`
- Create/Modify: `MEMORIA/tfm/Capitulos/00_resumen.tex`

- [ ] **Step 1: Redactar la Introducción** (≈5 pp):
  - `\section{Motivación}` — DL para predicción bursátil; problema de transferencia cross-market.
  - `\section{Preguntas de investigación}` — (a) ¿transfiere Stockformer de China al S\&P 500?; (b) si no, ¿qué sí genera retorno?
  - `\section{Contribución}` — doble (no-transferibilidad + pipeline cost-aware).
  - `\section{Organización del documento}` — recorrido por los 8 capítulos.
- [ ] **Step 2: Redactar el Resumen** (`00_resumen.tex`, ~250 palabras ES; opcional *Abstract* EN). Incluir el resultado clave honesto (Stockformer IC≈0; Lasso bate; Sharpe neto 0.95/0.58).
- [ ] **Step 3: Compilar** — Expected: PDF OK.
- [ ] **Step 4: Commit** — `git commit -am "docs(tfm): capítulo 1 Introducción + Resumen"`

---

## Task 12: Apéndices + pase final

**Files:**
- Modify: `MEMORIA/tfm/Apendices/{A_reproducibilidad,B_tablas_completas,C_estructura_repo}.tex`

- [ ] **Step 1: Apéndice A (reproducibilidad)** — comandos de los scripts (`run_cpu_ladder.py`, `run_weekly_strategy.py`, `run_weekly_robustness.py`), semillas, versiones (`requirements.txt`).
- [ ] **Step 2: Apéndice B (tablas completas)** — `\input` de las tablas completas, incluyendo `ladder_full_results.csv` (generar variante en Task 2 si se desea) y las ablaciones fund/realized.
- [ ] **Step 3: Apéndice C (estructura del repo)** — árbol de `lib/`, `scripts/`, `results/`; mapa de qué script produce qué resultado.
- [ ] **Step 4: Pase final** — compilar dos veces para resolver referencias cruzadas e índices:
  ```bash
  cd "MEMORIA/tfm" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && latexmk -pdf main.tex
  ```
  Verificar: `\listoffigures`/`\listoftables`/`\tableofcontents` poblados; `grep -i "undefined\|Warning: Citation" main.log` vacío; contar páginas (`pdfinfo main.pdf | grep Pages`) → objetivo 70-80.
- [ ] **Step 5: Commit** — `git commit -am "docs(tfm): apéndices y pase final de compilación"`

---

## Self-review (cobertura del spec)

- **§1 Narrativa síntesis** → Caps. 1, 5, 6, 7 (dos actos explícitos). ✔
- **§2 Opción A (dos partes)** → Tasks 5 y 6 son capítulos separados. ✔
- **§3 Estructura 8 caps + apéndices** → Tasks 4-12. ✔
- **§4 Cifras canónicas + honestidad Sharpe** → Tasks 5, 6 (restricción explícita), 8. ✔
- **§5 Activos (figuras/tablas/docs fuente)** → Tasks 2, 3; fuentes citadas en cada capítulo. ✔
- **§6 Formato/plantilla/flujo** → Task 1 (plantilla es + backend=bibtex), orden 4→5→6→3→7→8→2→1. ✔
- **§7 Fuera de alcance** → no hay tasks de experimentos nuevos ni inferencia live (solo como future work, Task 9). ✔

**Nota sobre metadatos de portada:** `\tutorTFM` y `\masterTFM` quedan como `[POR COMPLETAR]` en `preambulo.tex` — son datos que solo aporta el autor. Confirmar antes del pase final (Task 12).
