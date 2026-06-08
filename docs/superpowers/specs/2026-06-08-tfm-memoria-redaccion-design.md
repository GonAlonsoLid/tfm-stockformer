# Diseño de la memoria del TFM — Redacción

**Fecha:** 2026-06-08
**Autor:** Gonzalo Alonso Lidón
**Centro:** Universidad Pontificia Comillas — ICAI
**Idioma:** Español
**Formato:** LaTeX (plantilla Comillas, clase `book`, biblatex/biber)

---

## 1. Decisión narrativa (centro de gravedad)

**Marco: síntesis — "fracaso → qué sí funciona".**

La memoria cuenta una historia en dos actos que usa *todo* el trabajo empírico realizado:

1. **Acto I (fracaso):** se intenta adaptar Stockformer (modelo wavelet-transformer-grafo-multitarea calibrado para acciones chinas A-shares) al S&P 500. No genera señal predictiva (IC ≈ −0.003).
2. **Acto II (constructivo):** el fracaso motiva la pregunta "¿qué *sí* genera retorno?". La respuesta — confirmada empíricamente — es: **modelos simples + construcción de cartera cost-aware**, no redes complejas.

Esta narrativa se eligió frente a dos alternativas descartadas:
- *Estudio de fallo puro* (plan de marzo 2026-03-27, "8 modos de fallo"): deja lo constructivo como future work. Descartado: infrautiliza la estrategia semanal.
- *Pipeline de retornos puro* (giro registrado en memoria): relega Stockformer a prólogo. Descartado: infrautiliza las ablaciones y el estudio de transferencia.

**Contribución doble:** (a) evidencia empírica de no-transferibilidad cross-market de modelos DL financieros; (b) un pipeline reproducible que cuantifica que la palanca del Sharpe neto está en la *construcción de cartera*, no en la arquitectura del modelo.

## 2. Estructura del cuerpo empírico

**Opción A — Dos partes explícitas (elegida).** Capítulo "¿Transfiere Stockformer?" (Acto I) + capítulo "¿Qué sí funciona?" (Acto II). La estructura *es* el mensaje. Reparto de peso ≈ 50/50.

Descartadas: B (capítulo experimental único integrado — diluye el mensaje) y C (replicar 8 modos de fallo + apéndice constructivo — contradice el peso decidido).

## 3. Estructura de capítulos

| # | Capítulo | Págs aprox | Contenido |
|---|----------|-----------|-----------|
| 1 | Introducción | 5 | Motivación (DL bursátil + transferencia cross-market); doble pregunta de investigación: (a) ¿transfiere Stockformer de China al S&P 500?, (b) si no, ¿qué sí genera retorno?; contribución doble; organización del documento. |
| 2 | Estado del arte | 18–22 | Basado en `docs/RESEARCH PRESS`: (1) eficiencia de mercado China vs US; (2) feature engineering (Alpha360 vs alternativas); (3) arquitecturas DL (transformers, MLPs, SSM, foundation models); (4) wavelets/descomposición; (5) grafos de relaciones entre acciones; (6) loss de ranking; (7) metodología de entrenamiento; (8) construcción y evaluación de cartera + *shallow-beats-deep* (GKX 2020, Rahimikia 2025). |
| 3 | Stockformer: arquitectura original | 8 | Modelo de Ma et al. (2024): Alpha360 (6 OHLCV × 60 lags), DWT Sym2 (alta/baja frecuencia), encoder dual-channel con self-attention, embeddings struc2vec, cabezas multitarea (regresión + clasificación); resultados CSI 300/500; supuestos atados a microestructura china. |
| 4 | Metodología | 12 | Datos S&P 500 (universo, panel, splits, walk-forward purgado/embargado); feature engineering (Alpha360/Alpha158/macro); harness de evaluación unificado (rank IC, bootstrap CI, backtest L/S); *complexity ladder* (sanity → lineal → trees → Stockformer); loss de ranking (ListNet+IC+MAE); **auditoría anti-leakage (6/6 limpia)**. |
| 5 | **Parte I · ¿Transfiere Stockformer?** | 12 | Resultados de la ladder: Stockformer IC ≈ −0.003 vs Lasso +0.024 > LightGBM +0.014 > XGBoost +0.014; tests de significancia (Lasso vs SF: diff +0.032, t=1.86, p=0.064); ablaciones E1–E7; mapeo de cada resultado a un modo de fallo de la literatura. |
| 6 | **Parte II · ¿Qué sí funciona?** | 14 | Estrategia semanal market-neutral cost-aware: resampling semanal, neutralización FF6+sector, constructor cvxpy (dollar/beta-neutral, coste L1, no-trade bands, vol-target, caps), overlay VIX; **atribución de construcción** (la construcción cost-aware es la palanca, no la señal: IC semanal ≈ 0.003); resultados honestos; ablaciones Tier B/C negativas (fundamentales EDGAR y realized-vol/IVOL no mejoran el Sharpe neto). |
| 7 | Discusión | 8 | Por qué *simpler-is-better* (mapeo a literatura); honestidad sobre el Sharpe (ventana favorable vs estimador robusto); el efecto compuesto de los modos de fallo; comparación con benchmarks reportados (StockMixer, MASTER — China-only); limitaciones. |
| 8 | Conclusiones y trabajo futuro | 4 | Resumen; contribución doble; trabajo futuro: workflow de inferencia *live* (Phase 5, predicción de posiciones reales), sentiment GDELT, optimización E2E; implicaciones para ML financiero. |
| Ap. | Apéndices | — | A: configs/reproducibilidad y semillas · B: tablas completas con desviaciones · C: estructura del repositorio e instrucciones de reproducción. |

**Total estimado: ~70–80 páginas.**

## 4. Cifras canónicas (verificadas en `results/`, NO usar las del plan de marzo)

El plan de marzo cita cifras antiguas (IC = −0.005, etc.). Las cifras canónicas para la redacción son:

**Complexity ladder (split único, 250 días) — `results/ladder_results.csv`:**
- Lasso (5 params): IC **+0.0238**, Sharpe +0.91, ann +27.4%
- Ridge (376): IC +0.0173, Sharpe −2.85
- LightGBM (31): IC +0.0136, Sharpe +0.21
- XGBoost (9600): IC +0.0141, Sharpe −1.85
- Stockformer (~1M): IC ≈ **−0.003** (referencia)
- momentum/reversal: ≈ 0

**Significancia — `results/significance_tests.csv`:**
- Lasso vs Stockformer: mean_diff +0.0319, t=1.86, **p=0.064** (n=236)
- LightGBM vs SF: +0.0178, t=1.63, p=0.105

**Estrategia semanal — `results/weekly_strategy_summary.csv` (52 sem):**
- net Sharpe **0.948**, net ann 7.0%, net maxDD −4.75%, turnover 0.81, IC semanal 0.005

**Robustez walk-forward — `results/weekly_robustness_summary.csv` (215 sem OOS):**
- net Sharpe **0.579 ± 0.493 (SE)**, net ann 4.1%, maxDD −11.5%, IC 0.0033

**Atribución — `results/weekly_attribution.csv` (215 sem):**
- raw_decile 0.045 → +neutralize −0.40 → +cost_aware 0.47 → full 0.58
- ⚠️ La neutralización *resta* Sharpe; la construcción cost-aware es la que lo rescata.

### Restricción de honestidad (CRÍTICA)
El titular NO es "Sharpe neto 0.95" a secas. La redacción debe presentar **ambas** cifras: 0.95 en la ventana de 52 semanas y **0.58 ± 0.49** en el walk-forward de 215 semanas (estadísticamente no distinguible de cero). El IC de la señal semanal es ≈ 0.003. El mensaje es que *el retorno neto positivo proviene de la construcción de cartera cost-aware, no de poder predictivo de la señal*. Esto es coherente con la tesis "simpler-is-better" y con la auditoría anti-leakage limpia.

## 5. Activos disponibles para integrar

**Figuras (`results/figures/`):** `complexity_vs_ic.png`, `ic_comparison.png`, `ic_with_ci.png`, `weekly_oos_equity.png`, `weekly_strategy_equity.png`, `weekly_attribution.png`, `improvement_waterfall.png`, `metrics_heatmap.png`, `longshort_equity.png`, `ablation_table.tex`.

**Tablas (`results/*.csv`):** `ladder_results.csv`, `ladder_full_results.csv`, `significance_tests.csv`, `ablation_results.csv`, `weekly_strategy_summary.csv`, `weekly_robustness_summary{,_fund,_realized}.csv`, `weekly_attribution{,_fund,_realized}.csv`.

**Documentos fuente:** `docs/RESEARCH PRESS` (→ Cap. 2), `docs/tesis/PROPUESTA_PIPELINE_RETORNOS.md` (→ Cap. 6), `docs/superpowers/specs/2026-03-27-tfm-documentation-plan-design.md` (estructura E1–E7 → Cap. 5).

**Plantilla:** `MEMORIA/Posible plantilla para la memoria Latex - 2/` (main.tex, Chapter1/, main.bib, LogoUniversidadBN.pdf).

## 6. Formato y producción

- **Ubicación de la memoria:** nueva carpeta `MEMORIA/tfm/` (o equivalente) con `main.tex` adaptado a español + un `.tex` por capítulo bajo `Capitulos/`, `main.bib` (biblatex/biber), figuras copiadas/enlazadas desde `results/figures/`.
- **Adaptaciones a la plantilla:** `babel` español; portada Comillas (título, autor, tutor, "Máster ... — ICAI", "Madrid, junio 2026"); abstract ES (+ opcional EN); índices de figuras/tablas; bibliografía biblatex.
- **Flujo de trabajo de redacción:**
  1. Adaptar plantilla → esqueleto completo (main.tex + 8 capítulos vacíos con secciones + apéndices) que **compile** desde el día 1.
  2. Consolidar tablas LaTeX desde los CSV de `results/` y cablear figuras.
  3. Redactar capítulo a capítulo con checkpoints de revisión (orden sugerido: 4 → 5 → 6 → 3 → 7 → 8 → 2 → 1, de lo más fáctico/disponible a lo más narrativo).
  4. Bibliografía: poblar `main.bib` con las fuentes del Estado del arte.
- **Verificación:** la memoria debe compilar con `latexmk -pdf` (o `pdflatex` + `biber`) sin errores tras cada capítulo.

## 7. Fuera de alcance (YAGNI)

- No se ejecutan nuevos experimentos (la evidencia empírica está cerrada; los últimos tiers fueron negativos).
- No se construye el workflow de inferencia *live* aquí (queda como trabajo futuro, Cap. 8).
- No se genera la presentación de diapositivas en este ciclo (entregable separado).
- No se reescriben las cifras: se usan exclusivamente las de `results/` (sección 4).
