# Propuesta de pipeline para generar retornos en el S&P 500

> **Objetivo:** estrategia **long-short market-neutral**, rebalanceo **semanal**, que maximice el **Sharpe neto de costes** con métricas de riesgo controladas. Síntesis de investigación 2023–2026 (4 búsquedas paralelas) + evidencia empírica propia.

---

## 0. Conclusión central (la tesis del pipeline)

Toda la evidencia converge en un único mensaje, que además **mi propio experimento ya confirma**:

> **En S&P 500, la arquitectura del modelo importa poco. El retorno neto se gana con: (1) datos ricos, (2) control de costes/turnover, (3) neutralización, (4) ensemble de modelos simples y (5) la función de pérdida adecuada — NO con redes complejas.**

- Rahimikia et al. (2025, arXiv:2511.18578): sobre ~10.000 acciones US 2001–2023, **CatBoost/LightGBM baten a TODOS los foundation models y redes neuronales**. Los TSFM zero-shot dan R² negativo.
- Gu, Kelly & Xiu (2020, *RFS*): los mejores son **árboles y redes superficiales (NN3-4)**; las profundas (NN5) empeoran. "Shallow beats deep" es el resultado fundacional.
- Toda la "SOTA arquitectónica" con IC alto (MASTER, StockMixer, Mamba) es **solo China A-shares** — no transfiere a US (microestructura distinta; un paper muestra IC inflado 0.058 por límites de precio inejecutables mientras el Sharpe real cae).
- **Mi resultado:** Lasso (5 coefs, IC +0.024) > LightGBM (+0.014) > Stockformer (1M params, **−0.003**). Exactamente lo que predice la literatura.

**Expectativa honesta de Sharpe neto:** ~**0.8–1.3** es un objetivo "bueno" y defendible. Por encima de 1.5 neto, con datos públicos gratuitos, semanal y large-cap, **no está documentado de forma creíble**. No prometo más.

---

## 1. Arquitectura del pipeline (de datos a posiciones)

```
  Datos ricos (PIT)
        │  EDGAR fundamentales + GDELT sentiment + realized-vol + macro
        ▼
  Feature engineering cross-seccional  (≈30-40 features NUEVAS consolidadas)
        │  winsorize → Gaussian-rank por fecha → neutralizar size/sector
        ▼
  Ensemble de modelos simples  (LightGBM + ElasticNet, out-of-fold)
        │  loss de ranking (ListNet/margin), horizonte SEMANAL
        ▼
  Señal alpha cross-seccional  (por semana, por acción)
        │  residualizar vs Fama-French 6 + sector → EWMA smoothing
        ▼
  Construcción de cartera COST-AWARE  (cvxpy)
        │  dollar+beta-neutral, L1 coste, no-trade bands, caps, Ledoit-Wolf, vol-target 10%
        ▼
  Overlay de régimen (VIX terciles → escala gross)
        ▼
  Backtest con el harness unificado (Sharpe neto, MaxDD, turnover, costes 8 bps/lado, lag 1d)
```

---

## 2. Datos: qué conseguir gratis (la palanca nº1)

| Familia | Fuente gratuita | Histórico | Point-in-time | Veredicto |
|---|---|---|---|---|
| **Fundamentales** | **SEC EDGAR XBRL** `companyfacts` (bulk ZIP) | 2009–hoy | **SÍ** (campo `filed`) | **Backbone.** Filtrar `filed ≤ fecha`. Sin leakage. |
| Fundamentales (cross-check) | SimFin free | 10+ años | Parcial (publish date, lag 12m) | Solo backtest, no live |
| Earnings surprises | FMP free (250/día) | multi-año | Débil | Solo para SUE pre-calculado |
| **Noticias/sentiment** | **GDELT (BigQuery)** | 2015–hoy | SÍ (timestamp) | **Única fuente histórica gratis.** Re-scorear con FinBERT |
| Sentiment (live) | Finnhub free + RSS | ~1 año | SÍ | Recolectar **desde ya** para la parte forward |
| **Opciones / IV** | — | — | — | **NO hay histórico gratis.** Usar realized-vol + VIX como proxy; loggear yfinance chains forward-only |
| Macro extra | FRED (credit spread BAA-AAA) | largo | SÍ | Gratis, añadir |

**Advertencias de honestidad (no sobre-prometer):**
- IV/skew/put-call **histórico por acción 2018-2024: imposible gratis** → se sustituye por realized-vol + VIX.
- Reddit/StockTwits/NewsAPI histórico: no fiable → solo GDELT.
- yfinance/FMP fundamentales son **restated (no PIT)** → leakage si se usan para fechar el backtest. Solo EDGAR (y SimFin publish-date) son seguros.

---

## 3. Features: ~30-40 NUEVAS consolidadas (no 150)

Con ~477 acciones × ~400 semanas, las observaciones **cross-seccionales efectivas son ~400** (alta correlación intra-fecha). Más features = sobreajuste. **Consolidar, no acumular.**

| Familia | nº | Features clave | Evidencia |
|---|---|---|---|
| Precio/técnico (ya tengo) | ~consolidar 375 | momentum, reversión, vol, turnover | GKX: la familia más fuerte |
| **Earnings/PEAD** | 3 | SUE con decaimiento (τ≈35d), flag ventana earnings | Bernard-Thomas; **el fundamental más fuerte a horizonte semanal** |
| Profitabilidad/calidad | 4 | gross profitability (Novy-Marx), ROE, QMJ | FF5 RMW |
| Inversión | 2 | asset growth (CMA), net equity issuance | FF5 CMA |
| Value compuesto | 2 | (E/P, S/P, CF/P, EBIT/EV) z-medio | débil solo, usar compuesto |
| Momentum fundamental | 2 | Δ ventas YoY, Δ margen bruto | revisiones más rápidas que niveles |
| **Realized-vol/riesgo** | 4 | IVOL idiosincrático, vol 60d, beta, low-vol | Ang et al. 2006; **gratis, price-only** |
| Sentiment (GDELT) | 4-5 | nivel, cambio, cobertura anómala, dispersión, peor titular | Kirtac-Germano (calibrar gross→net) |
| Macro estado | 5 | VIX nivel+Δ, credit spread, term spread, trend de mercado | regime gates |
| Interacciones macro | 3-4 | momentum×market-state, value×credit-spread | Daniel-Moskowitz 2016 |

**Higiene (por fecha, sin leakage):** winsorize 1/99 → **Gaussian-rank** (mejor que z-score para fundamentales/sentiment) → **neutralizar vs log(mcap) + sector GICS** → `merge_asof` por `filed_date` para fundamentales → flags `was_missing` → CV walk-forward **purgada/embargada**.

---

## 4. Modelo: ensemble de simples + loss de ranking

1. **LightGBM/CatBoost** sobre el feature set rico, neutralizado, horizonte semanal, con penalización de turnover. (Máxima evidencia.)
2. **ElasticNet/Lasso** como segundo miembro (mi Lasso ya bate al Stockformer).
3. **Ensemble out-of-fold** (media) de (1)+(2); meta-learner lineal opcional.
4. **Loss de ranking**: ListNet o margin > MSE (+150-200 bps anuales en arXiv:2510.14156). **Ojo:** optimizar la loss para el objetivo de cartera, no para IC (el mejor IC ≠ mejor retorno).
5. **NO** construir MASTER/StockMixer/Mamba/foundation/Stockformer para US — evidencia China-only o negativa.

---

## 5. Construcción de cartera cost-aware (donde se gana el Sharpe neto)

Esta es **la restricción que ata** todo (mi experimento: el L/S diario pierde por costes). Receta y uplift documentado:

| Paso | Acción | Efecto |
|---|---|---|
| 1. **Semanal** | rebalanceo semanal (no diario) | corta turnover ~5× → **el salto más grande** |
| 2. **Neutralizar** | OLS por fecha: alpha ~ FF6 + log-mcap + dummies sector → residuo | IC honesto, sin sesgos factor/sector |
| 3. **Suavizar** | EWMA halflife ~2-3 sem sobre la señal | menos churn |
| 4. **Dimensionar** | Ledoit-Wolf Σ + vol-target 10% + caps name 2%/sector 10% | drawdown↓, vol estable |
| 5. **Pesos cost-aware** | cvxpy: max(μᵀw − γ wᵀΣw − κ·spread·\|Δw\|) s.a. Σw=0, βᵀw=0, ‖w‖₁≤2, turnover≤30%/sem, no-trade bands | **convierte Sharpe neto en positivo** |
| 6. **Overlay régimen** | VIX terciles → gross ×{1.0, 0.7, 0.4} | Calmar/cola ↑ (RegimeFolio: Sharpe 1.17 vs 0.66, MaxDD −29% vs −41%) |

**Coste realista large-cap:** ~**8 bps/lado** (spread efectivo ~3 bps + impacto + fees), multiplicador de estrés ligado a VIX. **Saltar BPQP/cvxpylayers** (E2E): a frecuencia semanal el predict-then-optimize captura casi todo el beneficio sin el coste de ingeniería.

**Librerías:** `statsmodels` (neutralización), `sklearn.LedoitWolf`, `cvxpy`/`cvxportfolio`, `riskfolio-lib`/`PyPortfolioOpt` (HRP fallback), `hmmlearn` (opcional).

---

## 6. Plan de construcción por tiers (para 4 días)

Reutiliza toda la infraestructura ya validada (`lib/eval_harness.py`, `lib/data_panel.py`).

**Tier A — Núcleo cost-aware (todo local, ~1.5 días) — máximo ROI**
Convierte lo que ya tengo en una estrategia real **sin datos nuevos**:
- Resampling semanal de panel/labels.
- Neutralización FF6+sector (descargar betas FF de Kenneth French; sectores GICS).
- Constructor cost-aware cvxpy (market-neutral, L1 coste, no-trade bands, vol-target, caps).
- Ensemble LightGBM+ElasticNet + loss de ranking.
- Overlay VIX. Backtest con harness extendido (Sharpe neto semanal).
→ *Hipótesis: esto solo ya da Sharpe neto positivo y es el core de la tesis.*

**Tier B — Datos fundamentales (~1-1.5 días) — mayor palanca de señal**
- Pipeline EDGAR XBRL point-in-time (`companyfacts` → PEAD/SUE, gross profitability, asset growth, net issuance, value compuesto) con `merge_asof` por `filed_date`.
- Re-correr ensemble; medir uplift OOS de IC/Sharpe con ablación.

**Tier C — Opcional si sobra tiempo**
- Sentiment GDELT (BigQuery) + FinBERT.
- Realized-vol/IVOL features (gratis, price-only — fácil, alta evidencia; podría subir a Tier A).

---

## 7. Qué NO hacer (ahorra tiempo y credibilidad)
- No construir transformers/Mamba/foundation models para US.
- No usar yfinance/FMP fundamentales para fechar el backtest (leakage).
- No prometer IV histórico por acción ni sentiment Reddit histórico (no hay gratis).
- No explotar interacciones macro pairwise (sobreajuste con ~400 obs).
- No rebalanceo diario.
- No optimización E2E/BPQP por ahora.

---

## Fuentes principales
Gu-Kelly-Xiu 2020 (RFS) · Rahimikia et al. 2025 (arXiv:2511.18578) · "Loss Functions for Stock Ranking" 2025 (arXiv:2510.14156) · Novy-Marx 2013 · Fama-French 2015 · Daniel-Moskowitz 2016 · Kirtac & Germano 2024 (FRL) · RegimeFolio 2025 (arXiv:2510.14986) · Cost-aware Portfolios 2024 (arXiv:2412.11575) · López de Prado 2018 · SEC EDGAR / GDELT / FRED.
