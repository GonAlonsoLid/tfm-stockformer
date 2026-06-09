# Pre-registro — PARS-US (predict-shallow, construct-smart, condition-on-regime)

**Fecha:** 2026-06-08 (fijado ANTES de evaluar sobre el tramo out-of-sample).
**Autor:** Gonzalo Alonso Lidón.
**Script:** `scripts/run_pars_us.py` (semilla fija `seed=0`).

Este documento fija, antes de mirar ningún resultado OOS, la hipótesis, los
hiperparámetros y el criterio de éxito de la contribución constructiva del TFM. Su
propósito es blindar el análisis contra el *p-hacking* y la selección de ventana: cualquier
desviación posterior se reportará como tal.

## 1. Hipótesis y contribución

Stockformer concentra ~1,04 M de parámetros en el **predictor** y deja la construcción de
cartera trivial (ordenación por deciles). PARS-US invierte ese reparto: predictor
parsimonioso (ensemble shallow LightGBM + ElasticNet ya validado en la escalera) +
construcción convexa cost-aware, y sitúa el **único objeto aprendido nuevo** en la
*combinación* de los dos aprendices, condicionada por el régimen de volatilidad, no en el
predictor por nombre.

Concretamente, el blend constante actual (0,5·GBM + 0,5·ENet sobre predicciones
estandarizadas) se sustituye por un blend con un peso por régimen, `w_r` para r ∈ {low,
normal, high}, de modo que la señal combinada es `w_r·z(GBM) + (1−w_r)·z(ENet)`. Son **3
grados de libertad**, congelados tras estimarse una sola vez sobre el OOF del tramo de
entrenamiento inicial.

## 2. Hiperparámetros fijados (no se ajustan a OOS)

Heredados del pipeline ya auditado (`run_weekly_strategy.py` / `run_weekly_robustness.py`):

- Coste: `cost_bps = 8` (por lado). Gross `= 2`. `name_cap = 0,04`. `target_vol = 0,10`.
- Suavizado EWMA `half-life = 2` semanas. Ventana de beta `= 60` días.
- Walk-forward: `init_train = 200` semanas (~4 años), `step = 26` (reentreno semestral).
- Aprendices: LightGBM (objetivo huber, 300 árboles, lr 0,02, 31 hojas, …) + ElasticNet
  (`alpha = 1e-4`, `l1_ratio = 0,5`), exactamente como en `train_ensemble`.

Nuevos, fijados aquí:

- Régimen: serie **VIX** del panel (`features/MACRO_VIX_level.csv`), que viene
  estandarizada (z-score), por lo que no aplican los umbrales en niveles 15/25. Se usan
  **terciles congelados en el tramo de entrenamiento inicial** (low por debajo del
  percentil 33, high por encima del 67, normal en medio), asignados a cada semana por el
  VIX conocido en su fecha de rebalanceo (sin look-ahead).
- Estimación del blend por régimen: partición temporal interna del tramo de entrenamiento
  inicial (último `inner_frac = 0,3` como validación, con `embargo = 1` semana). Para cada
  régimen, búsqueda en rejilla `w_r ∈ {0; 0,1; …; 1}` que maximiza el IC de Spearman
  *pooled* de la señal combinada frente al retorno realizado. Si un régimen tiene menos de
  `min_weeks = 8` semanas internas, `w_r = 0,5` (default neutro). Los `w_r` se **congelan**
  y se reutilizan en todas las ventanas walk-forward (no se reestiman por ventana).

## 3. Criterio de éxito (pre-registrado)

Se compara, sobre el **mismo** walk-forward de 215 semanas y la **misma** construcción
`full` cost-aware, el blend por régimen frente al blend constante 0,5/0,5.

- Se declarará **mejora** únicamente si `Sharpe_neto(régimen) − Sharpe_neto(base) > 1 SE`
  (con SE ≈ 0,49 sobre 215 semanas). En cualquier otro caso, el resultado es **NULO** y se
  acepta como tal: es el desenlace esperado de antemano.
- Métricas secundarias que se reportan en todo caso (no condicionan el veredicto): MaxDD,
  turnover medio, y el ICIR por régimen (la hipótesis más plausible es una estabilización
  modesta del ICIR entre regímenes, no una subida del IC medio).

## 4. Lo que NO se afirmará

- Que PARS-US mejora el IC medio: el blend reordena dos aprendices ya rank-comparables; se
  espera un IC combinado estadísticamente indistinguible del blend constante.
- Que cualquier diferencia de Sharpe es significativa: con 215 semanas y SE ≈ 0,49, una
  diferencia de menos de ~0,49 no es distinguible de cero y se reportará así.
- Que el régimen es una "pieza nueva de gestión de régimen": el pipeline `full` ya tiene un
  *overlay* de régimen sobre los pesos; la comparación honesta es blend-por-régimen vs
  blend-constante, manteniendo idéntico todo lo demás.

## 5. Auditoría

- Se contarán y reportarán las semanas en que el solver cost-aware devuelve libro-cero
  (`lib/portfolio.costaware_weights` → ceros), para no inflar el Sharpe con semanas vacías.
- El blend solo ve datos `≤` la fecha de rebalanceo; el régimen, el VIX `≤` esa fecha.

---

# Parte II — Estudio de construcción (frontera coste-Sharpe + término de riesgo)

**Fecha:** 2026-06-08 (fijado ANTES de evaluar OOS). **Script:** `scripts/run_construction.py`.

La atribución del TFM muestra que el valor económico vive en la construcción cost-aware, no
en la señal. Esta parte profundiza esa capa sobre el **mismo** walk-forward de 215 semanas y
la **misma** señal (ensemble constante 0,5/0,5, congelada), variando únicamente parámetros de
construcción. La señal se calcula una sola vez y se cachea; cada barrido reutiliza esa señal.

## A. Mapa del espacio de construcción (descriptivo, sin criterio de éxito)

Para cada perilla, manteniendo el resto en su valor por defecto (coste 8 bps, gross 2,
name_cap 0,04, half-life 2, vol-target 0,10), se reporta el Sharpe neto ± SE sobre las 215
semanas:

- `cost_bps ∈ {0; 2; 5; 8; 10; 15; 20; 30}` → curva Sharpe-coste y **break-even**.
- `gross ∈ {1; 1,5; 2; 2,5; 3}`.
- `name_cap ∈ {0,02; 0,03; 0,04; 0,06; 0,10}`.
- `smooth_halflife ∈ {1; 2; 3; 4}`.
- `target_vol ∈ {0,06; 0,08; 0,10; 0,12; 0,15}`.

Es un análisis de sensibilidad: caracteriza dónde es robusto el resultado y cuál es el coste
que lo anula. No se selecciona la perilla "ganadora"; el punto por defecto está pre-fijado.

## B. Término de riesgo con covarianza shrinkage (Ledoit-Wolf)

El optimizador `costaware_weights` admite un término `risk_aversion · wᵀΣw` hoy apagado
(`risk_aversion = 0`). Se activa con `Σ` = covarianza de retornos diarios de la ventana de 60
días con shrinkage de **Ledoit-Wolf** (`sklearn.covariance.LedoitWolf`; `Σ` por semana,
calculada solo con datos `≤` rebalanceo). Se barre
`risk_aversion ∈ {0; 10²; 10³; 10⁴; 10⁵; 10⁶}` (rejilla log que cubre el rango donde el
término muerde, dada la escala de la covarianza de retornos diarios; el `0` es el baseline).

**Criterio de éxito (pre-registrado):** se declara mejora solo si, para algún `λ` de la
rejilla, el Sharpe neto supera al baseline (`λ=0`) en `> 1 SE` (≈0,49), **o** si el MaxDD se
reduce de forma material (≥ 3 puntos porcentuales) sin empeorar el Sharpe. En otro caso, NULO,
aceptado de antemano. La hipótesis es un control de riesgo que mejore el MaxDD (hoy −11,5 %),
posiblemente con efecto modesto o nulo sobre el Sharpe. Métricas reportadas en todo caso:
Sharpe neto ± SE, MaxDD, gross Sharpe, turnover, por `λ`.

**No se afirmará** que el término de riesgo aporta señal predictiva (no la aporta: solo
reasigna el riesgo de la misma señal). Cualquier mejora de Sharpe < 1 SE se reporta como no
distinguible de cero.
