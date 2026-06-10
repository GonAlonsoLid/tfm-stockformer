# Diseño — Búsqueda de señal ampliada y transferencia modular del Stockformer

**Fecha:** 2026-06-10
**Estado:** propuesta de diseño (pendiente de revisión del autor)
**Ámbito:** extensión de la Parte II del TFM (`MEMORIA/tfm/`) + experimentos CPU sobre el pipeline existente.

---

## 1. Contexto y objetivo

La tesis responde RQ1 (Stockformer no transfiere al S&P 500; IC diario ≈ −0,003) y RQ2 (el
retorno neto viene de la construcción cost-aware, no de la señal; el momentum residual eleva el
Sharpe robusto a 0,72 sobre 311 semanas).

**Objetivo de esta extensión:** elevar la **búsqueda de señal** a eje central de RQ2 y probar si
piezas concretas de la arquitectura del Stockformer —que como un todo no transfiere— **destiladas a
señales simples y causales**, aportan contenido sobre el sustrato eficiente. Si no bastan, un banco
ampliado y **pre-registrado** de factores establecidos busca un positivo defendible.

**Restricción dura:** la DGX no está disponible → **CPU únicamente**. No se reentrena el Stockformer:
sus sesgos inductivos se computan como transformaciones de features / señales y se enchufan al
predictor *shallow* existente.

## 2. El reencuadre narrativo (independiente del resultado)

RQ2 pasa de "qué etapa del pipeline genera retorno" (respuesta: construcción) a **dos palancas
complementarias**:
1. **Construcción de cartera cost-aware** (palanca principal, ya establecida por la atribución).
2. **Búsqueda de señal disciplinada** que, sobre el predictor de features existente, identifica
   señales simples con contenido genuino —incluidas señales **destiladas de los sesgos inductivos del
   Stockformer**— que lo **complementan** (no lo sustituyen): se evalúan como `ensemble + señal`.

**Puente RQ1 ↔ RQ2:** lo profundo no transfiere en bloque, pero sus *ideas* (grafo de relaciones,
filtrado frecuencial), reducidas a señales simples y bien hechas (causales, sin fuga), merecen
probarse. Cierre natural del arco *shallow-beats-deep*.

**Robustez narrativa:** el reencuadre NO depende de que los trasplantes ganen. El momentum residual
ya ancla la sección. Si los nuevos candidatos aportan → refuerzan RQ2; si no → resultado negativo
honesto que refuerza RQ1. En ambos casos la estructura aguanta.

## 3. Señales (todas causales, sin fuga)

### Lote 1 — destiladas de la arquitectura del Stockformer

- **Señal de pares / grafo** *(destila el módulo de grafo + atención espacial)*.
  - Grafo de **correlación causal**: cada semana, correlaciones de retornos en ventana móvil
    **solo de pasado** (~120 días). Reutiliza/adapta `lib/dynamic_graph.py`, pero causal — corrige
    la fuga del check #6 del propio Stockformer.
  - Vecinos: top-k por correlación (k≈10–20).
  - Señal: media ponderada por correlación del **retorno reciente de los vecinos** (momentum de
    red / lead-lag; Cohen–Frazzini, momentum sectorial).
  - *Variante de reserva:* pares por sector GICS.
- **Señal de tendencia filtrada** *(destila la descomposición wavelet/dual-frecuencia)*.
  - Filtro **causal** (STL leak-free de `lib/decomposition.py` o wavelet por ventana pasada) para
    extraer la componente de baja frecuencia (tendencia).
  - Señal: **momentum de la tendencia filtrada** (familia momentum, que es la que funciona aquí).
  - *Variante de reserva:* reversal sobre el residuo de alta frecuencia.

### Lote 2 — banco de reserva pre-registrado (factores establecidos, CPU-feasible, datos presentes)

- **Baja volatilidad idiosincrática / IVOL** (Ang 2006) — `realized_vol.npz`.
- **Betting-against-beta** (Frazzini–Pedersen 2014).
- **Momentum gestionado por volatilidad** (Barroso–Santa-Clara 2015) — complementa al residual.
- **Estacionalidad** (retorno histórico del mismo mes; Heston–Sadka 2008).
- **Proximidad al máximo de 52 semanas** (George–Hwang 2004).
- **Trend / time-series momentum multi-horizonte** (`trend_indicator.npz`).

## 4. Marco de evaluación y disciplina

- **Modo señal**: cada candidato se mide **solo** y **sumado al ensemble** (estandarizados
  transversalmente), por la **misma construcción cost-aware** (`lib/portfolio.py`, `neutralize.py`,
  `regime.py`), **walk-forward de 311 semanas (2020–2026)**, **holdout reservado de 104 semanas**.
- **Pre-registro:** la lista completa de candidatos (Lote 1 + Lote 2) se **fija antes** de mirar el
  holdout. No hay búsqueda adaptativa.
- **Multiplicidad:** se reporta el número de candidatos; el listón de $t$ se sube en consecuencia
  (coherente con Harvey–Liu–Zhu, ya citado). El ganador se defiende por **holdout positivo +
  coherencia económica**, no por un IC in-sample suelto.
- **Métricas por fila:** IC, Sharpe neto, $t_{NW}$, holdout, turnover.

## 5. Tabla y criterios de éxito

Tabla de dos paneles (extiende `tablas/rq2_signal_search.tex`):

```
PANEL A — Factores clásicos          Sharpe  t_NW  Holdout  IC      (ya existe)
  Ensemble ML (base)                 +0.38  0.95   +0.20   0.0009
  ...                                  ...
  Ensemble + momentum residual       +0.72  1.82   +0.55   0.0204
PANEL B — Señales destiladas + banco pre-registrado            (nuevo)
  <cada candidato: solo y ensemble+señal>   ?     ?      ?      ?
  Ensemble + mejor combinación               ?     ?      ?      ?
```

**Criterio de éxito (pre-registrado):** un candidato "aporta" si su IC es **holdout-positivo** y su
Sharpe neto **bate la base de 0,38** (idealmente se acerca o supera 0,72). Se fija antes de mirar el
holdout.

## 6. Integración en código + tests

- **Módulo nuevo `lib/transplant_signals.py`**: `peer_graph_signal(...)` y `filtered_trend_signal(...)`
  (+ los factores del banco, o en `lib/baseline_signals.py`).
- **Registro** de los candidatos en el harness de búsqueda de señal (`scripts/run_signal_search.py` /
  `scripts/pipeline_rq2.py`), mismo enganche que el momentum residual.
- **Salida** a `results/` y tabla con `scripts/build_rq2_table.py`.
- **Tests de causalidad** (`tests/test_transplant_signals.py`): perturbar datos en $t' > t$ no puede
  cambiar la señal en $t$ — la misma disciplina anti-fuga del resto de la auditoría. No negociable.
- **Ejecución:** el banco es vergonzosamente paralelo (cada candidato es independiente); en la fase
  de ejecución se puede paralelizar con un workflow (un agente por candidato → tabla) para
  exhaustividad y velocidad.

## 7. Ubicación en la tesis y ediciones de texto

- **§6.8 se eleva a "Búsqueda de señal"** con cinco bloques: marco/disciplina, factores clásicos,
  destiladas (Lote 1), banco (Lote 2), síntesis. Puente desde §6.7 (PARS-US).
- **Subir a central en RQ2 (independiente del resultado, se escribe ya):** RQ2 (§1.2) → dos palancas;
  contribución (§1.3) suma la búsqueda de señal; resumen ES/EN y conclusiones §8.1 la mencionan como
  eje.
- **Dependiente del resultado (tras correr):** números de la tabla, qué señales sobreviven, y la
  redacción del/los hallazgos.

## 8. Viabilidad (confirmada 2026-06-10)

Presente y CPU-runnable: `data/Stock_SP500_2018-01-01_2026-03-16/` con 380 features, `ohlcv/`,
`flow.npz`, `label.csv`, `tickers.txt`, `realized_vol.npz`, `trend_indicator.npz`, `fundamentals.npz`.
`lib/dynamic_graph.py` (grafo del Stockformer) y `lib/decomposition.py` (STL leak-free) existen.
Pipeline de búsqueda de señal y CSVs previos en `results/`.

## 9. Manejo del resultado

Las filas del Panel B se reportan con su número real, **gane o pierda**:
- **Positivo** → refuerza la nueva RQ2 (señal destilada de la arquitectura que sí aporta).
- **Plano** → resultado negativo honesto que refuerza RQ1; el momentum residual sigue siendo el
  positivo de la tesis.

## 10. Fuera de alcance (framed-only, trabajo futuro)

Atención espacial densa, TCN, descriptores estadísticos tipo catch22/tsfresh, optimización
end-to-end. Se describen como ejes de continuación, no se ejecutan en este TFM.
