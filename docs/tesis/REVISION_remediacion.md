# Remediación de la revisión del TFM — 22 puntos verificados contra el repositorio

Cada punto de la revisión se ha localizado en el `.tex`, **comprobado contra los datos/scripts/CSV reales** del repositorio (no contra la prosa) y se acompaña de una redacción de reemplazo lista para pegar. `2026-06-11`.

## ESTADO (2026-06-11)

**APLICADO en el `.tex` (22 de 22):** P1, P2, P3, P4 (opción a, γ→λ), P5, P6, P7, P8, P9, P10 (mantener+nota), P11 (fila Stockformer en 5.1 + ElasticNet en 5.2), P12 (nueva §3 «Configuración de entrenamiento» + tabla), P13 (pre-registro citado de forma prudente + frases reformuladas), P14 (párrafo de multiplicidad, k=16), P15, P16 (título → opción B), M1 (Barroso fusionado; `ranking2025loss` con metadatos reales de Kwiatkowski & Chudziak y citado), M2, M3 (unidades en pie de tabla), M4, M5, M6 (Tabla A.4 eliminada). Build verificado limpio (latexmk directo: 0 refs/citas sin resolver, 76 págs).

**P7 y P9 resueltos (2026-06-11):**
- **P7 — falsa alarma del flag, y corregido un error mío.** La Tabla 6.7 NO sale de `results/signal_search.csv`; la genera `scripts/build_rq2_table.py`, que recomputa las 5 filas en vivo sobre la ventana unificada de 311 semanas con `CONFIGS["full"]`. Al ejecutarlo reproduce EXACTAMENTE la tabla, incluida la fila reversal (−0,66 / −1,73 / −1,67 / 1,15 / +0,0138). El −0,66 ≠ `signal_search.csv` porque son cómputos distintos (otra ventana y otra config). El generador imprime además gross del reversal ≈ −0,00, así que mi edición previa («Sharpe bruto también negativo») era incorrecta: revertida a «prácticamente nulo» en §6 y en el caption. Añadida la nota de unidades de la columna Holdout al generador y al `.tex`.
- **P9 — resuelto con `dp.load_panel`.** El split se calcula sobre el panel ALINEADO (1.999 días tras el buffer de 60 filas; `train_end=1499`, `val_end=1749`), no sobre el crudo. Ventanas reales: train 2018-03-29 → 2024-03-13 (1.499), val 2024-03-14 → 2025-03-13 (250), test **2025-03-14 → 2026-03-12 (250)**. El «250 días» del ladder ES el test split; el «258» que reporté antes era un error de índices crudos (`split_indices.json` no es lo que usa `load_panel`). Stockformer = 236 días comunes (su corrida arranca el 2025-04-03), que es la n del test de significancia. §4 y el caption de la Tabla 5.1 corregidos.

---

## Resumen ejecutivo

| # | Sev. | Estado | ¿Decisión del autor? | Asunto |
|---|------|--------|----------------------|--------|
| 1 | crítico | confirmado | no | Auditoría 6/6 vs 5/6 (10 sitios) |
| 2 | alto | confirmado | no | "Único Sharpe LS positivo" es falso (también LightGBM) |
| 3 | alto | confirmado | no | "Señal idéntica" contradice columna IC de Tabla 6.3 |
| 4 | crítico | confirmado | **sí** | Término de riesgo γwᵀΣw en ec. central, pero λ=0 en lo evaluado |
| 5 | crítico | confirmado | no | "offset 60" redactado como horizonte a 60 días |
| 6 | alto | confirmado | no | Holdout descrito como criterio de selección (el código no lo usa así) |
| 7 | alto | confirmado | **sí** | Reversal "sobrevive" vs "pierde" + fila de Tabla 6.7 no reproducible |
| 8 | menor | **NO confirmado** | no | La aritmética de 311 sem. SÍ cuadra; solo falta documentar 2.ª config |
| 9 | crítico | confirmado | **sí** | Ninguna fecha concreta (fechas reales ya extraídas) |
| 10 | medio | confirmado | **sí** | Costes 10 vs 8 bps sin reconciliar |
| 11 | alto | confirmado | **sí** | Tabla 5.1 sin Stockformer; Tabla 5.2 sin ElasticNet |
| 12 | alto | confirmado | no | Cap. 3 no documenta el entrenamiento real (hiperparámetros ya extraídos) |
| 13 | alto | confirmado | **sí** | Pre-registro nunca citado (el artefacto existe) + frase mal formulada |
| 14 | alto | confirmado | **sí** | Sin corrección por multiplicidad (mejor de ~16 ≈ ruido) |
| 15 | medio | confirmado | no | Falta peldaño MLP; reconocido pero mal enmarcado |
| 16 | alto | confirmado | **sí** | Título "estrategia rentable" lo desautoriza el Cap. 7 |
| M1 | menor | parcial | **sí** | Barroso duplicado; arXiv:2510.14156 huérfano (no ausente) |
| M2 | menor | confirmado | no | "Orden de magnitud" → 2,7× |
| M3 | menor | confirmado | no | Columna "Holdout" sin unidades (= Sharpe neto anual. del holdout) |
| M4 | menor | confirmado | no | Fig. 5.1 log con 0 parámetros (recortado a x=1) |
| M5 | menor | confirmado | no | LightGBM "31" = un solo árbol; "config. por defecto" es falso |
| M6 | menor | confirmado | **sí** | Tabla A.4 de un solo número, redundante con A.1 |

**12 puntos aplicables ya** (sin decisión): 1, 2, 3, 5, 6, 8, 12, 15, M2, M3, M4, M5.
**10 requieren una decisión tuya**: 4, 7, 9, 10, 11, 13, 14, 16, M1, M6.

---

## Tres correcciones a la propia revisión (importantes)

**P8 — el revisor se equivoca.** La aritmética "200 entren. + 311 OOS ≈ 511 sem ≈ 10 años" mezcla dos corridas distintas. El panel es de **~427 semanas (~8 años)**, no ~10. La corrida de 215 sem. OOS usa `--init_train 200`; la de 311 sem. usa `init_train=104` (default de `pipeline_rq2.py` / `run_signal_search*.py`). En ambos casos *entren. inicial + OOS ≈ 415*, dentro del panel. No hay contradicción: el único defecto (menor) es que §4 solo documenta la config de ~200 sem. **Bajo este punto de "urgente" a "menor".**

**M1 — matiz.** La entrada de arXiv:2510.14156 **sí existe** en `main.bib` (clave `ranking2025loss`), pero está **huérfana** (nunca se cita con `\autocite`), por eso no sale en la bibliografía. No hay que "añadir una entrada" sino citar la que ya está (y completar sus metadatos, que son de marcador de posición).

**Hallazgo nuevo (dentro de P7) — más serio que lo que cazó el revisor.** La fila "Reversal 1 sem." de la **Tabla 6.7** (Sharpe −0,66; t −1,73; holdout −1,67; turnover 1,15; IC +0,0138) **no coincide con ninguna fila de `results/signal_search.csv`** (rev1w_vol da −1,02/−2,21/−2,72/1,21/0,0123; rev1w da −1,08). El "−0,66" coincide con el campo `sel_sharpe` de `rev1w`, no con sus métricas de periodo completo. Es un problema de procedencia de la tabla que conviene resolver antes de la defensa.

---

## TIER CRÍTICO

### P1 — Auditoría "6/6 / limpia" vs "5/6 con fuga deliberada"  ·  *aplicable ya*

**Veredicto:** confirmado. `scripts/audit_alignment.py` define 6 comprobaciones; la 6.ª es un "Graph leakage flag" documentado como fuga conocida (el grafo se construye con todo el histórico, test incluido) **a favor** del modelo complejo. §4.6 lo dice bien (5/6); resumen, intro, Parte I, discusión y conclusiones dicen "limpia / las seis superadas". La versión **5/6 es la correcta y además más fuerte** (un sesgo concedido al Stockformer que aun así no produce señal).

**Ubicaciones:** `04_metodologia.tex:333-334` (correcta), `01_introduccion.tex:106-108`, `00_resumen.tex:10-11` y `:45-46` (inglés), `05_parte1_transferencia.tex:170-171`, `07_discusion.tex:157-163`, `08_conclusiones.tex:53-54`.

**Redacciones:**
- `01_introduccion.tex:107-108` → "(cinco verificaciones superadas y una sexta que detecta, a propósito, una fuga a favor del modelo complejo) que garantiza que el IC negativo de Stockformer es genuino y no un artefacto: el sesgo concedido juega en contra de la conclusión negativa y, aun así, esta se sostiene."
- `00_resumen.tex:11` → "una auditoría anti-fuga (\textit{leakage}) cuyo único hallazgo es una fuga deliberada a favor del modelo complejo".
- `00_resumen.tex:46` (EN) → "an anti-leakage audit whose only finding is a leak deliberately kept in favour of the complex model".
- `05_parte1_transferencia.tex:170-171` → "La auditoría anti-fuga (Sección~\ref{sec:met-leakage}) confirma que no hay fuga que favorezca espuriamente al Stockformer; la única detectada, en el grafo, juega a su favor y aun así el IC es negativo."
- `07_discusion.tex:158-160` → "verificó que la única fuga detectada (el grafo construido sobre todo el histórico) favorece al modelo complejo, de modo que la conclusión negativa no puede atribuirse a una ventaja indebida en su contra."
- `07_discusion.tex:161-162` → "al ser genuinamente nulo incluso concediendo al modelo complejo la única fuga existente".
- `08_conclusiones.tex:54` → "certificado por una auditoría anti-fuga cuya única incidencia es una fuga deliberada a favor del modelo complejo, lo que confirma que el cero es genuino".
- Las menciones genéricas (`08_conclusiones.tex:88`, `07_discusion.tex:262`) pueden quedarse.

### P4 — Término de riesgo: la ecuación central no describe lo evaluado  ·  **DECISIÓN**

**Veredicto:** confirmado. El "libro" por defecto **no** optimiza con término cuadrático de riesgo. `run_weekly_strategy.py:124` llama a `costaware_weights(...)` sin `Sigma` ni `risk_aversion` ⇒ `risk_aversion=0.0` (default) ⇒ el término se omite (`portfolio.py:46`). El barrido de §6.7.2 (`run_construction.py:61`) tiene `RISK_GRID=[0.0, ...]` con `0.0` como baseline. El pre-registro lo dice: "admite un término risk_aversion·wᵀΣw hoy apagado (risk_aversion=0)". Pero `eq:costaware` (§6.1) lo presenta como componente central con Ledoit-Wolf. Además hay incoherencia de notación: γ en §6.1, λ en §6.7.2.

**Ubicaciones:** `06_parte2_que_funciona.tex:96-99` (ecuación), `:110-116` (prosa), `:381-387` (§6.7.2).

**Decisión:** (a) dejar el término en la ecuación marcándolo opcional con λ=0 por defecto (más fiel al código); o (b) eliminarlo de `eq:costaware` y presentarlo solo en §6.7.2 como extensión (más limpio narrativamente). En ambos casos unificar γ→λ.

**Redacción (opción a):**
```latex
\begin{equation}\label{eq:costaware}
  \max_{w}\quad \alpha^{\top} w
  \;-\; \underbrace{\lambda\, w^{\top}\Sigma\, w}_{\text{opcional, } \lambda=0 \text{ por defecto}}
  \;-\; \frac{\kappa}{10^{4}}\,\bigl\lVert w - w_{\text{prev}}\bigr\rVert_{1}
\end{equation}
```
Prosa (`:110-116`): describir primero el término de coste L1 como central ("operar solo cuando la ventaja supera al coste") y el de riesgo como "penalización opcional… en el libro por defecto está desactivada (λ=0), y su efecto se evalúa por separado en §6.7.2 (Tabla~\ref{tab:construction-risk})".

### P5 — "offset 60" mal descrito (parece horizonte a 60 días)  ·  *aplicable ya*

**Veredicto:** confirmado. El 60 es un **buffer de warm-up** que se descarta; el emparejamiento es **contemporáneo** con etiqueta a 1 día. `data_panel.py:89-95` (`X=X[:T]`, `y=labels[lag:lag+T]`), `Multitask_Stockformer_utils.py:278-290` ("offset=60: features[0]=d_60 pairs with label[60]…correct; offset=59 → leakage"). La línea 44 (etiqueta a 1 día) es correcta; solo falla la frase "y_{t+60} con X_t" (mezcla índices pre- y post-buffer).

**Ubicación:** `04_metodologia.tex:50-57`.

**Redacción:** reemplazar el párrafo "Alineación variable–etiqueta" por:
> El invariante anti-leakage central es el desfase (`ALPHA360_LAG = 60`) de filas entre la rejilla bruta de precios y el panel emparejado. Las variables Alpha360 requieren un buffer de 60 retardos para estar definidas, así que las primeras 60 filas se descartan: la primera fila de variables corresponde al día 60 del calendario original. La etiqueta se recorta con el mismo desfase (`labels[60:]`), de modo que esa fila se empareja con su *propia* etiqueta del mismo día, el retorno a un día $y_{60}=\text{Close}_{61}/\text{Close}_{60}-1$. El emparejamiento es contemporáneo: las variables del día $t$ se asocian al retorno a un día $y_t$; el 60 es solo el número de filas de warm-up que se descartan, no un horizonte de predicción. Este desfase se aplica igual en todas las ramas (Stockformer y tabulares).

### P9 — Ninguna fecha concreta  ·  **DECISIÓN (parcial)** + *aplicable*

**Veredicto:** confirmado. Fechas **reales** extraídas del repo:
- **Panel:** 2018-01-02 → 2026-03-12 (2.059 días, 477 acciones).
- **Split 0,75/0,125/0,125** (`split_indices.json` train_end=1544, val_end=1801): entren. 2018-01-02→2024-02-21 (1.544), val. 2024-02-22→2025-03-03 (257), test 2025-03-04→2026-03-12 (258).
- **Semanal 52 sem.:** 2025-03-14 → 2026-03-06.
- **WF 215 sem.:** ~2022-01-28 → 2026-03-06 (init_train=200).
- **RQ2 311 sem.:** 2020-03-27 → 2026-03-06 (init_train=104); holdout 104 = 2024-03-15 → 2026-03-06.

**Decisión:** el "**250 días**" del ladder no coincide con el test del split (258 días) ni con los 236 días comunes del Stockformer. ¿A qué ventana exacta corresponde la Tabla 5.1 y debe el texto reflejar la diferencia 250/258/236 o homogeneizarse?

**Redacciones (anclar cada cifra una vez):** ver bloque (A)–(H) del plan — añadir paréntesis con fechas en `04_metodologia.tex` (panel, split, WF), captions de `ladder.tex`, `weekly_summary.tex`, `weekly_robustness.tex`, `rq2_signal_search.tex` y menciones de 311/215 en resumen e intro. Cuidado: "2020–2026" describe solo la ventana RQ2, no la de 215 sem. (empieza 2022) ni el split de la Parte I (test 2025-26).

---

## TIER ALTO

### P2 — "El único Sharpe LS positivo"  ·  *aplicable ya*
**Confirmado.** `ladder_results.csv`: lasso +0,91 **y** lightgbm +0,21 (el propio texto, `05:62`, ya dice "+0,21"). Resto negativos.
**Ubicación:** `05_parte1_transferencia.tex:50-52`.
**Redacción:** "…el **mayor** Sharpe \textit{long-short} de la tabla en este \textit{split}, de $+0.91$ (el único otro positivo es LightGBM, $+0.21$), con un retorno anualizado del $+27.4\%$."

### P3 — "La señal es idéntica en las cuatro variantes"  ·  *aplicable ya*
**Confirmado.** Lo que se fija es el **alfa crudo** del walk-forward (`preds`, generado una vez). Cada variante lo transforma: z-score / neutralización por beta / suavizado EWMA. El IC se mide sobre la señal **transformada** (`use`), no sobre el alfa crudo (`run_weekly_robustness.py:134`). Por eso 0,0123→0,0085→0,0033; `full` repite 0,0033 porque voltarget/regime solo escalan pesos.
**Ubicaciones:** `06_parte2_que_funciona.tex:213-218`, `:251-254`, `:258-261`, `:406-407`.
**Redacción:** reformular "se fija el alfa crudo del predictor (no se reentrena)… cada variante lo transforma antes de invertirlo, de modo que la columna IC mide la señal ya transformada que alimenta la cartera en cada etapa; por eso baja de 0,0123 a 0,0033 y coincide entre +cost_aware y full". Leyenda fig.: "El alfa crudo del predictor es el mismo en las cuatro variantes; lo que cambia es la construcción". `:406` "mantuvieron fijo el alfa del predictor".

### P6 — Holdout descrito como criterio de selección  ·  *aplicable ya*
**Confirmado** (y la solución NO es admitir una limitación: el procedimiento es defendible). El código selecciona por **t de Newey-West del periodo completo** (orden por `full_sharpe`; `ok=(full_t>2) and (hold_sharpe>0)`); el holdout solo **confirma el signo**. El estadístico de cabecera (1,82) es el t_NW del periodo completo, no del holdout.
**Ubicación:** `06_parte2_que_funciona.tex:424-428`.
**Redacción:** "…ningún ganador se declara por su IC dentro de muestra. La selección se hace por el $t$ de Newey-West del retorno neto sobre el periodo completo (con el listón de \textcite{harvey2016crosssection}) y por la coherencia económica del factor; el \textit{holdout} reservado no se usa para elegir, sino solo para confirmar a posteriori que el resultado mantiene el signo y no es un artefacto de ventana."

### P7 — Reversal "sobrevive" vs "pierde"  ·  **DECISIÓN**
**Confirmado** + hallazgo nuevo (ver arriba: fila de Tabla 6.7 no reproducible). Reconciliación: (a) **horizonte** distinto (reversión diaria del Lasso vs reversal semanal L/S); (b) **IC vs Sharpe** (IC+ ≈0,013 dentro del ruido no implica cartera rentable). Además: `06:445` dice "Sharpe bruto prácticamente nulo" pero el CSV da −0,37/−0,52 (negativo) y el caption dice "pierde incluso en bruto" — prosa, caption y CSV no concuerdan.
**Ubicaciones:** `05_parte1_transferencia.tex:48-49`, `06_parte2_que_funciona.tex:443-447`, `tablas/rq2_signal_search.tex:3`.
**Decisión:** ¿de qué corrida sale la fila "Reversal 1 sem." de la Tabla 6.7 y debe regenerarse desde el CSV correcto? ¿Es vol-escalado (rev1w_vol) o sin escalar (rev1w)?
**Redacciones:** Cap. 5 nota al pie → acotar a "reversión a un día… componente de ranking; la reversión a una semana convertida en cartera L/S no sobrevive a los costes pese a un IC también positivo". Cap. 6 → "Sharpe bruto **también negativo**, pese a un IC de ranking positivo. La aparente discrepancia con la reversión a un día del Lasso se explica por el horizonte y por la distancia entre IC y retorno…".

### P11 — Tablas con modelos faltantes  ·  **DECISIÓN**
**Confirmado.** Tabla 5.1 (`ladder.tex`) **no** tiene Stockformer (está en A.1 `ladder_full_body.tex`); causa: Stockformer se evaluó sobre 236 días, el resto sobre 250. Tabla 5.2 (`significancia.tex`) **omite** ElasticNet, que sí está en la 5.3 (`power_signtest.tex`); el dato existe en `power_signtest.csv` (ΔIC +0,0251, t 1,5272, p 0,1281, n 236).
**Decisión:** (A) añadir fila Stockformer a 5.1 con nota del horizonte distinto, o mantener 5.1 con 250 días y añadir remisión a A.1; e incluir ElasticNet en 5.2 (recomendado) o quitarlo también de 5.3.
**Redacciones:** fila `stockformer & transformer & 1\,041\,755 & -0.0033 & -0.0227 & -0.3492 & 0.7273 & -4.9526 \\`; fila `elasticnet & stockformer & 0.0251 & 1.5272 & 0.1281 & 236 \\`.

### P12 — Entrenamiento del Stockformer sin documentar  ·  *aplicable ya*
**Confirmado.** Hiperparámetros reales en `config/Multitask_Stock_SP500.conf` + `MultiTask_Stockformer_train.py`: Adam lr 1e-3, weight_decay 1e-5, grad_clip 0,3, ReduceLROnPlateau, max_epoch 50, early_stopping 15, batch 12, seed 1, dropout 0,2, T1=20/T2=2 (se evalúa el último paso), pérdida ranking (ListNet 0,5 + IC 0,3 + MAE 0,2) + clasif. 0,5, 2 capas/1 cabezal/dim 128, DWT Sym2, grafo estático; 1.041.755 parámetros.
**Redacción:** añadir §"Configuración de entrenamiento" al final del Cap. 3 (párrafo + tabla `tab:train-config`) y en `04:44` la frase de reconciliación T2/etiqueta: "El Stockformer opera con horizonte interno $T_2=2$, pero la predicción operativa que entra en el IC y la cartera es la del último paso, alineada con la etiqueta a un día."

### P13 — Pre-registro nunca citado  ·  **DECISIÓN**
**Confirmado.** Se invoca en ≥13 sitios pero nunca se cita `docs/tesis/PREREGISTRO_PARS_US.md`. Fecha interna 2026-06-08; git lo certifica en commit `9c1fefa` (2026-06-09). La frase "el resultado nulo que se pre-registró como desenlace esperado" (`06:374`) confunde criterio con resultado (aunque el propio artefacto usa esa expresión).
**Decisión:** ¿citar con fecha declarada 2026-06-08 + commit 9c1fefa, o describir prudentemente como "pre-registro interno versionado" sin afirmar marca de tiempo verificable anterior al OOS?
**Redacciones:** añadir en §4 nota citando el artefacto (ruta + commit + seed=0) y reformular `06:372-374` separando criterio (mejora solo si Δsharpe > 1 SE, SE≈0,49) de resultado.

### P14 — Sin corrección por multiplicidad  ·  **DECISIÓN**
**Confirmado.** 16 señales independientes (5 clásicas + 5 de arquitectura + 6 del banco). Bajo la nula, **E[max t]≈1,8 (k=16)** ≈ el 1,82 observado; p familiar ≈0,45 (no 0,03). Bonferroni 1 cola 5% ⇒ t≈2,73, próximo al t≈3 de Harvey ya citado en §5. El mejor resultado **no sobrevive** a multiplicidad.
**Decisión:** ¿reportar 16 (señales standalone de las tablas) o ~26 (variantes de los CSV)? Con 26: E[max]≈1,95, Bonferroni t≈2,8 (conclusión idéntica).
**Redacción:** párrafo nuevo al cierre de §6.8 cuantificando E[max] y Bonferroni; reforzar `06:476` ("siendo este el mejor de una búsqueda… no cruza el umbral ajustado por multiplicidad").

### P16 — El título promete más de lo que el cuerpo defiende  ·  **DECISIÓN (autor)**
**Confirmado.** `preambulo.tex:42`: "De la transferencia fallida **a una estrategia rentable**…". El Cap. 7 dice expresamente: "El valor del trabajo no está… en haber encontrado una estrategia rentable sin lugar a dudas" (Sharpe 0,72, t=1,82 < 1,96; sesgo de supervivencia; sin coste de borrow).
**Decisión / opciones:**
- A: "De la transferencia fallida **a la palanca del retorno**: parsimonia frente a Stockformer en el S&P 500"
- B: "**Parsimonia frente a Stockformer en el S&P 500: transferencia fallida y dónde reside el retorno**"
- C: "De la transferencia fallida **a una señal simple bien construida**: parsimonia frente a Stockformer en el S&P 500"

---

## TIER MEDIO

### P8 — Aritmética de 311 semanas  ·  *aplicable ya (degradado a menor)*
Ver "correcciones a la revisión". **No es contradicción.** Solo falta documentar en §4 la 2.ª calibración (init_train ~104 para las 311 sem.).
**Redacción:** insertar en `04:79-84` un párrafo: "En la Parte II se emplean dos calibraciones sobre el panel completo (~427 semanas, ~8 años): la principal con ventana inicial ~200 sem. y 215 OOS; la búsqueda de RQ2 parte antes (~104 sem.) para una ventana OOS más larga de 311 sem. (2020–2026), de las que las 104 últimas son holdout. En ambos casos, ventana inicial + OOS suman la longitud del panel."

### P10 — Costes 10 vs 8 bps  ·  **DECISIÓN**
**Confirmado** (ambas cifras fieles al código: ladder `FEE=0.001`=10 bps; Parte II `COST_BPS=8.0`). Falta reconciliar en el texto.
**Decisión:** mantener ambos con nota justificativa (universo large-cap más líquido + construcción cost-aware; break-even ~20 bps), o homogenizar y **reejecutar** la Parte II.
**Redacción (mantener+nota):** frase tras `04:215` justificando la diferencia y recordando que ninguna conclusión depende de ella.

### P15 — Falta el peldaño MLP  ·  *aplicable ya*
**Confirmado.** La escalera salta de árboles (~10⁴) a Stockformer (~10⁶) sin red superficial; el ganador de Gu-Kelly-Xiu (NN3–NN4) es precisamente lo ausente. Ya reconocido en `07:272-277` pero genérico.
**Redacción:** reescribir el párrafo "Huecos en la escalera" nombrando explícitamente el MLP superficial como "el contraste pendiente más informativo de la Parte I". (Coherencia: `04:246` "Cúspide" vs "L5" del CSV.)

---

## TIER MENOR

### M1 — Bibliografía  ·  **DECISIÓN**
Duplicado **real** de Barroso: `barroso2015momentum` (`main.bib:63-68`) y `barrososantaclara2015momentum` (`:133-139`). arXiv:2510.14156: la entrada `ranking2025loss` (`:116-120`) **existe pero está huérfana** y con metadatos de marcador de posición.
**Acción:** fusionar Barroso (conservar la que tiene `publisher`, repuntar `\autocite` de `06:498`); citar `\autocite{ranking2025loss}` en `02:230` en vez del identificador a mano.
**Decisión:** ¿título y autores reales de arXiv:2510.14156 para completar `ranking2025loss`? (puedo buscarlo yo).

### M2 — "Orden de magnitud" → 2,7×  ·  *aplicable ya*
`03_stockformer.tex:212-215`. 0,064 / 0,0238 = 2,69. Redacción: "…una magnitud que en mercados desarrollados rara vez se sostiene: unas 2,7 veces el mejor IC obtenido aquí (el Lasso, $+0{,}0238$)."

### M3 — Columna "Holdout" sin unidades  ·  *aplicable ya*
Es el **Sharpe neto anualizado sobre las 104 semanas de holdout** (`hold_sharpe`, WEEKS_PER_YEAR=52). Renombrar cabecera a "Sharpe holdout" / "Sharpe hold." en `rq2_signal_search.tex:8` y `rq2_modular_transfer.tex:8` + nota al pie.

### M4 — Fig. 5.1 log con 0 parámetros  ·  *aplicable ya*
`run_ladder_analysis.py:119` recorta a `x=1`; momentum/reversal se pintan en x=1. Añadir al caption (`05:77-79`): "Los controles sin parámetros (momentum y reversal, 0 par.) se sitúan por convención en $x=1$, al no ser representable el cero en escala logarítmica."

### M5 — LightGBM "31" = un solo árbol  ·  *aplicable ya*
**Confirmado.** `n_params = num_trees()*31`; que dé 31 ⇒ `num_trees()=1`: el early stopping detuvo el boosting en la 1.ª iteración. Además "configuración por defecto" es **falso** (huber, lr 0,01, subsample 0,7, colsample 0,5, reg L1/L2 ajustados; solo `num_leaves=31` es default).
**Redacción** (`05:61-63`): "…la parada temprana detuvo el boosting en la primera iteración, de modo que el ensemble se reduce a un único árbol de 31 hojas (el `num_leaves` por defecto). El gradient boosting no llega a construir un ensemble." Quitar "en su configuración por defecto".

### M6 — Tabla A.4 de un solo número  ·  **DECISIÓN**
**Confirmado.** `tab:ablation` = una celda (IC 0,0136), redundante con A.1 y con `05:62`. Las "métricas de error" prometidas no existen en el CSV (nan).
**Decisión:** eliminar (plegar en una frase que referencie A.1, recomendado) o enriquecer con la fila completa del LightGBM.

---

*Plan generado a partir de 22 investigaciones paralelas, cada una verificada contra `results/*.csv`, `output/`, `scripts/*.py`, `config/` y `docs/tesis/PREREGISTRO_PARS_US.md`.*
