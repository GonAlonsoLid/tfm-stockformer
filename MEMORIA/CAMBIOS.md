# Registro de cambios de la memoria (11 de junio de 2026)

Revisión mayor de la memoria aplicando la crítica del documento "Revisión crítica y plan
de corrección del manuscrito sobre Stockformer y S&P 500" y las indicaciones adicionales
(WhatsApp). El PDF anterior está guardado en `MEMORIA/versiones_anteriores/main_2026-06-11.pdf`.

## 1. Tesis central rebajada a formulación condicional (crítica: prioridad alta)

- Resumen, introducción (1.3) y conclusiones (8.1) reescritos. "El veredicto es negativo.
  Stockformer no transfiere" pasa a "En esta implementación y sobre el universo analizado,
  Stockformer no muestra evidencia robusta de transferibilidad al S&P 500".
- "Y explica el retorno" pasa a "candidato plausible/prometedor, no confirmado": el
  momentum residual se presenta como evidencia direccional en todo el texto (resumen,
  6.6, 7.4, 8.1).
- Eliminadas las fórmulas de autolegitimación: "La honestidad metodológica forma parte de
  la contribución", "La cifra honesta es...", "La lección del trabajo...", "El fallo está
  sobredeterminado" (como titular), etc.
- La introducción ya no adelanta cifras concretas de resultados (1.3 describe la
  naturaleza de las contribuciones; los números quedan en los capítulos empíricos).

## 2. Alcance del universo acotado (crítica: prioridad alta)

- En introducción, estado del arte (2.1), metodología (4.1) y limitaciones (7.4) se
  explicita que el S&P 500 es el segmento large-cap y una aproximación del mercado, no
  "el mercado estadounidense", citando la metodología oficial de S&P DJI.
- El sesgo de supervivencia del universo (componentes actuales con cobertura completa,
  Yahoo Finance, sin reconstrucción histórica) se declara ya en 4.1 y se repite como
  amenaza a la validez, con CRSP citado como alternativa de calidad investigadora.
- No se reconstruyó el universo point-in-time (exigiría rehacer los datos): se aplicó la
  salida mínima que admite la propia crítica, la rebaja explícita del alcance.

## 3. Amenazas a la validez adelantadas (crítica: estructura)

- Nueva sección 4.7 "Amenazas a la validez": universo/supervivencia, fuga del grafo,
  ausencia de reproducción en origen, una sola semilla + pérdida adaptada, y estado real
  del "prerregistro". La introducción remite a ella desde 1.3.
- La auditoría de fugas (4.6) ya no presenta la fuga del grafo solo como "refuerzo" de la
  conclusión: se documenta como ventaja deliberada al modelo complejo que condiciona la
  lectura de los resultados.
- "Pre-registrado" sustituido en todo el texto por "criterios fijados de antemano y
  documentados (en el repositorio de código)", al no existir un depósito externo trazable.

## 4. Coherencia §4.5 (crítica: coherencia)

- Eliminada la afirmación de que mantener la regresión estándar en LightGBM/XGBoost
  "aísla la arquitectura": ahora se explica que la comparación NO separa arquitectura y
  pérdida, y que se diseña deliberadamente como comparación favorable al modelo complejo.
- Nueva tabla de notación y siglas (Tabla 4.1) con IC, ICIR, tNW, muestra de reserva,
  avance temporal, rotación, pb, etc., referenciada desde las tablas de resultados.
- Enlace explícito entre el horizonte diario de RQ1 y la explotación semanal de RQ2 en
  1.2, 4.1 y arranque de la sección 6.6.

## 5. Capítulo 2 reestructurado (crítica: estructura)

- Arranque reescrito: "Este capítulo revisa los antecedentes necesarios para formular las
  preguntas de investigación...", sin anunciar la tesis ni el veredicto.
- §2.1 con citas primarias: Fama (1970) para eficiencia, Liu, Stambaugh y Yuan (2019)
  para la estructura del mercado chino.
- Eliminadas las menciones a "pilares de la tesis" dentro del estado del arte.

## 6. Capítulo 6 (indicaciones WhatsApp)

- Eliminada la antigua sección 6.6 "Ablaciones de datos (Tier B/C)" y su apoyo en el
  apéndice (sección A.2 con las tablas robustez_fund y robustez_realized).
- Eliminada la antigua sección 6.7 "Los límites de la construcción cost-aware" completa
  (frontera de costes, blend por régimen, término de riesgo), con sus tablas
  (construction_cost, construction_risk, pars_us) y la figura construction_frontier.
  Los ficheros .tex de esas tablas siguen en `tablas/` pero ya no se incluyen.
- Término λ·wᵀΣw eliminado de la ecuación (6.1) y de su explicación: como λ=0 en todo el
  libro por defecto (y la sección que lo evaluaba se ha eliminado), la formulación queda
  solo con retorno esperado y penalización L1 de costes. Se indica que el control del
  riesgo cuadrático queda delegado en las restricciones y el dimensionado por volatilidad.
  La referencia a Ledoit-Wolf se mantiene solo en el estado del arte (2.8).
- La antigua 6.8 (ahora 6.6, "La señal de la estrategia final: momentum residual"):
  - Se explica de forma explícita la diferencia entre la estrategia base del resto del
    capítulo (señal del ensemble sola, 215 semanas desde 2022) y la evaluación de esta
    sección (311 semanas desde 2020, con 104 de reserva, base 0,38).
  - El momentum residual se integra como pieza de señal de la construcción de cartera de
    la estrategia final, no como añadido externo (nuevo párrafo tras la tabla de búsqueda).

## 7. Capítulo 7 simplificado (WhatsApp: "simplificar 7", "7.3 out")

- Eliminada la sección 7.3 "Honestidad sobre el Sharpe" y la figura del mapa de calor
  (metrics_heatmap), que la crítica también señalaba por el z-score por columna. Su
  contenido esencial (0,95 ventana favorable frente a 0,58±0,49 robusto) ya estaba y se
  conserva en 6.3-6.4; la cautela económica pasa al último párrafo de limitaciones.
- 7.1, 7.2 y 7.4 (antes 7.5) recortados y desdramatizados; el capítulo pasa de 5 a 4
  secciones.

## 8. Tono, negritas/cursivas y anglicismos (crítica: estilo; WhatsApp)

- Eliminadas todas las negritas y cursivas de la prosa (\textbf, \textit, \emph) en los
  ocho capítulos, el resumen, el apéndice y las tablas.
- Anglicismos traducidos con forma estable (el término inglés se introduce una vez):
  harness → marco de evaluación; complexity ladder → escalera de complejidad; leakage →
  fuga de información; holdout → muestra de reserva; walk-forward → protocolo de avance
  temporal; cost-aware → construcción con control de costes; features → variables;
  ensemble → combinación de modelos; drawdown → caída máxima; turnover → rotación;
  backtest → simulación histórica; momentum crash → desplome de momentum; etc.
- Criterio terminológico declarado en nota al pie de la introducción, citando el
  Diccionario panhispánico de dudas (RAE).
- Símbolos ≈, ∼ y ± eliminados de la prosa (sustituidos por "en torno a", "unas", o el
  valor exacto); se conservan solo en tablas, ecuaciones y resultados formales.
- Frases del resumen divididas (una idea metodológica por frase); siglas definidas en su
  primera aparición; tNW fuera del resumen.

## 9. Títulos analíticos (crítica: estructura/estilo)

- Cap. 5: "Parte I: ¿Transfiere Stockformer?" → "Parte I: transferencia de Stockformer
  al S&P 500".
- Cap. 6: "Parte II: ¿Qué sí funciona?" → "Parte II: atribución del retorno neto".
- 6.8: "¿Dónde reside el contenido predictivo de la señal?" → "La señal de la estrategia
  final: momentum residual". 6.7.2 ("¿Se puede mejorar el libro simple?...") eliminada
  con su sección.
- Otros: "Harness de evaluación" → "Marco de evaluación"; "La complexity ladder" → "La
  escalera de complejidad"; "Auditoría anti-leakage" → "Auditoría de fugas de
  información"; "Robustez walk-forward" → "Robustez con avance temporal".
- El título de portada NO se ha tocado (suele estar registrado oficialmente). La crítica
  sugiere como alternativa: "Transferibilidad cross-market de Stockformer al S&P 500 y
  atribución del retorno neto". Decisión pendiente del autor.

## 10. Figuras y tablas (crítica: leyendas)

- Captions reescritas en tono descriptivo (qué se muestra, ventana, estimador, coste),
  sin conclusión interpretativa: figuras 5.1, 5.2, 6.1, 6.2, 6.3 y todas las tablas.
- Tabla 5.1 (escalera): la caption advierte de la ventana heterogénea (250 frente a 236
  días) y remite a la Tabla 5.2 (días comunes) para las comparaciones. No era posible
  recalcular una versión en ventana común sin re-ejecutar el pipeline.
- Tablas de la Parte II: notas fijas con definiciones (SE, reserva, tNW, rotación, coste
  por lado) y remisión a la tabla de notación 4.1. "Holdout" → "Reserva" en cabeceras.

## 11. Bibliografía ampliada (crítica: referencias)

Nuevas entradas, todas citadas en el texto (de 21 a 32 referencias):
- S&P U.S. Indices Methodology (S&P DJI) — alcance del índice, GICS.
- SEC EDGAR Application Programming Interfaces — fundamentales point-in-time.
- CRSP US Stock Databases — alternativa sin sesgo de supervivencia.
- Yang et al. (2020), Qlib (arXiv:2009.11189) — origen de Alpha158/Alpha360.
- Chen y Guestrin (2016), XGBoost — referencia canónica.
- Ke et al. (2017), LightGBM — referencia canónica.
- Newey y West (1987) — fundamento del estadístico tNW.
- RAE, Diccionario panhispánico de dudas — criterio de extranjerismos.
- Fama (1970) — hipótesis de mercado eficiente (§2.1).
- Liu, Stambaugh y Yuan (2019), "Size and value in China" — microestructura china (§2.1).
- Yahoo Finance — fuente de datos citada formalmente.
- Stockformer completada con datos definitivos: Ma, Xue, Lu y Chen, Expert Systems with
  Applications 273 (2025) 126803; preprint arXiv:2401.06139.

Pendiente de verificar por el autor: la referencia Rahimikia y Poon (2025, "en prensa")
no se ha podido confirmar con esos datos exactos; conviene localizar la versión final o
sustituirla por la fuente correcta antes del depósito.

## 12. Exigencias de la crítica NO aplicables editorialmente

Estas requieren re-ejecutar experimentos y quedan fuera de esta revisión; el texto ya
declara su ausencia como límite (4.7 y 7.4) y rebaja las conclusiones en consecuencia:
- Recalcular resultados principales sin la fuga del grafo.
- Universo point-in-time con delistings.
- Reproducción del modelo en el mercado de origen (CSI 300/500).
- Varias semillas y ablación con la pérdida original.
Se recogen como prioridades en "Trabajo futuro" (8.2).

## Nota sobre el PDF

`tfm/main.pdf` se ha regenerado en un entorno sin los paquetes babel-spanish, siunitx y
biblatex (se sustituyeron localmente por equivalentes: nombres en español manuales y
natbib/unsrtnat). El resultado es fiel en contenido y numeración, pero la separación
silábica y el formato exacto de la bibliografía pueden diferir ligeramente del build
local. Recomendado: ejecutar `./build.sh` en tu máquina para la versión definitiva.
Las fuentes del repositorio (preambulo.tex, main.tex) NO se han modificado en ese
aspecto y compilan igual que antes.
