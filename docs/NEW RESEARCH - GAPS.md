# Cómo rescatar un Stockformer que fracasa en el S&P 500

**El modelo Stockformer adaptado al S&P 500 no necesita un parche menor: necesita una reingeniería integral de features, función de pérdida, estrategia de trading y régimen de entrenamiento.** Con un IC de −0.003, el modelo actual genera predicciones indistinguibles del azar — o peor, ligeramente invertidas. La evidencia reciente (2024–2026) identifica las causas raíz: las features Alpha360 carecen de señal en mercados eficientes, la función de pérdida MSE no optimiza ranking, la estrategia long-only con beta 1.33 es un proxy apalancado del mercado, y la complejidad del modelo (transformer + wavelets + grafos + multitask) probablemente sobreajusta al ruido. La buena noticia: existe un conjunto claro de intervenciones, respaldadas por papers recientes, que pueden transformar estos resultados.

---

## El diagnóstico: por qué Alpha360 no funciona en mercados eficientes

El problema más crítico del modelo actual no es la arquitectura, sino el **input**. Alpha360 — 6 campos OHLCV con 60 lags diarios normalizados por z-score cross-seccional — fue diseñado para el mercado chino (CSI 300/500), donde la microestructura es radicalmente distinta: liquidación T+1, límites de precio del ±10%, dominio de inversores retail, y correlaciones más predecibles entre acciones. En el S&P 500, donde el **85% del volumen es institucional o algorítmico**, las ineficiencias capturables con datos OHLCV puros son mínimas.

La evidencia empírica lo confirma rotundamente. En los benchmarks oficiales de Qlib sobre CSI 300, un Transformer vanilla con Alpha360 alcanza un IC de apenas **0.0114** — el peor de todos los modelos evaluados — mientras que LightGBM con Alpha158 (158 indicadores técnicos ingeniados) logra **IC 0.0448** e IR de 1.016. MASTER (AAAI 2024), el estado del arte actual en predicción cross-seccional, alcanza IC ~0.064, pero exclusivamente en mercados chinos. Para el S&P 500, incluso los mejores modelos producen ICs del orden de **0.01–0.05** en datos diarios — un IC negativo como −0.003 indica un problema fundamental de señal, no un problema de capacidad del modelo.

La investigación de Wang (2025, *Journal of International Money and Finance*) aporta un hallazgo clave: usando **920 features** (incluyendo fundamentales, momentum, liquidez y factores macroeconómicos) sobre todas las acciones estadounidenses desde 1957 hasta 2021, Autoformer supera consistentemente a redes neuronales simples en horizontes de 1, 3 y 12 meses. Los predictores dominantes no son técnicos sino fundamentales: **net equity expansion, return on assets, sales-to-price, y dummies sectoriales**. Esto sugiere que para el S&P 500, la ampliación de features hacia datos fundamentales y macroeconómicos es imprescindible.

Un estudio de spillovers globales (2025) encontró que para predecir el S&P 500, las señales más fuertes provienen de índices globales: DJIA (42%), KOSPI (18%), Russell 2000 (15%). La incorporación de **datos macro** (VIX, curva de rendimientos, spreads de crédito) y **sentiment de noticias** es crítica. Kirtac & Germano (2024, *Finance Research Letters*) demostraron que sentiment derivado de LLMs (OPT, GPT-3) sobre 965K artículos financieros alcanza **74.4% de accuracy** en clasificación de dirección, con estrategias long-short generando **Sharpe de 3.05** — muy superior al Sharpe negativo del modelo puramente técnico.

**Recomendación prioritaria**: Reemplazar Alpha360 por un conjunto híbrido que incluya Alpha158, factores fundamentales (P/E, ROA, earnings surprises), datos macroeconómicos (VIX, tasas, yield curve), sentiment vía FinBERT/LLM, y señales cross-market (índices globales). Para la integración multimodal, la arquitectura MSGCA (2024–2025) propone **gated cross-attention** que maneja la fusión de series temporales + texto + grafos, mejorando entre 8–31% sobre métodos unimodales.

---

## Modelos estado del arte: qué realmente funciona en 2024–2026

### Transformers especializados para acciones

**MASTER** (Li et al., AAAI 2024) es actualmente el modelo de referencia en predicción cross-seccional de acciones. Su innovación central es modelar correlaciones inter-stock tanto momentáneas como cross-temporales, combinadas con un mecanismo de **market-guided gating** que selecciona features dinámicamente según las condiciones del mercado. Alterna entre atención intra-stock (temporal) e inter-stock (cross-seccional). En CSI 300, supera a HIST, GAT, GRU y XGBoost. Su repositorio open-source está integrado con Qlib.

**StockMixer** (Fan & Shen, AAAI 2024) aporta un hallazgo provocador: una arquitectura **puramente MLP** con tres módulos de mixing (indicadores → temporal → cross-stock) iguala o supera a transformers complejos en NASDAQ, NYSE y S&P 500. Esto cuestiona directamente si la complejidad de Stockformer se justifica: si un MLP simple funciona igual, el modelo complejo probablemente sobreajusta.

**MATCC** (CIKM 2024) extiende el enfoque de MASTER extrayendo explícitamente tendencias de mercado como guía y minando correlaciones cross-temporales. **MERA** (WWW 2025) introduce Mixture-of-Experts con retrieval-augmented representations, abordando la heterogeneidad sectorial del S&P 500 mediante expertos especializados por tipo de patrón.

El benchmark más directamente relevante es el de **Kwiatkowski & Chudziak (2026, Springer LNCS)**, que compara Vanilla Transformer, CrossFormer, MASTER adaptado e iTransformer para ranking diario y selección de portfolio en el S&P 500. Este paper — junto con su estudio sobre funciones de pérdida (arXiv:2510.14156) — es lectura obligatoria para el proyecto.

### State Space Models: Mamba como alternativa al transformer

Los SSMs, liderados por Mamba, emergen como alternativa seria a los transformers para datos financieros. **FinMamba** (arXiv:2502.06707, 2025) argumenta que el mecanismo de atención del transformer **sobrepondera outliers y anomalías**, mientras que el mecanismo selectivo de Mamba captura tendencias suaves y patrones recurrentes de forma más robusta — un argumento especialmente relevante para el S&P 500, donde la señal es débil y el ruido domina.

**SAMBA** (Mehrabian et al., ICASSP 2025) combina Mamba bidireccional con convolución de grafos adaptativa, logrando resultados superiores en **DJIA, NASDAQ y NYSE** con complejidad lineal O(n) vs. O(n²) de la atención. **MaGNet** (arXiv:2511.00085, 2025) extiende esto con hipergrafos duales para capturar relaciones de grupo de orden superior. **AGSMNet** (2025) combina Gaussian Short-Time Fourier Transform con Mamba, ofreciendo una alternativa directa al enfoque wavelet + transformer de Stockformer.

La recomendación es clara: considerar **reemplazar el encoder temporal basado en transformer con bloques Mamba**, manteniendo o mejorando el componente de grafos. La complejidad lineal de Mamba permite además procesar secuencias más largas.

### Foundation models: promesa limitada, excepciones notables

El paper más importante del campo es **"Re(Visiting) Time Series Foundation Models in Finance"** (Rahimikia et al., arXiv:2511.18578, 2025). Sus conclusiones son contundentes: los TSFMs genéricos (TimesFM, Chronos) **funcionan peor que modelos lineales** en zero-shot sobre datos financieros. Sin embargo, modelos **pre-entrenados desde cero sobre datos financieros** generan ganancias sustanciales tanto estadísticas como económicas. Además, la combinación de TSFMs con factores financieros y **data augmentation sintética** mejora consistentemente los resultados.

Dos foundation models financieros específicos merecen atención. **Kronos** (Shi et al., 2025) es un decoder-only transformer pre-entrenado sobre **12 mil millones de registros K-line** de 45 bolsas globales, tratando los candlesticks como un "lenguaje" con gramática propia. **FinCast** (Zhu et al., CIKM 2025) usa Mixture-of-Experts con **20 mil millones de puntos temporales** de crypto, forex, futuros y acciones, superando a TimesFM y Chronos en zero-shot financiero. Ambos son candidatos para fine-tuning o como extractores de features.

---

## Mejoras arquitectónicas concretas para el Stockformer

### Descomposición: wavelet vs. alternativas

La DWT con Sym2 actual tiene una limitación poco discutida: **puede filtrar señal junto con el ruido** si los parámetros no están calibrados para la estructura espectral del mercado estadounidense. Un estudio de evaluación en S&P 500 (arXiv:2408.12408, 2024) encontró que **DWT con wavelet db4** (no Sym2) mejoró significativamente la predicción de dirección cuando se combinó con xLSTM-TS.

Las alternativas principales son tres. **VMD** (Variational Mode Decomposition) optimizado con algoritmos genéticos redujo el MAPE un **79.84%** vs. LSTM standalone para commodities (Scientific Reports, 2025). **CEEMDAN** consistentemente supera a EMD y EEMD, pero tiene un **riesgo crítico de information leakage**: si se aplica a todo el dataset antes del split train/test, la información futura contamina el entrenamiento. Un paper de Nature Scientific Reports (2024) documenta este problema y propone **Sliding Window Decomposition** como solución. **STL** (Seasonal-Trend-Loess) es inherentemente segura contra leakage y el modelo STGAT (2025) la integra exitosamente con graph attention para predicción de acciones en mercados chinos y estadounidenses.

La recomendación es reemplazar DWT(Sym2) con **VMD en modo sliding-window** o **STL**, verificando estrictamente que no haya leakage temporal en la descomposición.

### Grafos: de Struc2Vec estático a grafos dinámicos

El enfoque actual de Struc2Vec para construir el grafo tiene dos limitaciones: es **estático** (las relaciones entre acciones no cambian durante el entrenamiento) y captura solo similitud estructural, no relaciones económicas reales. La literatura reciente propone mejoras sustanciales:

- **Grafos dinámicos aprendidos** (MASTER, HSGNN): Construir el grafo end-to-end como parte del entrenamiento, permitiendo que las relaciones evolucionen con el mercado
- **Grafos multi-relacionales** (HSGNN, 2025): Combinar grafos de flujo de dinero, cadena de suministro e industria en una única representación heterogénea
- **Hipergrafos** (MDHAN KDD 2024, CI-STHPAN AAAI 2024): Capturan relaciones de grupo de orden superior — por ejemplo, que las FAANG se mueven juntas no como pares sino como grupo. MDHAN mostró mejoras significativas en datasets de EEUU y China
- **Grafos causales implícitos** (2024, MDPI Information): Minan relaciones causales (no solo correlacionales) entre acciones, distinguiendo influencia directa de co-movimiento espurio

Para el S&P 500 específicamente, la combinación de **grafos de industria GICS + supply-chain + correlación dinámica aprendida** reemplazaría eficazmente a Struc2Vec.

### Funciones de pérdida: el cambio de mayor impacto individual

Con IC ≈ −0.003, el modelo no está aprendiendo ranking cross-seccional. Esto apunta directamente a un problema con la función de pérdida. El estudio sistemático de Kwiatkowski & Chudziak (arXiv:2510.14156, 2025), evaluando múltiples funciones de pérdida para **ranking de acciones en el S&P 500** con una arquitectura PortfolioMASTER, concluye que las **loss functions orientadas a ranking** (pairwise y listwise) superan claramente a MSE para selección de portfolio.

Las opciones más prometedoras son:

- **ListNet** (listwise): optimiza la distribución completa de rankings, correlacionando directamente con IC
- **RankNet** (pairwise): aprende ordenamiento relativo entre pares de acciones
- **IC-loss diferenciable**: optimizar directamente L = 1 − IC(predicciones, retornos), donde IC se calcula como correlación de Spearman diferenciable
- **Return-weighted loss** (arXiv:2502.17493, 2025): pondera la pérdida por el retorno real, enfocando el aprendizaje en detectar oportunidades de alto crecimiento — alcanzó **61.73% de retorno anual con Sharpe 1.18** sobre 2019–2024
- **Heteroscedastic Gaussian loss** (Yang et al., IPM 2024): modela retornos como distribuciones heteroscedásticas, prediciendo simultáneamente media y desviación estándar — mejora absoluta de 20–50% en retorno de portfolios Top 20

La configuración recomendada es una **loss combinada**: L_total = α·ListNet + β·(1−IC) + γ·MSE, donde los pesos se calibran por validación.

---

## Entrenamiento: cómo evitar sobreajuste en señales débiles

### Rolling window y concept drift

El uso de splits estáticos train/val/test es inadecuado para datos financieros que exhiben concept drift. La configuración recomendada es **walk-forward rolling window**: entrenar sobre una ventana de 2–3 años, validar sobre 3–6 meses, testear sobre 1–3 meses, y avanzar iterativamente. Reentrenar al menos mensualmente.

**DoubleAdapt** (Zhao et al., KDD 2023) es el estado del arte para adaptación incremental en predicción de acciones: un enfoque de meta-learning basado en MAML que adapta conjuntamente datos y modelo ante cambios de distribución. Funciona sobre Qlib y está validado en CSI 100/300/500. Su insight clave: "las tendencias futuras dependen principalmente de tendencias recientes; es más beneficioso aprender patrones nuevos de datos recientes que memorizar patrones históricos de largo plazo."

### Regularización agresiva contra sobreajuste

Dado el ratio señal/ruido extremadamente bajo del S&P 500 (~0.01–0.05 de IC alcanzable), se necesita regularización mucho más agresiva que la estándar:

- **Inyección de ruido gaussiano** (σ = 0.01–0.03) sobre features Alpha360 durante entrenamiento — actúa como regularización ridge implícita (demostrado teóricamente en ICML 2021)
- **RLSTM** (PMC 8446482): un módulo de prevención que procesa series aleatorias en paralelo al input real, impidiendo que el modelo memorice patrones espurios. Probado exitosamente en **S&P 500**
- **Dropout elevado** (0.2–0.3) en atención y capas ocultas, con gradient clipping
- **Reducción de complejidad del modelo**: el Stockformer completo (transformer + wavelets + grafos + multitask) puede estar sobreparametrizado para la señal disponible. Considerar reducir cabezas de atención, capas, o dimensiones de embedding
- **Feature subsampling aleatorio** por batch — similar a la selección de features de random forests

### Data augmentation y self-supervised learning

La augmentación de datos financieros requiere preservar la estructura temporal, correlaciones cross-seccionales y colas pesadas. Las técnicas más prometedoras son:

**GANs financieros** (JRFM 2024) generan datos sintéticos que replican distribuciones de retornos reales, mejorando significativamente la precisión de forecasting. **FTS-Diffusion** (2024) usa modelos de difusión para capturar patrones temporales invariantes de escala. Para la clasificación up/down, **SMOTE** combinado con dropout mejoró la accuracy a 94.9% en predicción de tendencias (Heliyon 2024). Las técnicas simples — jittering, window slicing, magnitude warping — son frecuentemente más prácticas que los métodos generativos.

El pre-entrenamiento self-supervised emerge como una técnica particularmente poderosa. **CI-STHPAN** (AAAI 2024) demostró que "incluso sin información de grafos, el self-supervised learning basado en Transformer Encoder supera los resultados SOTA" — evidencia fuerte de que las representaciones pre-entrenadas importan más que las arquitecturas complejas. **Contrastive learning** (SGLNet 2024, DGRCL ICAART 2025) construye representaciones robustas sin requerir labels, reduciendo la dependencia de datos supervisados ruidosos.

---

## Estrategia de trading: de long-only a market-neutral

### Construcción del portfolio long-short

El beta de 1.33 y alpha de −40.96% revelan que el portfolio actual es esencialmente un proxy apalancado del mercado que selecciona acciones peores que el promedio. La migración a long-short es la corrección más urgente.

El enfoque estándar (Gu, Kelly & Xiu, 2020, *Review of Financial Studies*) es: rankear todas las acciones por retorno predicho → ir largo el quintil/decil superior → ir corto el quintil/decil inferior → equal-weight dentro de cada lado. Gu et al. demostraron que modelos ML con 94 características generan spreads long-short de ~0.7% mensual. Para mayor granularidad, ponderar proporcionalmente al z-score de la predicción (score-weighted approach).

El paper de **"Deep Learning in Long-Short Stock Portfolio Allocation"** (arXiv:2411.13555, 2024) aplica directamente LSTM y Transformer a S&P 500 para estrategias long-short, integrando Mean-Variance Optimization para determinar pesos y mejorando Sharpe y max drawdown vs. equal-weight.

Para neutralizar el beta, tres opciones escalonadas: (1) **Dollar-neutral**: igualar exposición larga y corta en dólares; (2) **Beta-neutral**: ajustar pesos para que Σw_i·β_i ≈ 0; (3) **Factor-neutral**: regresar la señal contra Market, Size, Value, Momentum, Profitability e Investment (Fama-French 6), y usar solo los residuos como señal alpha.

### Portfolio optimization end-to-end

Más allá de equal-weight, la optimización diferenciable de portfolios ha avanzado significativamente. **BPQP** (NeurIPS 2024) integra una capa de optimización convexa diferenciable en la red neuronal, mejorando el Sharpe de **0.65 a 1.28** vs. el enfoque predict-then-optimize en dos etapas. La implementación usa cvxpylayers dentro de PyTorch, permitiendo backpropagation a través de la optimización. **Deep Declarative Risk Budgeting** (arXiv:2504.19980, 2025) extiende este enfoque con mayor estabilidad y reproducibilidad.

La ruta de implementación recomendada es incremental: primero replace equal-weight por **Hierarchical Risk Parity (HRP)** como baseline robusta, luego Mean-Variance con covariance shrinkage, y finalmente optimización end-to-end diferenciable.

### Ensemble y gestión de riesgo

Los ensembles heterogéneos dominan consistentemente a modelos individuales. La configuración óptima combina **al menos un modelo tree-based** (LightGBM/XGBoost, que captura umbrales no lineales), **un modelo recurrente/SSM** (LSTM o Mamba), y **un modelo basado en atención** (Stockformer o MASTER), con un meta-learner simple (Ridge o XGBoost) sobre predicciones out-of-fold.

**RegimeFolio** (arXiv:2510.14986, 2024) proporciona un framework completo: clasificador de régimen basado en VIX (3 estados), ensembles sectoriales entrenados por régimen, y optimización MV dinámica con shrinkage. Sobre 34 acciones large-cap estadounidenses (2020–2024), alcanzó **retorno anualizado del 25.1% con Sharpe 1.17**. La integración de detección de régimen (via VIX o Hidden Markov Models) permite reducir exposición en períodos de alta volatilidad y ajustar dinámicamente la agresividad del modelo.

---

## Repositorios y recursos para implementación inmediata

El ecosistema open-source actual ofrece implementaciones listas para experimentar. **Microsoft Qlib** (15k+ stars) es la plataforma de referencia, con soporte para Alpha158/Alpha360, modelos desde LightGBM hasta MASTER, y ahora integración con **R&D-Agent-Quant** (arXiv:2505.15155, 2025) que usa LLMs (GPT-4o, o3-mini) para generar y optimizar factores automáticamente — evaluado exitosamente en **NASDAQ 100**. MASTER y StockMixer (ambos SJTU-DMTai) tienen repositorios activos integrados con Qlib. **SAMBA** (GitHub: Ali-Meh619/SAMBA) ofrece Graph-Mamba validado en datos de DJIA, NASDAQ y NYSE. **FinRL** (14k+ stars) proporciona agentes RL (PPO, SAC, TD3) para trading, con la reciente integración FinRL-DeepSeek para combinar LLMs con RL. El repositorio **stock-top-papers** (GitHub: marcuswang6/stock-top-papers) mantiene un catálogo curado de papers de predicción de acciones de venues top (KDD, WWW, AAAI, IJCAI, NeurIPS) con enlaces a código.

---

## Hoja de ruta priorizada de intervenciones

**Fase 1 — Correcciones críticas (impacto alto, esfuerzo bajo)**. Primero, verificar que no haya bugs de alineación temporal o data leakage — un IC negativo puede indicar labels desfasadas. Segundo, reemplazar MSE por una loss de ranking (ListNet + IC auxiliar). Tercero, construir un baseline con LightGBM sobre Alpha158 para establecer el IC alcanzable con las features correctas. Cuarto, migrar a long-short con neutralización dollar-neutral y beta-hedge con futuros SPY.

**Fase 2 — Mejoras de features y entrenamiento (impacto alto, esfuerzo moderado)**. Ampliar features: incorporar Alpha158, indicadores macro (VIX, yield curve), sentiment vía FinBERT, y señales cross-market. Implementar walk-forward rolling window con reentrenamiento mensual. Aplicar regularización agresiva: dropout 0.2–0.3, noise injection, gradient clipping, y potencialmente reducir la dimensionalidad del modelo. Reemplazar DWT(Sym2) por VMD sliding-window o STL.

**Fase 3 — Arquitectura y estrategia avanzada (esfuerzo alto, retorno potencialmente alto)**. Evaluar MASTER y StockMixer como arquitecturas alternativas sobre S&P 500 datos. Reemplazar Struc2Vec por grafos dinámicos multi-relacionales (industria GICS + supply-chain + correlación aprendida). Considerar sustituir el encoder temporal por bloques Mamba (FinMamba). Implementar ensemble heterogéneo (tree + SSM + attention). Integrar optimización de portfolio end-to-end (cvxpylayers/BPQP). Aplicar detección de régimen con VIX para dynamic position sizing.

## Conclusión

La investigación 2024–2026 converge en un mensaje claro: para mercados eficientes como el S&P 500, **la calidad de las features y la función de pérdida importan más que la sofisticación arquitectónica**. StockMixer demostró que un MLP simple puede igualar a transformers complejos; Rahimikia et al. confirmaron que LightGBM sigue siendo extraordinariamente competitivo; y el estudio de funciones de pérdida sobre S&P 500 mostró que optimizar ranking directamente transforma los resultados. El IC negativo actual probablemente refleja una combinación de features insuficientes para el mercado estadounidense, una loss function inadecuada, y posiblemente un bug de alineación temporal — no una deficiencia fundamental del concepto Stockformer. La arquitectura wavelet + grafos + multitask tiene mérito teórico, pero debe adaptarse con features enriquecidas (fundamentales, macro, sentiment), grafos dinámicos, loss de ranking, entrenamiento rolling, y una estrategia market-neutral para tener oportunidad de generar alpha en el S&P 500. Los ICs realistas alcanzables para acciones large-cap estadounidenses se sitúan en el rango 0.01–0.05 — modestos pero económicamente significativos si se implementan con una estrategia de trading disciplinada.