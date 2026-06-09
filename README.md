# TFM — Transferencia cross-market de Stockformer al S&P 500

Trabajo Fin de Máster. Estudia, en dos partes, si un modelo de *deep learning* financiero
calibrado en un mercado transfiere a otro estructuralmente distinto, y, cuando no lo hace,
qué genera realmente el retorno neto.

- **Parte I (RQ1):** ¿transfiere **Stockformer** (arquitectura *wavelet-transformer-grafo*,
  ~10⁶ parámetros, calibrada sobre acciones chinas A-shares) al S&P 500 a horizonte semanal?
  Mediante una *complexity ladder* y una auditoría anti-fuga limpia, la respuesta es **no**:
  su IC *out-of-sample* cae a ≈ −0,003 y lo bate un Lasso de cinco coeficientes (+0,0238).
- **Parte II (RQ2):** ¿qué genera el retorno neto? Una atribución etapa a etapa señala la
  **construcción de cartera cost-aware**, y una búsqueda disciplinada de señal encuentra un
  factor concreto con contenido: el **momentum residual**. El pipeline `momentum residual +
  ensemble` con construcción cost-aware da un Sharpe neto robusto de 0,72 (t = 1,82) sobre
  un *walk-forward* de 311 semanas — mejora apreciable, al borde de la significancia.

La memoria completa (LaTeX) está en `MEMORIA/tfm/`.

## Estructura del repositorio

```
MEMORIA/tfm/            Memoria del TFM (LaTeX). Compilar con build.sh
├── Capitulos/          8 capítulos
├── Apendices/          Anexo B (tablas completas)
├── tablas/             Fragmentos LaTeX de tablas (generados desde results/*.csv)
├── figuras/            Figuras de la memoria (PNG)
├── main.tex, preambulo.tex, main.bib
└── build.sh            latexmk a convergencia (backend bibtex)

lib/                    Módulos compartidos del pipeline de retornos
                        (data_panel, weekly_panel, portfolio, neutralize, ...)

scripts/                Pipeline de reproducción (ver "Reproducir resultados")
└── sp500_pipeline/     Descarga OHLCV y embeddings de grafo (Struc2Vec)

Stockformermodel/       Implementación del modelo Stockformer (Parte I)
MultiTask_Stockformer_train.py   Entrenamiento de Stockformer (GPU)
config/                 Configs de entrenamiento de Stockformer

results/                Cifras canónicas (CSV) que alimentan las tablas de la memoria
cpt/                    Checkpoints entrenados de Stockformer
output/                 Salida de inferencia de Stockformer
tests/                  Tests del pipeline (pytest)

data/                   Panel S&P 500 (~7 GB, externo — no versionado, ver .gitignore)
_local/                 Material local fuera del repo (demo, planning, refs, infra DGX)
```

> `data/` y `_local/` no se versionan (`.gitignore`). El panel de datos se reconstruye con
> el pipeline; `_local/` guarda material auxiliar (app Streamlit, notebooks, PDFs de
> referencia, bundle de la DGX) que no forma parte de la memoria.

## Cómputo

El entrenamiento y la evaluación de **Stockformer requieren GPU** y se ejecutaron en la
estación NVIDIA DGX de la Universidad. El resto del trabajo (la *complexity ladder* de
modelos simples, la estrategia semanal y los análisis de robustez) corre **en CPU**.

## Compilar la memoria

```sh
cd MEMORIA/tfm && ./build.sh        # -> main.pdf
```

Si el árbol tiene `.aux` viejos, `latexmk -C` antes de `build.sh` evita un falso aviso de
convergencia.

## Reproducir resultados

```sh
pip install -r requirements.txt
```

**1. Datos** (descarga OHLCV, features Alpha360/158, fundamentales, embeddings de grafo):

```sh
python scripts/build_pipeline.py --config config/Multitask_Stock_SP500.conf
```

**2. Stockformer** (GPU) — entrenamiento e inferencia:

```sh
python MultiTask_Stockformer_train.py --config config/Multitask_Stock_SP500.conf
python scripts/run_inference.py        --config config/Multitask_Stock_SP500.conf
```

**3. Parte I — complexity ladder y significancia** (CPU):

```sh
python scripts/run_cpu_ladder.py
python scripts/run_ladder_analysis.py
python scripts/run_power_signtest.py
```

**4. Parte II — estrategia semanal, robustez y construcción** (CPU):

```sh
python scripts/run_weekly_strategy.py
python scripts/run_weekly_robustness.py --init_train 200 --step 26
python scripts/run_construction.py
python scripts/run_pars_us.py
```

**5. RQ2 — búsqueda de señal y pipeline ganador** (CPU):

```sh
python scripts/run_signal_search.py     # ronda 1 (reversal, momentum, ...)
python scripts/run_signal_search2.py    # ronda 2 (familia momentum)
python scripts/run_signal_search3.py    # ronda 3 (ensemble + momentum residual, ventana justa)
python scripts/run_signal_search4.py    # ronda 4 (multi-factor con fundamentales)
python scripts/pipeline_rq2.py          # pipeline concreto: momentum residual + cost-aware
python scripts/build_rq2_table.py       # tabla LaTeX de la búsqueda (tab:rq2-signal-search)
```

**6. Tablas LaTeX** a partir de los CSV:

```sh
python scripts/build_latex_tables.py
```

## Cita (modelo original)

```
Ma, B., Xue, Y., Lu, Y., & Chen, J. (2025). Stockformer: A price-volume factor stock
selection model based on wavelet transform and multi-task self-attention networks.
Expert Systems with Applications, 273, 126803. https://doi.org/10.1016/j.eswa.2025.126803
```
