# Entrenar el estudio de ablación (E1–E7) en la DGX

> Objetivo: rellenar `results/ablation_results.csv` con el IC real de cada experimento
> E1–E5 (Stockformer) — E6 (LightGBM) ya está medido — para que la tabla de la memoria
> (`tab:ablation`) y la Sección 5.3 dejen de tener solo una fila.

## 0. Por qué la DGX y no el portátil

El smoke-test local confirmó que el pipeline es correcto (carga de datos ✓, construcción
del modelo ✓, bucle de entrenamiento ✓), pero la atención espacial de Stockformer es
O(N²) sobre las ~477 acciones y **agota la RAM de un equipo normal** (OOM en 24 GB). En la
DGX (H200, 143 GB VRAM) entra sin problema. No hay ningún bug que arreglar: solo falta GPU.

## 1. Qué se entrena

| Exp | Loss | Features | Grafo | Config | Estado |
|-----|------|----------|-------|--------|--------|
| E1 | MSE | Alpha360 | estático | `config/experiment_E1_mse_baseline.conf` | a entrenar |
| E2 | Ranking | Alpha360 | estático | `config/experiment_E2_ranking_baseline.conf` | a entrenar |
| E3 | Ranking | Alpha360 | dinámico (`graph_type=learned`) | `config/experiment_E3_dynamic_graph.conf` | a entrenar |
| E4 | Ranking | todas (`max_features=-1`) | estático | `config/experiment_E4_rich_features.conf` | a entrenar |
| E5 | Ranking | todas | dinámico | `config/experiment_E5_all_improvements.conf` | a entrenar |
| E6 | LightGBM | Alpha158 | — | `config/Multitask_Stock_SP500.conf` | **ya medido (IC +0.0136)** |

Todas las configs apuntan al **mismo** directorio de datos ya existente
(`data/Stock_SP500_2018-01-01_2026-03-16/`: `flow.npz`, `features/`, `corr_adj.npy`,
`128_corr_struc2vec_adjgat.npy`, `trend_indicator.npz`). **No hay que construir datos nuevos.**

> E7 (walk-forward del mejor config) va aparte, con `scripts/run_walkforward.py` usando la
> config E5; añádelo después si quieres la fila E7 en la tabla.

## 2. Preparar el entorno en la DGX

Sigue la guía del servidor (VPN Comillas → VS Code Remote-SSH → contenedor Docker). Resumen:

```bash
# En tu /workspace de la DGX
git clone <tu-repo>.git tfm-stockformer
cd tfm-stockformer

# Sube el directorio de datos (≈250 MB) a data/ (scp / rsync / VS Code).
# Debe quedar: data/Stock_SP500_2018-01-01_2026-03-16/{flow.npz,features/,*.npy,...}

# Levanta el contenedor (Dockerfile/requirements.txt del repo) y entra en él:
docker compose up -d --build
docker exec -it <tu-contenedor> bash
pip install -r requirements.txt      # si no está ya en la imagen
```

Comprueba GPU dentro del contenedor: `nvidia-smi` debe listar una H200.

## 3. Lanzar el entrenamiento

```bash
# Elige una GPU poco cargada (mira nvidia-smi) y lanza el estudio completo:
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ablations_dgx.sh

# Solo un experimento (para probar primero):
CUDA_VISIBLE_DEVICES=0 bash scripts/run_ablations_dgx.sh --only E1
```

El script:
1. Ejecuta `scripts/run_ablation.py` → entrena E1–E5 (50 épocas c/u) y recoge E6, escribiendo
   `results/ablation_results.csv`.
2. Regenera `MEMORIA/tfm/tablas/ablation.tex` con `scripts/build_latex_tables.py`.

**Sugerencia:** prueba primero `--only E1`. Si produce una línea de IC en el log
(`log/STOCK/ablation_E1_mse_baseline`) y una fila en el CSV, lanza el resto.

## 4. Cerrar el círculo en la memoria

```bash
# 1) Trae de vuelta el CSV a tu máquina local:
#    scp dgx:.../tfm-stockformer/results/ablation_results.csv results/

# 2) Regenera la tabla y recompila:
python3 scripts/build_latex_tables.py
cd MEMORIA/tfm && ./build.sh        # tab:ablation ya muestra E1-E6

# 3) Actualiza la prosa de la Sección 5.3 (05_parte1_transferencia.tex):
#    - sustituye la mención "su checkpoint arroja IC ~-0.005" por el IC real de E2,
#    - cita los IC reales de E1, E3, E4, E5 (y E7 si lo entrenas),
#    - elimina el bloque "% NOTA ..." al final de la sección.
```

Avísame cuando tengas el `ablation_results.csv` completo y actualizo yo la prosa de la 5.3
con los números reales y, si quieres, añado la tabla de resultados también en el Capítulo 5
(ahora solo está en el Apéndice B).

## 5. Resolución de problemas

- **OOM en la DGX:** baja `batch_size` en la config (de 12 a 8/6) o usa una GPU más libre.
- **`nvidia-smi` no aparece:** revisa el `docker-compose.yaml` (`count: 1`, `capabilities: [gpu]`).
- **No genera IC en el log:** confirma que `[loss] type` y el resto de la config son válidos
  y que el directorio de datos está completo.
