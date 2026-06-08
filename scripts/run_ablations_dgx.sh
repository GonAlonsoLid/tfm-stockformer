#!/usr/bin/env bash
# ============================================================================
# Entrenamiento del estudio de ablación (E1-E5 Stockformer + E6 LightGBM)
# en la DGX, y regeneración de la tabla LaTeX de la memoria.
#
# Requisitos: GPU NVIDIA visible (en la DGX), datos en
#   data/Stock_SP500_2018-01-01_2026-03-16/  (flow.npz, features/, grafo, ...)
#
# Uso (desde la raíz del repo, dentro del contenedor de la DGX):
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_ablations_dgx.sh
#   # o un subconjunto:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/run_ablations_dgx.sh --only E1
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=================================================================="
echo " GPU visible: ${CUDA_VISIBLE_DEVICES:-(no fijada — usará la 0)}"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv 2>/dev/null || \
  echo "  (nvidia-smi no disponible — ¿estás en la DGX y dentro del contenedor?)"
echo "=================================================================="

# 1) Entrena E1-E5 (Stockformer) y recoge E6 (LightGBM) -> results/ablation_results.csv
#    Cada experimento entrena 50 épocas; en una H200 son del orden de minutos-hora cada uno.
python3 scripts/run_ablation.py "$@"

# 2) Regenera la tabla LaTeX de ablación a partir del CSV actualizado.
python3 scripts/build_latex_tables.py

echo ""
echo "=================================================================="
echo " HECHO."
echo " - Resultados:  results/ablation_results.csv"
echo " - Tabla LaTeX: MEMORIA/tfm/tablas/ablation.tex (tab:ablation)"
echo ""
echo " Siguientes pasos para la memoria:"
echo "   1) Copia results/ablation_results.csv de vuelta a tu máquina local."
echo "   2) python3 scripts/build_latex_tables.py   (si lo regeneras en local)"
echo "   3) cd MEMORIA/tfm && ./build.sh"
echo "   4) Actualiza la prosa de la Seccion 5.3 con los IC reales de E1-E5/E7"
echo "      (y elimina la NOTA y la mencion al checkpoint de E2)."
echo "=================================================================="
