# Stockformer S&P500 — TFM Adaptation

Adaptation of the **Stockformer** model (originally trained on Chinese A-share markets) to the **S&P 500** universe, developed as a Master's thesis (TFM). The goal is to evaluate whether the architecture — wavelet-based feature decomposition, multi-task self-attention, and Alpha360-style price/volume factors — generalises to US equity markets.

## What this project does

The original Stockformer paper introduced a stock ranking model combining:
- Wavelet transform for multi-scale temporal feature extraction
- Multi-task self-attention (price prediction + trend classification)
- 360 price/volume ratio features (Alpha360) as model inputs

This repository adapts that pipeline end-to-end for the S&P 500:

1. **Data collection** — downloads daily OHLCV for S&P 500 constituents via yfinance (2018–2024)
2. **Feature engineering** — reconstructs 360 Alpha360-style price/volume ratio features from raw OHLCV
3. **Graph embedding** — builds Struc2Vec structural embeddings from a return-correlation graph
4. **Model training** — trains the Stockformer architecture on the S&P 500 dataset
5. **Evaluation** — computes IC/ICIR metrics and runs a daily-rebalanced portfolio backtest vs SPY

## Based on

Ma, Bohan; Xue, Yushan; Lu, Yuan & Chen, Jing. (2025). "Stockformer: A price-volume factor stock selection model based on wavelet transform and multi-task self-attention networks". *Expert Systems with Applications*, 273, 126803. DOI: [https://doi.org/10.1016/j.eswa.2025.126803](https://doi.org/10.1016/j.eswa.2025.126803)

## File Structure

```
scripts/
├── build_pipeline.py        # Data pipeline orchestrator (steps 1–5)
├── build_alpha360.py        # Alpha360 feature builder (invoked by build_pipeline.py)
├── sp500_pipeline/          # Step scripts: download, normalize, serialize, graph embedding
├── run_inference.py         # Standalone inference on saved checkpoint
├── compute_ic.py            # IC / ICIR evaluation
└── run_backtest.py          # Portfolio backtest vs SPY

config/
└── Multitask_Stock_SP500.conf   # All paths and hyperparameters for the S&P500 run

data/
└── Stock_SP500_2018-01-01_2024-01-01/   # Created by build_pipeline.py
    ├── ohlcv/               # Parquet files (one per ticker)
    ├── features/            # 360 Alpha360-style price/volume ratio CSVs
    ├── flow.npz             # Normalized return sequences
    ├── trend_indicator.npz  # Trend label arrays
    └── 128_corr_struc2vec_adjgat.npy  # Graph embeddings

app.py                       # Streamlit interface
MultiTask_Stockformer_train.py  # Model training entry point
```

## Quick Start

Reproduce the full pipeline from a fresh clone in seven steps.

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Build data pipeline (download → features → embeddings)

```sh
python scripts/build_pipeline.py --config config/Multitask_Stock_SP500.conf
```

This single command runs all five steps:
- Downloads S&P 500 OHLCV data via yfinance
- Normalizes and splits the dataset
- Serializes `flow.npz` and `trend_indicator.npz`
- Builds Struc2Vec graph embeddings
- Generates 360 Alpha360-style price/volume ratio features

On re-runs, completed steps are skipped automatically (sentinel-based idempotency).

### 3. Train the model

```sh
python MultiTask_Stockformer_train.py --config config/Multitask_Stock_SP500.conf
```

> **GPU required for reasonable runtime.** The training script runs on CPU but will be extremely slow. [Kaggle](https://www.kaggle.com/) provides free GPU notebooks suitable for this workload. For a quick smoke test (2 epochs), add `--max_epoch 2`.

### 4. Run inference

```sh
python scripts/run_inference.py --config config/Multitask_Stock_SP500.conf
```

Generates prediction CSVs in `output/` using the saved checkpoint.

### 5. Compute IC / ICIR evaluation metrics

```sh
python scripts/compute_ic.py
```

Prints Spearman IC, ICIR, and Pearson IC for the test period.

### 6. Run portfolio backtest

```sh
python scripts/run_backtest.py
```

Builds a daily-rebalanced top-K portfolio, computes returns vs SPY, and saves `backtest_results.csv` and `backtest_positions.csv`.

### 7. Launch the Streamlit interface

```sh
streamlit run app.py
```

Opens an interactive browser interface for exploring predictions and backtest results.

---

## Citation

If you build on this work or the original Stockformer model, please cite the original paper:

```
Ma, B., Xue, Y., Lu, Y., & Chen, J. (2025).
Stockformer: A price-volume factor stock selection model based on
wavelet transform and multi-task self-attention networks.
Expert Systems with Applications, 273, 126803.
https://doi.org/10.1016/j.eswa.2025.126803
```
