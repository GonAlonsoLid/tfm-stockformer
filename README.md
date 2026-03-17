# Stockformer Data File Explanation and Execution Methods

[中文版本](README中文版.md)

## "Stockformer" Code Overview
This paper, titled "Stockformer: A Price-Volume Factor Stock Selection Model Based on Wavelet Transform and Multi-Task Self-Attention Networks," is currently under review at Expert Systems with Applications. You can read the preprint version of the paper on SSRN: [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4648073](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4648073).

This work has now been formally published in **Expert Systems with Applications**.
You can read the full article here:
[https://www.sciencedirect.com/science/article/pii/S0957417425004257?dgcid=coauthor](https://www.sciencedirect.com/science/article/pii/S0957417425004257?dgcid=coauthor).

## Original Dataset and Sub-Datasets After Partition
Due to the large size of both the original data (which contains 360 price and volume factors) and the processed data (which also contains 360 factors), the author has stored them on a cloud drive for readers to use. The folder `Stock_CN_2018-03-01_2024-03-01` contains the original data, while other folders hold the processed sub-datasets.

- **Original data and Sub-Datasets** (raw, compressed): [Baidu Netdisk](https://pan.baidu.com/s/1dnmzt9F2Ug9bCQDZwZ2e4Q?pwd=ykqp)
- **Original data and Sub-Datasets** (decompressed, ready to view): [Google Drive](https://drive.google.com/drive/folders/1ZJpjHiIIkjfbtPIcAmi2nfLNv6VC5ym_?usp=drive_link)

There are a total of 14 sub-datasets after processing, providing data support for backtesting over different time periods. The detailed contents of these datasets are as follows:

|             | Training Set | Training Set | Validation Set | Validation Set | Test Set    | Test Set    |
| ----------- | ------------ | ------------ | -------------- | -------------- | ----------- | ----------- |
| Dataset     | Start Date   | End Date     | Start Date     | End Date       | Start Date  | End Date    |
| Subset 1    | 2018-03-01   | 2020-02-28   | 2020-03-02     | 2020-06-30     | 2020-07-01  | 2020-10-29  |
| Subset 2    | 2018-05-31   | 2020-05-29   | 2020-06-01     | 2020-09-23     | 2020-09-24  | 2021-01-25  |
| Subset 3    | 2018-08-27   | 2020-08-26   | 2020-08-27     | 2020-12-25     | 2020-12-28  | 2021-04-28  |
| Subset 4    | 2018-11-28   | 2020-11-27   | 2020-11-30     | 2021-03-30     | 2021-03-31  | 2021-07-28  |
| Subset 5    | 2019-03-04   | 2021-03-02   | 2021-03-03     | 2021-06-30     | 2021-07-01  | 2021-11-01  |
| Subset 6    | 2019-06-03   | 2021-06-01   | 2021-06-02     | 2021-09-27     | 2021-09-28  | 2022-01-26  |
| Subset 7    | 2019-08-28   | 2021-08-26   | 2021-08-27     | 2021-12-28     | 2021-12-29  | 2022-05-05  |
| Subset 8    | 2019-11-29   | 2021-11-30   | 2021-12-01     | 2022-03-31     | 2022-04-01  | 2022-08-01  |
| Subset 9    | 2020-03-04   | 2022-03-03   | 2022-03-04     | 2022-07-04     | 2022-07-05  | 2022-11-02  |
| Subset 10   | 2020-06-03   | 2022-06-06   | 2022-06-07     | 2022-09-28     | 2022-09-29  | 2023-02-03  |
| Subset 11   | 2020-08-31   | 2022-08-30   | 2022-08-31     | 2022-12-29     | 2022-12-30  | 2023-05-05  |
| Subset 12   | 2020-12-02   | 2022-12-01   | 2022-12-02     | 2023-04-03     | 2023-04-04  | 2023-08-02  |
| Subset 13   | 2021-03-05   | 2023-03-06   | 2023-03-07     | 2023-07-05     | 2023-07-06  | 2023-11-03  |
| Subset 14   | 2021-06-04   | 2023-06-05   | 2023-06-06     | 2023-09-28     | 2023-10-09  | 2024-01-30  |

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

If you use this model or the dataset in your research, please cite our paper as follows:

Ma, Bohan; Xue, Yushan; Lu, Yuan & Chen, Jing. (2025). "Stockformer: A price-volume factor stock selection model based on wavelet transform and multi-task self-attention networks". Expert Systems with Applications, 273, 126803. DOI: [https://doi.org/10.1016/j.eswa.2025.126803](https://doi.org/10.1016/j.eswa.2025.126803)

This citation provides all the necessary details such as the full list of authors, the title of the paper, the publication date, and direct links to the paper for easy access and verification.
