# Architecture

**Analysis Date:** 2026-03-09

## Pattern Overview

**Overall:** Research ML pipeline — data preprocessing, graph-augmented transformer training, inference, and backtesting as distinct sequential stages.

**Key Characteristics:**
- Multitask learning: simultaneous regression (return prediction) and classification (trend direction) heads share a common backbone
- Wavelet decomposition of input time series into low-frequency (XL) and high-frequency (XH) components before encoding
- Graph-structured inter-stock relationships encoded via Struc2Vec embeddings used as spatial positional encodings
- Dual encoder processes XL and XH streams independently, then fuses them adaptively before multitask output heads
- Configuration-driven execution: all hyperparameters and file paths sourced from `.conf` files via `configparser`

## Layers

**Data Preprocessing:**
- Purpose: Convert raw stock CSV data into model-ready `.npz` arrays and graph adjacency embeddings
- Location: `data_processing_script/`
- Contains: Jupyter notebooks (factor construction), Python scripts (date slicing, flow/indicator/adj generation)
- Depends on: Raw label CSVs, Alpha-360 factor CSVs
- Used by: Training entry point via config file paths

**Utility / Support:**
- Purpose: Dataset class, loss functions, metrics, wavelet disentanglement, logging helpers
- Location: `lib/Multitask_Stockformer_utils.py`, `lib/graph_utils.py`
- Contains: `StockDataset`, `disentangle()`, `metric()`, `_compute_regression_loss()`, `_compute_class_loss()`, `loadGraph()`
- Depends on: NumPy, PyTorch, pytorch-wavelets, pandas, scikit-learn
- Used by: `MultiTask_Stockformer_train.py`

**Model Definition:**
- Purpose: All neural network module definitions
- Location: `Stockformermodel/Multitask_Stockformer_models.py`
- Contains: `Stockformer`, `StockformerBackbone`, `StockformerOutput`, `dualEncoder`, `temporalAttention`, `sparseSpatialAttention`, `adaptiveFusion`, `temporalConvNet`, `FeedForward`, `temporalEmbedding`
- Depends on: PyTorch, pytorch-wavelets (indirectly)
- Used by: `MultiTask_Stockformer_train.py`

**Training Orchestration:**
- Purpose: Full train/val/test loop with checkpointing and TensorBoard logging
- Location: `MultiTask_Stockformer_train.py`
- Contains: `train()`, `res()` (validation), `test()`, `test_res()`, main data loading block
- Depends on: `lib/`, `Stockformermodel/`, `config/`
- Used by: User/CLI invocation

**Backtesting:**
- Purpose: Post-hoc portfolio construction and performance evaluation of model predictions
- Location: `Backtest/Backtest.ipynb`
- Contains: Jupyter notebook consuming output CSVs
- Depends on: `output/` prediction CSVs
- Used by: Researcher analysis

**Output Post-processing:**
- Purpose: Attach date/stock index back to raw prediction arrays, compute softmax probabilities
- Location: `data_processing_script/stockformer_input_data_processing/results_data_processing.py`
- Contains: `load_and_index_data()`, `apply_extraction_and_softmax()`
- Depends on: `output/` raw CSV files, label reference CSV
- Used by: Researcher / backtest pipeline

## Data Flow

**Preprocessing Pipeline:**

1. Raw stock data (price/volume) → `data_processing_script/volume_and_price_factor_construction/` notebooks build Alpha-360 factors
2. `data_Interception.py` slices label CSV and Alpha-360 CSVs to a target date range
3. `Stockformer_data_preprocessing_script.py` converts sliced label CSV → `flow.npz`, `trend_indicator.npz`, `corr_adj.npy`, edge list → `128_corr_struc2vec_adjgat.npy` (Struc2Vec graph embeddings)
4. Resulting `.npz`/`.npy` files are referenced in `config/Multitask_Stock.conf`

**Training Data Flow:**

1. `StockDataset.__init__()` in `lib/Multitask_Stockformer_utils.py` loads `flow.npz`, `trend_indicator.npz`, and Alpha-360 CSV folder
2. `disentangle()` applies DWT (Discrete Wavelet Transform) to split time series into low (XL) and high (XH) frequency components
3. `seq2instance()` converts time series to sliding-window input/output pairs of length T1/T2
4. `loadGraph()` in `lib/graph_utils.py` loads the Struc2Vec adjacency embedding (`adjgat`)
5. `Stockformer.forward(xl, xh, te, bonus, indicator, adjgat)` runs the dual encoder and output heads
6. Combined regression + classification loss computed via `masked_mae` and `CrossEntropyLoss`
7. Best model checkpoint saved to `cpt/STOCK/`; training logs written to `log/STOCK/`; TensorBoard events to `runs/`

**Inference / Test Flow:**

1. `test()` loads best checkpoint from `cpt/STOCK/`
2. `test_res()` runs batch inference; collects `pred_class`, `pred_regress`, `label_class`, `label_regress`
3. Raw predictions saved to `output/Multitask_output_*/classification/` and `output/Multitask_output_*/regression/` as CSV
4. `results_data_processing.py` reattaches datetime index and stock columns, computes softmax probabilities
5. Post-processed CSVs fed into `Backtest/Backtest.ipynb`

**State Management:**
- No in-memory global state beyond the PyTorch model and data arrays loaded at startup
- `device` is set as a module-level global in `Stockformermodel/Multitask_Stockformer_models.py` and `lib/Multitask_Stockformer_utils.py`
- Training state (best MAE, epoch) managed as local variables inside `train()`

## Key Abstractions

**Stockformer (top-level model):**
- Purpose: Combines backbone feature extractor with multitask output heads
- Examples: `Stockformermodel/Multitask_Stockformer_models.py` — `class Stockformer`
- Pattern: Composition of `StockformerBackbone` + `StockformerOutput`; MAML-style init applied to classifier

**StockformerBackbone:**
- Purpose: Wavelet-domain dual-stream spatio-temporal encoder
- Examples: `Stockformermodel/Multitask_Stockformer_models.py` — `class StockformerBackbone`
- Pattern: Input embedding → stacked `dualEncoder` layers → projection heads → `adaptiveFusion`

**dualEncoder:**
- Purpose: Single encoder block operating on both low and high frequency streams in parallel
- Examples: `Stockformermodel/Multitask_Stockformer_models.py` — `class dualEncoder`
- Pattern: XL through `temporalAttention` + `sparseSpatialAttention`; XH through `temporalConvNet` + `sparseSpatialAttention`

**StockDataset:**
- Purpose: PyTorch Dataset encapsulating all data loading, DWT decomposition, and sliding-window sampling
- Examples: `lib/Multitask_Stockformer_utils.py` — `class StockDataset`
- Pattern: `__init__` does all preprocessing; `__getitem__` returns pre-computed arrays

## Entry Points

**Training Entry Point:**
- Location: `MultiTask_Stockformer_train.py`
- Triggers: `python MultiTask_Stockformer_train.py --config config/Multitask_Stock.conf`
- Responsibilities: Parse config, load datasets, instantiate model, run `train()` loop, run `test()`

**Data Preprocessing Entry Points:**
- Location: `data_processing_script/stockformer_input_data_processing/data_Interception.py` (date slicing)
- Location: `data_processing_script/stockformer_input_data_processing/Stockformer_data_preprocessing_script.py` (flow/adj generation)
- Triggers: Direct Python execution with hardcoded paths (designed for remote GPU server)

**Results Post-processing Entry Point:**
- Location: `data_processing_script/stockformer_input_data_processing/results_data_processing.py`
- Triggers: Direct Python execution after test inference completes

**Backtest Entry Point:**
- Location: `Backtest/Backtest.ipynb`
- Triggers: Jupyter notebook execution consuming post-processed output CSVs

## Error Handling

**Strategy:** Minimal — bare exception catch in `test()` for `EOFError` on corrupted checkpoint files; otherwise errors propagate as Python exceptions.

**Patterns:**
- Directory existence checked and created with `os.makedirs()` before writing log/model/output files
- `np.nan_to_num()` and masked MAE used to handle NaN/zero values in financial time series
- No retry logic or structured error hierarchy

## Cross-Cutting Concerns

**Logging:** Plain-text file logging via `log_string()` in `lib/Multitask_Stockformer_utils.py`; additionally TensorBoard `SummaryWriter` for scalar metrics per epoch

**Validation:** No runtime schema/data validation; correctness relies on correct file format conventions (`.npz` with `result` key, specific CSV layouts)

**Reproducibility:** Controlled via `seed` in config, applied to `random`, `numpy`, `torch`, and `torch.cuda` seeds at startup

---

*Architecture analysis: 2026-03-09*
