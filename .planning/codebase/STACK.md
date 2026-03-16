# Technology Stack

**Analysis Date:** 2026-03-09

## Languages

**Primary:**
- Python 3.9 - All model training, data processing, and backtesting code
- Python 3.8 - Referenced in `.vscode/launch.json` as the original development Python (miniconda on remote server)

**Secondary:**
- INI/ConfigParser - Model configuration via `config/Multitask_Stock.conf`

## Runtime

**Environment:**
- Python 3.9 (local venv at `venv/`)
- Python 3.8 (remote server via miniconda, path: `/root/miniconda3/bin/python3.8`)
- Original execution environment: Linux-64 (AutoDL cloud GPU server at `/root/autodl-tmp/`)

**Package Manager:**
- pip (venv-based)
- Lockfile: `requirements.txt` present (partial — does not list all installed packages)

## Frameworks

**Core ML:**
- PyTorch (2.8.0 in local venv; requirements.txt comments out `torch==2.0.1+cu117` for CUDA) - Deep learning model training, `nn.Module` classes, `torch.utils.tensorboard`
- pytorch-wavelets 1.3.0 - Discrete Wavelet Transform (DWT) used for signal disentanglement into low/high frequency components

**Data Science:**
- NumPy 1.24.4 - Array operations, data loading from `.npz` files
- pandas - DataFrame manipulation, CSV I/O, time-series alignment
- scikit-learn 1.1.2 - `normalize`, `LogisticRegressionCV`, used in factor processing and graph utilities
- scipy 1.9.3 - Sparse matrix operations (`scipy.sparse`), eigendecomposition (`eigsh`), optimization (`minimize`)
- statsmodels 0.14.0 - OLS regression for factor neutralization (`data_processing_script/volume_and_price_factor_construction/4_neutralization.py`)

**Visualization:**
- matplotlib 3.7.1 - Training and backtest result plotting
- seaborn - Backtest comparison line plots (`Backtest/Backtest.ipynb`)

**Quantitative Finance:**
- qlib (Microsoft) - Alpha factor generation (Alpha158, Alpha360), baseline model training (LGB, XGB, CatBoost, LSTM, GRU, TCN, Transformer, Localformer, ALSTM, GATs), backtesting workflow; initialized with `qlib.init(provider_uri=...)`

**Graph Embedding:**
- Struc2Vec (custom `ge` library) - Graph structural embedding for stock correlation adjacency matrix, used in `data_processing_script/stockformer_input_data_processing/Stockformer_data_preprocessing_script.py`
- networkx 3.2.1 - Graph construction from edge lists

**Build/Dev:**
- TensorBoard (via `torch.utils.tensorboard.SummaryWriter`) - Training loss and validation metric logging
- tqdm 4.62.3 - Progress bars for training loops

## Key Dependencies

**Critical:**
- `torch` - Core model (`Stockformermodel/Multitask_Stockformer_models.py`), training loop (`MultiTask_Stockformer_train.py`)
- `pytorch_wavelets` - DWT disentanglement in `lib/Multitask_Stockformer_utils.py` (`DWT1DForward`, `DWT1DInverse`)
- `qlib` - Factor data source and baseline model comparison in `Backtest/Backtest.ipynb`
- `numpy` - `.npz` data loading for all model inputs

**Infrastructure:**
- `scipy.sparse` / `scipy.sparse.linalg` - Graph Laplacian and eigendecomposition in `lib/graph_utils.py`
- `pandas` - CSV-based data exchange between pipeline stages
- `networkx` - Correlation graph construction for Struc2Vec embedding

## Configuration

**Environment:**
- Model hyperparameters and file paths configured via INI file: `config/Multitask_Stock.conf`
- Sections: `[file]`, `[data]`, `[train]`, `[param]`
- Key parameters: `cuda=0`, `max_epoch=100`, `batch_size=12`, `learning_rate=0.001`, `dims=128`, `wave=sym2`
- CUDA device selected at runtime: `torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")`

**Build:**
- No build system; run directly with `python MultiTask_Stockformer_train.py --config ./config/Multitask_Stock.conf`
- VSCode launch config: `.vscode/launch.json` (targets remote server path, `PYTHONPATH` set to workspace root)

## Platform Requirements

**Development:**
- Python 3.9+ with venv
- CUDA-capable GPU strongly recommended (CUDA 11.7 referenced in comments)
- qlib data provider initialized locally at `~/.qlib/qlib_data/ch_data` or remotely

**Production:**
- Original training environment: AutoDL cloud GPU server (Linux-64), path prefix `/root/autodl-tmp/Stockformer/`
- Model checkpoints saved to `cpt/STOCK/`
- TensorBoard logs saved to `runs/`
- Prediction outputs saved to `output/`

---

*Stack analysis: 2026-03-09*
