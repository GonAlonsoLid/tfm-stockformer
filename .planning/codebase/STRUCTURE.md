# Codebase Structure

**Analysis Date:** 2026-03-09

## Directory Layout

```
tfm-stockformer/
├── MultiTask_Stockformer_train.py   # Main training entry point
├── requirements.txt                 # Python dependency list
├── README.md                        # English documentation
├── README中文版.md                  # Chinese documentation
├── config/                          # Configuration files (.conf)
│   └── Multitask_Stock.conf         # Hyperparameters, file paths, training settings
├── Stockformermodel/                # Neural network model definitions
│   └── Multitask_Stockformer_models.py
├── lib/                             # Shared utility libraries
│   ├── Multitask_Stockformer_utils.py
│   └── graph_utils.py
├── data_processing_script/          # Offline data preparation pipelines
│   ├── stockformer_input_data_processing/   # Scripts converting raw data to model inputs
│   │   ├── Stockformer_data_preprocessing_script.py
│   │   ├── data_Interception.py
│   │   └── results_data_processing.py
│   └── volume_and_price_factor_construction/  # Factor engineering notebooks
│       ├── 1_stock_data_consolidation.ipynb
│       ├── 2_data_preprocessing.ipynb
│       ├── 3_qlib_factor_construction.ipynb
│       ├── 4_neutralization.py
│       └── 5_factor_verification.ipynb
├── Backtest/                        # Backtesting notebook
│   └── Backtest.ipynb
├── cpt/                             # Model checkpoints (saved PyTorch model files)
│   └── STOCK/
│       └── saved_model_Multitask_<start>_<end>   # Binary model file
├── log/                             # Training log text files
│   └── STOCK/
│       └── log_Multitask_<start>_<end>
├── output/                          # Inference results as CSV files
│   └── Multitask_output_<start>_<end>/
│       ├── classification/          # Classification predictions and labels
│       └── regression/              # Regression predictions and labels
├── runs/                            # TensorBoard event files
│   └── Multitask_Stockformer/
│       └── Stock_CN_<start>_<end>/version0/
├── .planning/codebase/              # GSD analysis documents
└── venv/                            # Python virtual environment (not committed)
```

## Directory Purposes

**`config/`:**
- Purpose: INI-style configuration files read by `configparser` at training time
- Contains: One `.conf` file per experiment configuration (dataset, model hyperparameters, file paths)
- Key files: `config/Multitask_Stock.conf`
- Sections in conf: `[file]`, `[data]`, `[train]`, `[param]`

**`Stockformermodel/`:**
- Purpose: PyTorch `nn.Module` definitions for the Stockformer architecture
- Contains: Single file defining all model components (embeddings, attention, wavelet layers, output heads)
- Key files: `Stockformermodel/Multitask_Stockformer_models.py`

**`lib/`:**
- Purpose: Shared helper functions imported by the training script
- Contains: Loss functions, metrics, dataset class, graph loading utilities
- Key files:
  - `lib/Multitask_Stockformer_utils.py` — `log_string`, `metric`, `StockDataset`, `save_to_csv`, loss helpers
  - `lib/graph_utils.py` — `loadGraph`, Laplacian utilities

**`data_processing_script/`:**
- Purpose: Offline, pre-training data preparation; not imported during model training
- Two sub-pipelines:
  - `data_processing_script/volume_and_price_factor_construction/` — numbered Jupyter notebooks (1–5) for raw data consolidation, factor construction via Qlib, neutralization, and verification
  - `data_processing_script/stockformer_input_data_processing/` — Python scripts that convert processed CSVs into `.npz`/`.npy` arrays consumed by the training script
- Output data lands in a `data/` directory (not committed; paths are hardcoded to `/root/autodl-tmp/...` in original scripts)

**`Backtest/`:**
- Purpose: Post-training portfolio backtesting and performance evaluation
- Contains: `Backtest/Backtest.ipynb` — single Jupyter notebook

**`cpt/STOCK/`:**
- Purpose: Persisted model weights; written by training script via `args.model_file`
- Naming pattern: `saved_model_Multitask_<YYYY-MM-DD>_<YYYY-MM-DD>` (date range of training data)
- Generated: Yes — do not commit large binary files

**`log/STOCK/`:**
- Purpose: Plain-text training logs written by `log_string()` utility
- Naming pattern: `log_Multitask_<YYYY-MM-DD>_<YYYY-MM-DD>`
- Generated: Yes

**`output/Multitask_output_<start>_<end>/`:**
- Purpose: CSV dumps of model predictions and ground-truth labels after inference
- Sub-directories: `classification/` (predicted classes, probabilities) and `regression/` (predicted returns)
- Naming pattern follows training date range
- Generated: Yes

**`runs/`:**
- Purpose: TensorBoard `SummaryWriter` event files for loss/metric visualization
- Generated: Yes

**`.planning/codebase/`:**
- Purpose: GSD analysis documents (ARCHITECTURE.md, STACK.md, STRUCTURE.md, etc.)
- Generated: No — committed planning artifacts

## Key File Locations

**Entry Point:**
- `MultiTask_Stockformer_train.py`: Main script; parses CLI args, reads `.conf`, trains and evaluates model, writes checkpoints, logs, and output CSVs

**Configuration:**
- `config/Multitask_Stock.conf`: Defines all file paths and hyperparameters; passed via `--config` flag at runtime

**Model Definition:**
- `Stockformermodel/Multitask_Stockformer_models.py`: All PyTorch module classes

**Utilities:**
- `lib/Multitask_Stockformer_utils.py`: Dataset class (`StockDataset`), metrics, loss functions, CSV export
- `lib/graph_utils.py`: Graph adjacency matrix loading

**Data Pipeline (offline):**
- `data_processing_script/volume_and_price_factor_construction/1_stock_data_consolidation.ipynb` through `5_factor_verification.ipynb` — run sequentially
- `data_processing_script/stockformer_input_data_processing/data_Interception.py` — filters date ranges
- `data_processing_script/stockformer_input_data_processing/Stockformer_data_preprocessing_script.py` — converts CSVs to `.npz`/`.npy` arrays

**Backtesting:**
- `Backtest/Backtest.ipynb`

## Naming Conventions

**Files:**
- Model and utility files: `PascalCase` with underscores, prefixed by component (`Multitask_Stockformer_models.py`, `Multitask_Stockformer_utils.py`)
- Data processing scripts: `snake_case.py` or numbered `N_description.ipynb`
- Config files: `PascalCase_Dataset.conf` (e.g., `Multitask_Stock.conf`)
- Checkpoint/log files: `<type>_Multitask_<start-date>_<end-date>` (no extension for checkpoints)

**Directories:**
- Model namespace directories use `PascalCase` (`Stockformermodel/`)
- Utility/artifact directories use `snake_case` or `lowercase` (`lib/`, `config/`, `cpt/`, `log/`, `output/`, `runs/`)
- Dataset-scoped sub-directories use `STOCK` as the dataset identifier (e.g., `cpt/STOCK/`, `log/STOCK/`)

**Data directory convention (external):**
- Dataset folders follow `Stock_CN_<YYYY-MM-DD>_<YYYY-MM-DD>` inside a `data/` root
- Alpha factor sub-folders follow `Alpha_360_<YYYY-MM-DD>_<YYYY-MM-DD>`

## Where to Add New Code

**New model architecture or component:**
- Add to `Stockformermodel/Multitask_Stockformer_models.py` or create a new file in `Stockformermodel/` following the `PascalCase` naming pattern
- Import into `MultiTask_Stockformer_train.py` using `from Stockformermodel.<module> import <Class>`

**New loss function or metric:**
- Add to `lib/Multitask_Stockformer_utils.py`

**New graph utility:**
- Add to `lib/graph_utils.py`

**New experiment configuration:**
- Create a new `.conf` file in `config/`, following the structure of `config/Multitask_Stock.conf` with `[file]`, `[data]`, `[train]`, `[param]` sections

**New data preprocessing step:**
- Add a numbered script or notebook to the relevant sub-directory under `data_processing_script/`:
  - Factor engineering: `data_processing_script/volume_and_price_factor_construction/`
  - Model input conversion: `data_processing_script/stockformer_input_data_processing/`

**New backtest or analysis notebook:**
- Place in `Backtest/`

**New training entry point (e.g., single-task variant):**
- Place at the project root alongside `MultiTask_Stockformer_train.py`, following the `PascalCase_Stockformer_train.py` naming pattern

## Special Directories

**`venv/`:**
- Purpose: Python virtual environment
- Generated: Yes
- Committed: No (listed in implicit gitignore behavior; not tracked)

**`cpt/`:**
- Purpose: Model checkpoint storage
- Generated: Yes (written during training)
- Committed: Yes (current checkpoint is committed for reproducibility)

**`output/`:**
- Purpose: Inference result CSVs
- Generated: Yes
- Committed: Yes (current results are committed)

**`runs/`:**
- Purpose: TensorBoard event logs
- Generated: Yes
- Committed: Yes (current run is committed)

**`.ipynb_checkpoints/`:**
- Purpose: Jupyter autosave artifacts at root and inside `Stockformermodel/`
- Generated: Yes
- Committed: Inadvertently committed; should be gitignored

---

*Structure analysis: 2026-03-09*
