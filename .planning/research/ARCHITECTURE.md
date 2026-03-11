# Architecture Patterns

**Domain:** Brownfield ML research platform — stock prediction, market adaptation (CSI 300 → S&P 500)
**Researched:** 2026-03-09
**Confidence:** HIGH (based on direct codebase analysis + established Python ML project patterns)

---

## Current Architecture (Baseline)

The existing system is a flat, script-oriented layout with six distinct execution stages that run sequentially but are not integrated into a unified pipeline. Each stage was designed for a specific remote server environment and is not runnable in a fresh clone.

```
[Jupyter notebooks]          [Python scripts]          [Training script]
factor construction    →     .npz/.npy generation  →   MultiTask_Stockformer_train.py
(manual, sequential)         (hardcoded paths)          (config-driven, but path gaps)

                                                              ↓
                                                     [output/ CSVs]
                                                              ↓
                                                     [Backtest.ipynb]
                                                     (manual notebook)
```

**Structural problems this causes:**
- Stage boundaries exist only by convention, not by code
- No shared config object spans all stages; each script parses paths independently
- Streamlit has no integration point — there is no callable API, only scripts with side effects
- S&P 500 adaptation requires touching seven files across three directories to change a dataset

---

## Recommended Architecture

### Design Principle: Preserve the Model, Restructure Around It

The model architecture (`Stockformermodel/`) and training loop logic in `MultiTask_Stockformer_train.py` are sound and should be treated as stable internals. All refactoring effort concentrates on the layer that connects data to model (the data pipeline) and the layer that connects model outputs to users (Streamlit + backtesting).

The recommended layout introduces a `src/` package so all Python code is importable, eliminating the need for `sys.path.append` hacks and enabling a test suite.

### Target Layout

```
tfm-stockformer/
├── src/
│   └── stockformer/              # installable package (pip install -e .)
│       ├── __init__.py
│       ├── config.py             # Config dataclass + loader (replaces configparser calls)
│       ├── data/
│       │   ├── __init__.py
│       │   ├── download.py       # yfinance S&P 500 OHLCV download → Parquet
│       │   ├── features.py       # Feature engineering (TA indicators, normalization)
│       │   ├── graph.py          # Struc2Vec adjacency generation (replaces preprocessing script)
│       │   └── dataset.py        # StockDataset (migrated from lib/, path from config)
│       ├── model/
│       │   ├── __init__.py
│       │   └── multitask_stockformer.py  # Model classes (migrated from Stockformermodel/)
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py        # train/val/test loop (extracted from train script)
│       │   └── metrics.py        # metric(), masked_mae etc. (migrated from lib/)
│       ├── inference/
│       │   ├── __init__.py
│       │   └── predict.py        # load checkpoint → return DataFrame of predictions
│       ├── backtest/
│       │   ├── __init__.py
│       │   └── engine.py         # Portfolio construction + performance metrics
│       └── utils/
│           ├── __init__.py
│           ├── logging.py        # log_string wrapper
│           └── graph_utils.py    # loadGraph (migrated from lib/)
├── app/
│   └── streamlit_app.py          # Streamlit UI — imports from src/stockformer/
├── scripts/
│   ├── download_sp500.py         # CLI: download S&P 500 data
│   ├── build_features.py         # CLI: raw Parquet → feature Parquet
│   ├── build_model_inputs.py     # CLI: features → .npz/.npy model inputs
│   └── train.py                  # CLI: python scripts/train.py --config config/sp500.conf
├── config/
│   ├── Multitask_Stock.conf      # Original CSI 300 config (preserved, read-only reference)
│   └── sp500.conf                # New S&P 500 config (all paths relative or env-var driven)
├── data/                         # NOT committed — generated artifacts
│   ├── raw/                      # Parquet files from yfinance
│   ├── features/                 # Feature-engineered Parquet files
│   └── model_inputs/             # .npz / .npy files consumed by StockDataset
├── output/                       # NOT committed — prediction CSVs, backtest results
├── cpt/                          # NOT committed — model checkpoints
├── tests/
│   ├── test_metrics.py
│   ├── test_dataset.py
│   └── test_model_forward.py
├── setup.py (or pyproject.toml)
└── requirements.txt              # Fixed: torch uncommented, platform-agnostic
```

---

## Component Boundaries

Each component has one defined input contract and one defined output contract. No component reaches past its boundary to read files it doesn't own.

| Component | Input | Output | Communicates With |
|-----------|-------|--------|-------------------|
| `data.download` | ticker list, date range, config paths | Parquet files in `data/raw/` | filesystem only |
| `data.features` | raw Parquet from `data/raw/` | feature Parquet in `data/features/` | filesystem only |
| `data.graph` | feature Parquet (returns/correlations) | `adjgat.npy` in `data/model_inputs/` | filesystem only |
| `data.dataset` | config (all paths), `model_inputs/` | `StockDataset` Python object in memory | `config`, filesystem |
| `model` | config (hyperparams), device | instantiated `Stockformer` nn.Module | `config` |
| `training.trainer` | `StockDataset`, `Stockformer`, config | checkpoint file, TensorBoard events, log file | `data.dataset`, `model`, `training.metrics`, filesystem |
| `inference.predict` | checkpoint path, `StockDataset`, config | DataFrame of predictions (in memory) | `model`, `data.dataset`, filesystem |
| `backtest.engine` | predictions DataFrame, prices DataFrame | performance metrics dict, equity curve DataFrame | `inference.predict` (indirectly) |
| `app.streamlit_app` | user input via UI | rendered charts and tables | `inference.predict`, `backtest.engine`, `data.download` |
| `config.py` | `.conf` file path | typed `Config` dataclass | all other components |

**Key boundary rule:** `app/streamlit_app.py` never imports from `scripts/` and never touches raw files. It calls functions from `src/stockformer/` that return DataFrames or dicts. No subprocess calls, no notebook execution.

---

## Data Flow

The data flow is strictly left-to-right through five stages. No stage reads data from a stage to its right.

```
Stage 1: Download
  yfinance API
    → data/raw/<ticker>.parquet    (daily OHLCV, per-ticker or combined)

Stage 2: Feature Engineering
  data/raw/*.parquet
    → data/features/features.parquet    (momentum, RSI, MACD, vol indicators, normalized)

Stage 3: Model Inputs
  data/features/features.parquet
    → data/model_inputs/flow.npz           (return series as model flow input)
    → data/model_inputs/trend_indicator.npz (classification labels)
    → data/model_inputs/adjgat.npy          (Struc2Vec graph embedding)

Stage 4: Training
  data/model_inputs/*.npz + *.npy  [via StockDataset]
    + config/sp500.conf
    → cpt/STOCK/saved_model_*       (checkpoint)
    → log/STOCK/log_*               (training log)
    → runs/*/                       (TensorBoard events)

Stage 5a: Inference
  cpt/STOCK/saved_model_*
  + data/model_inputs/*.npz          [via StockDataset, test split]
    → output/predictions.parquet     (regression + classification predictions with datetime index)

Stage 5b: Backtest
  output/predictions.parquet
  + data/features/features.parquet   (for price series, benchmark)
    → output/backtest_results.json   (metrics: Sharpe, drawdown, alpha)
    → output/equity_curve.parquet    (cumulative return series)

Stage 6: Streamlit UI
  Triggers stages 5a and 5b on demand
  Reads output/*.parquet for display
  Never triggers stages 1-4 (those are CLI-only pipeline steps)
```

**Intermediate file formats:**
- Raw and feature data: Parquet (columnar, fast, pandas-native)
- Model inputs: `.npz` and `.npy` (preserves existing StockDataset contract)
- Predictions and backtest results: Parquet + JSON (queryable by Streamlit without model reload)
- Checkpoints: PyTorch `.pt` (or existing extension-less convention)

---

## Patterns to Follow

### Pattern 1: Config Dataclass as Single Source of Truth

Replace scattered `configparser` calls with one config object constructed at startup and passed explicitly to all components.

```python
# src/stockformer/config.py
from dataclasses import dataclass
import configparser
from pathlib import Path

@dataclass
class StockformerConfig:
    # [file] section
    raw_data_dir: Path
    features_dir: Path
    model_inputs_dir: Path
    alpha_360_dir: Path        # was hardcoded in StockDataset
    adjgat_path: Path
    flow_path: Path
    indicator_path: Path
    model_file: Path
    log_dir: Path
    output_dir: Path
    tensorboard_dir: Path
    # [data] section
    num_nodes: int
    train_ratio: float
    val_ratio: float
    # [train] section
    max_epoch: int
    batch_size: int
    learning_rate: float
    seed: int
    device: str                # "cpu" or "cuda"
    # [param] section
    d_model: int
    n_heads: int
    # ... etc.

    @classmethod
    def from_conf(cls, path: str | Path) -> "StockformerConfig":
        cfg = configparser.ConfigParser()
        cfg.read(path)
        base = Path(path).parent.parent  # project root relative resolution
        return cls(
            raw_data_dir=base / cfg["file"]["raw_data_dir"],
            ...
        )
```

All paths in `sp500.conf` use relative paths from the project root, making the project relocatable. No file in `src/` contains any string starting with `/root/` or any absolute path.

### Pattern 2: Stage Functions with Explicit Signatures

Each pipeline stage is a plain function that takes a config and returns nothing (writes to disk) or returns a DataFrame (for in-memory stages). No global side effects.

```python
# src/stockformer/data/features.py
def build_features(cfg: StockformerConfig) -> None:
    raw = pd.read_parquet(cfg.raw_data_dir)
    # ... compute indicators ...
    features.to_parquet(cfg.features_dir / "features.parquet")
```

CLI scripts in `scripts/` call these functions and add argument parsing. Streamlit calls these same functions. The logic is never duplicated.

### Pattern 3: Inference as a Callable Function (not a script)

The single most important change for Streamlit integration. The existing `test()` function in `MultiTask_Stockformer_train.py` writes CSVs and returns nothing. It must be refactored to return a DataFrame.

```python
# src/stockformer/inference/predict.py
def run_inference(
    cfg: StockformerConfig,
    dataset: StockDataset,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """Load checkpoint, run forward pass on test split, return predictions."""
    model = Stockformer(...)
    model.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    model.eval()
    # ... batch inference ...
    return pd.DataFrame({
        "date": dates,
        "ticker": tickers,
        "pred_return": regression_preds,
        "pred_direction": class_preds,
        "true_return": regression_labels,
    })
```

Streamlit calls `run_inference()` and renders the returned DataFrame. No CSV intermediary required for the UI path (though CSV export remains available for thesis reproducibility).

### Pattern 4: Streamlit as a Thin Presentation Layer

`app/streamlit_app.py` contains only UI logic: widget definitions, layout, and calls to functions in `src/stockformer/`. It does not contain data processing logic, model code, or file I/O beyond reading pre-computed output Parquets.

```python
# app/streamlit_app.py — skeleton
import streamlit as st
from stockformer.config import StockformerConfig
from stockformer.inference.predict import run_inference
from stockformer.backtest.engine import run_backtest

cfg = StockformerConfig.from_conf("config/sp500.conf")

st.sidebar.date_input("Start date", ...)
st.sidebar.date_input("End date", ...)

if st.button("Run Predictions"):
    predictions = run_inference(cfg, ...)   # cached with @st.cache_data
    st.dataframe(predictions)

if st.button("Run Backtest"):
    results = run_backtest(predictions, ...)
    st.line_chart(results["equity_curve"])
```

Streamlit session state holds DataFrames. Model weights are loaded once and cached via `@st.cache_resource`. The training pipeline is never triggered from the UI — it is a CLI-only operation.

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Incremental Script Patching

**What it looks like:** Replace the seven hardcoded `/root/autodl-tmp/` strings one at a time without restructuring. Keep running the scripts in isolation.

**Why bad:** The Streamlit requirement cannot be satisfied by script-level patching. Scripts have side effects, write to disk, and return nothing — they cannot be called from a Streamlit button. The result is a codebase where the CLI path and the UI path diverge and duplicate logic.

**Instead:** Extract logic into functions first, then scripts and Streamlit both call the functions.

### Anti-Pattern 2: Notebook-Driven Feature Engineering in Production Path

**What it looks like:** Keep the numbered Jupyter notebooks (`1_stock_data_consolidation.ipynb` through `5_factor_verification.ipynb`) as the canonical data pipeline. Add S&P 500 cells to them.

**Why bad:** Notebooks cannot be imported, tested, or called programmatically. The Streamlit app cannot trigger a notebook. Notebooks accumulate hidden state between cells and are hard to diff in git.

**Instead:** Convert the core logic of each notebook to functions in `src/stockformer/data/features.py`. Keep one exploratory notebook in a `notebooks/` directory for EDA only — not as a pipeline stage.

### Anti-Pattern 3: Loading the Model on Every Streamlit Interaction

**What it looks like:** Call `torch.load()` and `model.load_state_dict()` inside the Streamlit button handler, so every user click reloads model weights from disk.

**Why bad:** Model loading takes 1-5 seconds. A responsive UI requires the model to stay in memory between interactions.

**Instead:** Use `@st.cache_resource` on the function that loads the model. The model is loaded once per Streamlit session and reused across button clicks.

### Anti-Pattern 4: One Monolithic Config File for All Markets

**What it looks like:** Modify `config/Multitask_Stock.conf` directly to point to S&P 500 data, overwriting the CSI 300 config.

**Why bad:** Destroys reproducibility of the original CSI 300 baseline. The thesis likely needs to reference both markets. Config drift between the two makes results non-comparable.

**Instead:** `config/Multitask_Stock.conf` is frozen as the CSI 300 reference. `config/sp500.conf` is a new file for the US market. Both share the same config schema — only path and hyperparameter values differ.

---

## What to Refactor vs What to Leave Alone

### Leave Alone (stable, working)

| Component | Reason |
|-----------|--------|
| `Stockformermodel/Multitask_Stockformer_models.py` — all model classes except `Stockformer_raw` | Core thesis artifact; working; architecture is the research contribution |
| `config/Multitask_Stock.conf` | CSI 300 baseline; freeze for reproducibility |
| The `.conf` schema (`[file]`, `[data]`, `[train]`, `[param]` sections) | Working convention; extend with new keys, don't redesign |
| `lib/graph_utils.py` — `loadGraph()` function | Works correctly; migrate location, don't rewrite |
| Train/val/test split logic | Correct time-series split; preserve as-is |

### Refactor (fix to unblock the project)

| Component | Change Required | Why |
|-----------|----------------|-----|
| All hardcoded `/root/autodl-tmp/` paths | Replace with config-driven paths | Portability — project does not run without this |
| `StockDataset.__init__` — hardcoded `alpha_360_dir` | Move to config `[file]` section | Same reason |
| `MultiTask_Stockformer_train.py` — `test()` / `test_res()` | Extract to `inference/predict.py` returning DataFrame | Required for Streamlit integration |
| `data_processing_script/` notebooks (1-5) | Convert to functions in `src/stockformer/data/features.py` | Required to replace Qlib with custom TA features for S&P 500 |
| `Backtest/Backtest.ipynb` | Convert to `src/stockformer/backtest/engine.py` | Required for Streamlit integration |
| `results_data_processing.py` | Fix `applymap` → `map`; absorb into `inference/predict.py` | Known bug; also wrong layer placement |
| `lib/Multitask_Stockformer_utils.py` | Split into `data/dataset.py`, `training/metrics.py`, `utils/logging.py` | Separation of concerns; enables targeted testing |

### Remove (dead code)

| Item | Action |
|------|--------|
| `Stockformer_raw` class | Delete — unused predecessor |
| `laplacian`, `largest_k_lamb`, `get_eigv` in `graph_utils.py` | Delete — spectral path was abandoned |
| `.ipynb_checkpoints/` directories | Remove from git, add to `.gitignore` |
| Commented-out loss weighting blocks in `train.py` | Delete — decision already made, keep code readable |

---

## Scalability Considerations

This is a thesis research platform. Scalability to production is out of scope. The relevant scalability question is: does the architecture support the full S&P 500 universe (~500 stocks)?

| Concern | Current (CSI 300, ~300 stocks) | S&P 500 (~500 stocks) | Mitigation |
|---------|-------------------------------|----------------------|------------|
| Struc2Vec graph generation | O(n^2) edge list, works at n=300 | Slow at n=500; feasible but takes hours | Apply correlation threshold (keep edges where |corr| > 0.5) before Struc2Vec |
| Alpha features CSV loading (3x at dataset init) | Slow at n=300 | Slower at n=500 | Pre-merge feature CSVs into single Parquet during Stage 2; load once |
| Model memory (dense adjacency matrix) | 300x300 = 90K entries | 500x500 = 250K entries | Fits in RAM and GPU memory at both scales |
| Streamlit inference latency | Not measured | ~30-60 seconds on CPU | Cache inference results per config; show progress bar |
| Batch size on CPU | batch_size=12 with GPU | Reduce to batch_size=4 for CPU | Add `device` key to config; document CPU batch sizing |

---

## Suggested Build Order

This order reflects dependency chains between components. Each phase produces a stable artifact the next phase depends on.

**Phase 1 — Foundation**
1. Add `.gitignore` and clean repo
2. Create `src/stockformer/` package skeleton with `setup.py` or `pyproject.toml`
3. Implement `config.py` with `StockformerConfig` dataclass; update `sp500.conf` with all relative paths
4. Fix all hardcoded paths by wiring to config (unblocks everything else)

Rationale: Nothing else can proceed reproducibly until paths are config-driven. This is the single highest-leverage change.

**Phase 2 — S&P 500 Data Pipeline**
1. Implement `data/download.py` (yfinance → Parquet)
2. Implement `data/features.py` (TA indicators replacing Qlib Alpha-360)
3. Implement `data/graph.py` (Struc2Vec adjacency, with correlation threshold)
4. Implement `data/dataset.py` (migrate `StockDataset`, wire `alpha_360_dir` from config)

Rationale: Model training for S&P 500 is impossible without this data. Produces the `.npz`/`.npy` files the existing `StockDataset` contract already expects.

**Phase 3 — Training and Inference**
1. Migrate model classes to `model/multitask_stockformer.py`; delete `Stockformer_raw`
2. Migrate metrics and losses to `training/metrics.py`; fix `wape` return
3. Extract training loop to `training/trainer.py`
4. Implement `inference/predict.py` returning DataFrame
5. Add `scripts/train.py` as the new CLI entry point

Rationale: Produces a reproducible training run on S&P 500. The DataFrame-returning inference function is the prerequisite for both backtest and Streamlit.

**Phase 4 — Backtest and Portfolio**
1. Implement `backtest/engine.py` (portfolio construction, Sharpe, drawdown, alpha vs SPY)
2. Write `scripts/run_backtest.py` for standalone CLI use
3. Validate results against the original `Backtest/Backtest.ipynb` notebook output

Rationale: Backtesting results are the primary thesis evaluation artifact. Must be validated against the existing notebook before the notebook is retired.

**Phase 5 — Streamlit Interface**
1. Implement `app/streamlit_app.py` skeleton with config loading and cached model
2. Add period selection, prediction display, backtest visualization panels
3. Wire `@st.cache_resource` for model, `@st.cache_data` for DataFrames

Rationale: Streamlit depends on callable functions from phases 3 and 4. The UI is the last layer added, not an early integration point.

**Phase 6 — Test Suite and Thesis Outputs**
1. Write `tests/test_metrics.py`, `tests/test_dataset.py`, `tests/test_model_forward.py`
2. Generate thesis charts and ablation figures from saved output Parquets
3. Document environment setup in README

---

## Sources

- Direct codebase analysis: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`, `.planning/codebase/CONCERNS.md` (2026-03-09)
- Project requirements: `.planning/PROJECT.md` (2026-03-09)
- Patterns drawn from established Python ML project conventions (`src/` layout, `pyproject.toml`, Streamlit `st.cache_resource` API — HIGH confidence, standard documented patterns)
- Streamlit caching model: `@st.cache_resource` for non-serializable objects (PyTorch models), `@st.cache_data` for DataFrames — documented in Streamlit official docs, HIGH confidence
- Struc2Vec O(n^2) scaling concern derived from `CONCERNS.md` analysis of `Stockformer_data_preprocessing_script.py` — HIGH confidence (direct code reading)
