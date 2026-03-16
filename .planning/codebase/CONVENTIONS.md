# Coding Conventions

**Analysis Date:** 2026-03-09

## Naming Patterns

**Files:**
- Script entry points use PascalCase with underscores: `MultiTask_Stockformer_train.py`
- Utility modules use PascalCase with underscores: `Multitask_Stockformer_utils.py`, `graph_utils.py`
- Data processing scripts use snake_case or numbered prefixes: `data_Interception.py`, `4_neutralization.py`
- Jupyter notebooks use numbered prefixes: `1_stock_data_consolidation.ipynb`, `2_data_preprocessing.ipynb`
- Model files follow pattern: `Multitask_Stockformer_models.py`

**Classes:**
- Neural network modules use camelCase starting with lowercase for component type: `temporalEmbedding`, `sparseSpatialAttention`, `temporalAttention`, `adaptiveFusion`, `dualEncoder`
- Top-level model classes use PascalCase: `Stockformer`, `StockformerBackbone`, `StockformerOutput`, `StockDataset`
- Helper classes mix conventions: `Chomp1d`, `FeedForward`, `temporalConvNet`

**Functions:**
- Top-level functions use snake_case: `log_string`, `metric`, `masked_mae`, `disentangle`, `loadGraph`, `save_to_csv`
- Training/evaluation functions use snake_case: `train`, `test`, `res`, `test_res`
- Internal helper functions use snake_case with underscore prefix for "private" intent: `_compute_regression_loss`, `_compute_class_loss`
- Data processing functions use snake_case: `filter_date_range`, `save_filtered_data`, `generate_temporal_embeddings`

**Variables:**
- Short single-letter names are common for tensors in forward methods: `x`, `Q`, `K`, `V`, `B`, `T`, `N`, `D`
- Batch data uses descriptive snake_case: `trainXL`, `trainXH`, `valXL`, `pred_class`, `label_regress`
- Model predictions use `hat_y` prefix: `hat_y_class`, `hat_y_regress`, `hat_y_l_class`
- Configuration parameters use short single-letter args: `args.h`, `args.d`, `args.L`, `args.s`, `args.w`, `args.j`
- Loop indices: `i`, `j`, `batch_idx`, `start_idx`, `end_idx`

**Constants/Config Keys:**
- Config sections are uppercase: `[file]`, `[data]`, `[train]`, `[param]`
- Config keys are lowercase snake_case: `learning_rate`, `batch_size`, `train_ratio`

## Code Style

**Formatting:**
- No auto-formatter detected (no `.prettierrc`, `.black`, `pyproject.toml`, or similar)
- Mixed indentation style: 4 spaces consistently within files
- Blank lines used to separate logical blocks inside functions but not strictly between all methods
- Inline comments are in English for explanations (e.g. `# Initialize parser`, `# Regression task metric computation`)

**Linting:**
- No linting configuration detected (no `.flake8`, `.pylintrc`, `setup.cfg`)
- `warnings.filterwarnings('ignore')` used in `4_neutralization.py` to suppress all warnings

## Import Organization

**Order (observed pattern):**
1. Standard library imports (`os`, `sys`, `math`, `csv`, `time`, `random`, `re`)
2. Third-party imports (`numpy`, `pandas`, `torch`, `sklearn`, `scipy`, `tqdm`, `matplotlib`)
3. Local/project imports (`from lib.X import ...`, `from Stockformermodel.X import ...`)

**Path Aliases:**
- No path aliases; local imports use direct relative module paths: `from lib.Multitask_Stockformer_utils import ...`
- `sys.path.append(...)` is used in some scripts to add remote server paths: `sys.path.append('/root/autodl-tmp/...')`

**Wildcard imports:**
- `from numpy import *` used in `4_neutralization.py` (anti-pattern, pulls all numpy symbols into namespace)

## Error Handling

**Patterns:**
- Minimal error handling; exceptions are caught only at specific bottlenecks
- Model loading in `MultiTask_Stockformer_train.py` uses a bare `except EOFError` catch: the `test()` function catches `EOFError` when loading a checkpoint and prints a message before returning early
- No custom exception classes defined
- `np.nan_to_num()` is used defensively in metric calculations to prevent NaN propagation
- `np.errstate(divide='ignore', invalid='ignore')` used in `metric()` to suppress numpy division warnings locally
- Directory creation uses `os.makedirs()` with `exist_ok=True` in some places and manual `if not os.path.exists()` checks in others (inconsistent)

## Logging

**Framework:** Custom `log_string` function defined in `lib/Multitask_Stockformer_utils.py`

```python
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)
```

**Patterns:**
- Log file is opened as a plain file handle at the top of the training script and passed as a parameter
- All structured training output (epoch, loss, metrics) goes through `log_string`
- Ad-hoc `print()` used in data-processing scripts and for non-training messages
- TensorBoard used for scalar metric visualization: `tensor_writer.add_scalar(...)`
- Log file path is configured via `config/Multitask_Stock.conf`

## Comments

**When to Comment:**
- Chinese-language inline comments used extensively to explain intent and data shapes
- Tensor shape annotations are common in docstrings and inline: `'''TE:[B,T,2]'''`, `# [B,T,N,F]`
- Large blocks of commented-out code are left in place (dead code, not removed): commented-out weight balancing logic, alternative model loading, alternative LR scheduler targets

**JSDoc/TSDoc:**
- No type annotations or docstrings in the Python standard sense
- Shape comments are the primary documentation: `'''x:[B,T,N]\nbatch_size:[B,T,N,D2]'''`

## Function Design

**Size:**
- Training/evaluation functions are large (50-100 lines); no decomposition into smaller units
- Model `forward()` methods are concise (10-20 lines each)

**Parameters:**
- Heavy use of positional parameters with short names
- The main `train()` function takes 17 positional parameters; no use of dataclasses or parameter objects
- `args` namespace object from `argparse` is used as a global configuration object, accessed from within functions via closure

**Return Values:**
- Evaluation functions return tuples: `(acc, mae, rmse, mape)`
- Model `forward()` returns tuples: `(hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress)`

## Module Design

**Exports:**
- No `__all__` declarations; all public names are importable
- Utilities are gathered in `lib/Multitask_Stockformer_utils.py`; models in `Stockformermodel/Multitask_Stockformer_models.py`

**Global State:**
- `device` variable is declared as a module-level global and mutated via `global device` in model `__init__` methods: `global device; device = dev`
- `criterion` (CrossEntropyLoss) is a module-level global in `lib/Multitask_Stockformer_utils.py`
- `args` (argparse namespace) is used as a de-facto global in the training script

**Hardcoded Paths:**
- Several absolute paths pointing to the original training server (`/root/autodl-tmp/...`) are hardcoded in source files, making scripts non-portable:
  - `MultiTask_Stockformer_train.py`: `tensorboard_folder = '/root/autodl-tmp/...'`
  - `lib/Multitask_Stockformer_utils.py`: `path = '/root/autodl-tmp/...'`
  - `data_processing_script/stockformer_input_data_processing/results_data_processing.py`

---

*Convention analysis: 2026-03-09*
