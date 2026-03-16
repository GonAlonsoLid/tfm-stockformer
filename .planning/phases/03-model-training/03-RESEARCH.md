# Phase 3: Model Training - Research

**Researched:** 2026-03-11
**Domain:** PyTorch model training, configparser-driven config, inference script design
**Confidence:** HIGH

## Summary

Phase 3 is a configuration and wiring task, not an architectural one. The Stockformer model (`MultiTask_Stockformer_train.py`) is already fully implemented and config-driven via `configparser`. The sole missing pieces are: (1) a new `.conf` file that points at the Phase 2 S&P500 outputs, and (2) a standalone inference script that reuses the existing `test_res()` logic. No model code changes are needed.

The key risk is incorrect path and dimension wiring in the config. `StockDataset` computes `infea` dynamically from the number of CSV files in `alpha_360_dir`, so no manual dimension editing is needed. The most likely failure mode is a mismatch between the `data_dir` path in the config versus where `build_pipeline.py` actually wrote the outputs.

The inference script pattern already exists inside `test()` + `test_res()` in the training script. The task is to extract those functions into a standalone `scripts/run_inference.py` that accepts `--config` and optionally `--checkpoint`.

**Primary recommendation:** Write the config file first with correct paths, run the training script with `--max_epoch 2` as a smoke test to validate data loading before committing to a full 50-epoch run.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Config filename: `config/Multitask_Stock_SP500.conf`
- Data directory: `./data/Stock_SP500_2018-2024/` (mirrors `./data/Stock_CN_2021-2024/` convention)
- Date range: 2018-01-01 to 2024-12-31
- Train/val/test ratios: keep 0.75/0.125/0.125 (matches Phase 2 split)
- All other hyperparameters (`layers`, `heads`, `dims`, `wave`, `level`) kept from original config
- `max_epoch = 50` as default in config (down from original 100)
- Overridable via `--max_epoch` CLI flag at runtime (e.g., `--max_epoch 100` for full run)
- `batch_size = 12` unchanged
- No early stopping — run full max_epoch, save best checkpoint (lowest val MAE)
- Create `scripts/run_inference.py` as a standalone script separate from the training script
- Reads config via `--config` flag (same as training script)
- Optional `--checkpoint` flag overrides `model_file` from config; defaults to config path
- Output structure unchanged: `output_dir/classification/` and `output_dir/regression/` CSVs
- Phase 2's TA feature CSVs used directly as `alpha_360_dir` — no adapter step needed
- Format: one CSV per feature, rows=dates, cols=stock tickers (wide format)
- `infea = num_feature_CSVs + 2` (StockDataset computes this dynamically at load time)

### Claude's Discretion
- Exact list of 60-80 features (researcher determines the full set from standard US TA indicators)
- How to handle stocks dropped from S&P500 during 2018-2024 (survivorship bias)
- TensorBoard log directory naming convention for S&P500 runs
- Exact `cpt/` and `log/` subdirectory naming

### Deferred Ideas (OUT OF SCOPE)
- US Alpha-360 equivalent with 200+ features
- Cross-sectional price-volume features (relative ranks, turnover proxies)
- Walk-forward cross-validation
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| MODEL-01 | New `.conf` config file targeting S&P500 data with correct paths, feature dimensions, and date ranges | Config format fully understood from existing `Multitask_Stock.conf`; all required keys identified |
| MODEL-02 | Model trains end-to-end on S&P500 data and saves checkpoints successfully | Training script flow fully analyzed; checkpoint save logic at `args.model_file` whenever val MAE improves |
</phase_requirements>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.8.0 (pinned in requirements.txt) | Model training, GPU execution | Already in project |
| pytorch-wavelets | 1.3.0 (pinned, patched) | DWT1DForward/Inverse wavelet decomposition | Required by Stockformer architecture |
| numpy | 1.24.4 (pinned) | Array loading, slicing, NPZ files | Already in project |
| pandas | 2.3.3 (pinned) | CSV feature file loading in StockDataset | Already in project |
| tensorboard | >=2.14,<3 | SummaryWriter for training loss + val metrics | Already used in training script |
| configparser | stdlib | Parses `.conf` INI files | Used by existing training script |
| argparse | stdlib | CLI flags including `--config`, `--max_epoch` | Used by existing training script |
| tqdm | 4.62.3 (pinned) | Progress bar per epoch and per batch | Already in project |

### No New Dependencies

Phase 3 requires zero new packages. All required libraries are already in `requirements.txt`.

## Architecture Patterns

### Config File Structure (INI format via configparser)

The existing `config/Multitask_Stock.conf` is the exact template to copy. The new file must have these four sections with these exact keys:

```ini
[file]
traffic = ./data/Stock_SP500_2018-2024/flow.npz
indicator = ./data/Stock_SP500_2018-2024/trend_indicator.npz
adj = ./data/Stock_SP500_2018-2024/corr_adj.npy
adjgat = ./data/Stock_SP500_2018-2024/128_corr_struc2vec_adjgat.npy
model = ./cpt/STOCK/saved_model_Multitask_SP500_2018-2024
log = ./log/STOCK/log_Multitask_SP500_2018-2024
alpha_360_dir = ./data/Stock_SP500_2018-2024/features
output_dir = ./output/Multitask_output_SP500_2018-2024
tensorboard_dir = ./runs/Multitask_Stockformer/Stock_SP500_2018-2024

[data]
dataset = STOCK
T1 = 20
T2 = 2
train_ratio = 0.75
val_ratio = 0.125
test_ratio = 0.125

[train]
cuda = 0
max_epoch = 50
batch_size = 12
learning_rate = 0.001
seed = 1

[param]
layers = 2
heads = 1
dims = 128
samples = 1
wave = sym2
level = 1
```

**Critical path notes:**
- `alpha_360_dir` must point to the subdirectory containing one CSV per TA feature. Phase 2's `feature_engineering.py` writes to `{data_dir}/features/` (confirmed by `save_feature_csvs()` writing `tmp_path / 'features' / 'ROC_5.csv'` in tests). Use `./data/Stock_SP500_2018-2024/features`.
- `adj` (`corr_adj.npy`) is loaded but then commented out in the current `loadGraph()` implementation — only `adjgat` is actually used at runtime. Both keys must still be present in the config because `argparse` registers both.
- `model` path has no extension — `torch.save()` writes the state dict directly to that path.
- `log` path has no extension — `open(args.log_file, 'w')` opens it as a plain text log.

### StockDataset `infea` Computation

```python
# Source: lib/Multitask_Stockformer_utils.py line 143
self.infea = bonus_all.shape[-1] + 2  # Last dimension of bonus_all plus one
```

`bonus_all.shape[-1]` equals the number of feature CSV files in `alpha_360_dir`. With Phase 2's 69 TA features, `infea = 69 + 2 = 71`. This is passed automatically to the Stockformer constructor — no config key needed.

### Training Script Entry Point

```python
# Source: MultiTask_Stockformer_train.py line 418
model = Stockformer(infeature, args.h*args.d, outfea_class, outfea_regress, args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)
```

Parameters: `infeature` (=`infea`=71), `args.h*args.d` (=1*128=128), `outfea_class=2`, `outfea_regress=1`, `args.L=2`, `args.h=1`, `args.d=128`, `args.s=1`, `args.T1=20`, `args.T2=2`.

### Checkpoint Save/Load Pattern

```python
# Save (training): MultiTask_Stockformer_train.py line 357
torch.save(model.state_dict(), args.model_file)  # saves whenever val MAE improves

# Load (inference): MultiTask_Stockformer_train.py line 363
model.load_state_dict(torch.load(args.model_file))
```

### Inference Script Pattern

The inference script (`scripts/run_inference.py`) must:
1. Parse `--config` and `--checkpoint` (optional, overrides `model_file` from config)
2. Load config via `configparser` (same pattern as training script)
3. Build `StockDataset(args, mode='test')`
4. Instantiate `Stockformer` with same constructor signature
5. Call `torch.load()` + `model.load_state_dict()`
6. Call `test_res()` logic inline (or import it) to produce prediction CSVs

The two functions to reuse from the training script are `test()` and `test_res()`. Since they close over `args` and `log` in the training script, the inference script must either import them carefully or reproduce them with explicit parameter passing.

**Recommended approach:** reproduce the logic inline in `run_inference.py` rather than importing from `MultiTask_Stockformer_train.py` — that module runs side effects (opens files, creates directories, creates TensorBoard writer) at import time, making it non-importable safely.

### TensorBoard Log Directory Convention

The training script auto-creates versioned subdirectories:
```python
# Source: MultiTask_Stockformer_train.py lines 101-112
subfolders = [f.name for f in os.scandir(tensorboard_folder) if f.is_dir()]
versions = [int(folder.replace('version', '')) for folder in subfolders if folder.startswith('version')]
next_version = 0 if not versions else max(versions) + 1
new_folder = os.path.join(tensorboard_folder, f'version{next_version}')
```

Each training run creates `tensorboard_dir/version0/`, `tensorboard_dir/version1/`, etc. No configuration needed — it auto-increments.

**Recommended `tensorboard_dir`:** `./runs/Multitask_Stockformer/Stock_SP500_2018-2024`

### Recommended `cpt/` and `log/` Subdirectory Naming

Following the `Stock_CN` pattern in the original config:
- Checkpoint: `./cpt/STOCK/saved_model_Multitask_SP500_2018-2024`
- Log: `./log/STOCK/log_Multitask_SP500_2018-2024`

The training script calls `os.makedirs()` for both directories automatically if they don't exist.

### Recommended Project Structure After Phase 3

```
config/
├── Multitask_Stock.conf          # Original CN config (unchanged)
└── Multitask_Stock_SP500.conf    # NEW: S&P500 config

scripts/
├── build_pipeline.py             # Phase 2 pipeline orchestrator
├── smoke_test.py                 # Phase 1 smoke test
└── run_inference.py              # NEW: standalone inference script

cpt/STOCK/
└── saved_model_Multitask_SP500_2018-2024   # Checkpoint (created at training time)

log/STOCK/
└── log_Multitask_SP500_2018-2024           # Training log (created at training time)

output/Multitask_output_SP500_2018-2024/
├── classification/
│   ├── classification_pred_last_step.csv
│   └── classification_label_last_step.csv
└── regression/
    ├── regression_pred_last_step.csv        # Used by Phase 4 evaluation
    └── regression_label_last_step.csv

runs/Multitask_Stockformer/Stock_SP500_2018-2024/
└── version0/                               # TensorBoard events
```

### Anti-Patterns to Avoid

- **Hardcoding `infea`:** Never set `infea` manually in the config. `StockDataset` computes it dynamically from the number of CSV files. Adding or removing features will break the model if `infea` is hardcoded.
- **Importing from training script:** `MultiTask_Stockformer_train.py` runs `open(args.log_file, 'w')`, `os.makedirs()`, and `SummaryWriter()` at module import time (not inside `if __name__ == '__main__'`). Importing it from inference script will fail with missing config/files.
- **Relative paths from wrong cwd:** All `.conf` paths start with `./`. The training script must be launched from the project root, not from within `scripts/`. This matches existing usage.
- **Missing output subdirectory:** `test_res()` creates `output_dir/classification/` and `output_dir/regression/` via `os.makedirs(..., exist_ok=True)`. The inference script must do the same or reuse this pattern.
- **GPU fallback oversight:** `device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")` — on MPS (Apple Silicon) or CPU-only machines, `cuda=0` in config falls back gracefully to CPU. No config change needed for local dev.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config parsing | Custom parser | `configparser.ConfigParser()` | Already wired in training script; all args registered |
| Directory creation | Manual `os.path.exists` checks | `os.makedirs(..., exist_ok=True)` | Already used in `test_res()` and training script |
| Checkpoint saving | Custom serialization | `torch.save(model.state_dict(), path)` | PyTorch standard; already in training script |
| Dataset loading | Custom data loader | `StockDataset(args, mode='test')` | Already validated against Phase 2 output format |
| Inference metrics | Custom metric code | `metric()` from `lib/Multitask_Stockformer_utils.py` | Already implemented |
| CSV output | Custom writer | `save_to_csv()` from `lib/Multitask_Stockformer_utils.py` | Already produces correct format for Phase 4 |

**Key insight:** Phase 3 is almost entirely configuration, not code. The only new code is the inference script, which is a thin wrapper around existing functions.

## Common Pitfalls

### Pitfall 1: Wrong `alpha_360_dir` Path

**What goes wrong:** `StockDataset.__init__()` calls `os.listdir(path)` on `alpha_360_dir`. If the path points to the parent data directory instead of the `features/` subdirectory, it will try to read `flow.npz`, `trend_indicator.npz`, etc. as CSVs and crash with a pandas parse error.

**Why it happens:** `build_pipeline.py` writes feature CSVs to `{data_dir}/features/`. The config must point to `./data/Stock_SP500_2018-2024/features`, not `./data/Stock_SP500_2018-2024/`.

**How to avoid:** Verify the exact subdirectory written by `feature_engineering.py` (confirmed as `features/` in test fixtures: `tmp_path / 'features' / 'ROC_5.csv'`).

**Warning signs:** Error message mentioning CSV parse failure on a `.npz` or `.npy` file.

### Pitfall 2: Training Script Side Effects at Import

**What goes wrong:** If inference script does `from MultiTask_Stockformer_train import test_res`, Python executes the module top-level code: `open(args.log_file, 'w')` fails because args aren't configured, or silently creates empty log/checkpoint directories.

**Why it happens:** All setup code in the training script is at module level, not guarded by `if __name__ == '__main__'`.

**How to avoid:** The inference script must be entirely standalone — copy/inline the `test_res()` logic rather than importing it. Alternatively, import only from `lib/` modules (`StockDataset`, `metric`, `save_to_csv`, `loadGraph`) and `Stockformermodel/`.

**Warning signs:** `FileNotFoundError` or `AttributeError` on `args.log_file` at import time.

### Pitfall 3: Missing `corr_adj.npy` Causing KeyError

**What goes wrong:** Config key `adj` points to `corr_adj.npy`. The file is loaded by `loadGraph()` but the load line is commented out in the current implementation. However, `argparse` still registers `--adj_file` from the config. If the file doesn't exist, no error occurs at runtime (it's never actually `np.load()`ed). But if Phase 2 failed to produce it, the build is incomplete.

**Why it happens:** `loadGraph()` only loads `adjgat` currently (`adj` load is commented out). Both files must still exist for pipeline completeness.

**How to avoid:** Confirm Phase 2 produces `corr_adj.npy` before starting Phase 3. `build_pipeline.py` sentinels check for `128_corr_struc2vec_adjgat.npy` (adjgat) but not `corr_adj.npy` separately.

**Warning signs:** Phase 4 evaluation breaks if `adj` is later re-enabled.

### Pitfall 4: `cuda=0` Config Value on CPU-Only Machine

**What goes wrong:** Training script parses `cuda = 0` from config and passes it as `f"cuda:0"`. On a Mac or CPU-only machine, `torch.cuda.is_available()` returns `False` and it correctly falls back to CPU. But if user explicitly passes `--cuda 0` on a machine with no GPU, PyTorch raises `AssertionError: Invalid device id`.

**Why it happens:** The fallback logic checks `torch.cuda.is_available()` but the explicit CLI flag bypasses this check when `cuda` resolves to an actual device string.

**How to avoid:** Train with `cuda = 0` in config and rely on the fallback. Document this in the config comments.

**Warning signs:** `AssertionError: Invalid device id` at model initialization.

### Pitfall 5: Temporal Embedding Assumes 50-Slot Time-of-Day (Not Daily)

**What goes wrong:** `generate_temporal_embeddings()` in `Multitask_Stockformer_utils.py` assumes intraday time slots (21 slots per day, 12 months). For daily stock data, the TE encoding is not semantically meaningful, but the model still trains because it's just an embedding input — not a strict time constraint.

**Why it happens:** The original code was designed for intraday traffic data and repurposed for stock data. The S&P500 adaptation inherits this.

**How to avoid:** Leave it as-is per the thesis framing (preserve architecture, don't alter). The temporal embedding still encodes monotonic temporal position, just not literally "time of day."

**Warning signs:** None — the model trains correctly even with a non-ideal TE encoding.

## Code Examples

### Minimal Training Smoke Test (Verify Config Works)

```bash
# Run from project root — 2 epochs to verify data loading, checkpoint creation, TensorBoard
python MultiTask_Stockformer_train.py \
    --config config/Multitask_Stock_SP500.conf \
    --max_epoch 2

# Expected: creates cpt/STOCK/saved_model_Multitask_SP500_2018-2024, log file, version0/ TensorBoard dir
```

### Run Inference on Saved Checkpoint

```bash
python scripts/run_inference.py \
    --config config/Multitask_Stock_SP500.conf

# With explicit checkpoint override:
python scripts/run_inference.py \
    --config config/Multitask_Stock_SP500.conf \
    --checkpoint ./cpt/STOCK/saved_model_Multitask_SP500_2018-2024
```

### Inference Script Skeleton (Key Pattern)

```python
# scripts/run_inference.py — standalone, does NOT import from training script
import argparse
import configparser
import os
import math
import numpy as np
import torch
from lib.Multitask_Stockformer_utils import StockDataset, metric, save_to_csv, log_string
from lib.graph_utils import loadGraph
from Stockformermodel.Multitask_Stockformer_models import Stockformer

# Two-phase arg parsing: config first, then override with CLI flags
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None)
args_init, _ = parser.parse_known_args()

config = configparser.ConfigParser()
config.read(args_init.config)

# Register all config values as argparse defaults
parser.add_argument('--cuda', default=config['train']['cuda'])
parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
parser.add_argument('--T1', type=int, default=config['data']['T1'])
parser.add_argument('--T2', type=int, default=config['data']['T2'])
parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'])
parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])
parser.add_argument('--L', type=int, default=config['param']['layers'])
parser.add_argument('--h', type=int, default=config['param']['heads'])
parser.add_argument('--d', type=int, default=config['param']['dims'])
parser.add_argument('--j', type=int, default=config['param']['level'])
parser.add_argument('--s', type=float, default=config['param']['samples'])
parser.add_argument('--w', default=config['param']['wave'])
parser.add_argument('--traffic_file', default=config['file']['traffic'])
parser.add_argument('--indicator_file', default=config['file']['indicator'])
parser.add_argument('--adj_file', default=config['file']['adj'])
parser.add_argument('--adjgat_file', default=config['file']['adjgat'])
parser.add_argument('--model_file', default=config['file']['model'])
parser.add_argument('--alpha_360_dir', default=config['file']['alpha_360_dir'])
parser.add_argument('--output_dir', default=config['file']['output_dir'])
args = parser.parse_args()

# --checkpoint flag overrides model_file
checkpoint_path = args.checkpoint if args.checkpoint else args.model_file
device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
```

### Verifying Phase 2 Outputs Are Present Before Training

```python
import os
DATA_DIR = "./data/Stock_SP500_2018-2024"
required = [
    "flow.npz",
    "trend_indicator.npz",
    "corr_adj.npy",
    "128_corr_struc2vec_adjgat.npy",
    "features",  # directory
]
for item in required:
    path = os.path.join(DATA_DIR, item)
    assert os.path.exists(path), f"Missing Phase 2 output: {path}"
feature_count = len(os.listdir(os.path.join(DATA_DIR, "features")))
assert feature_count >= 60, f"Need >= 60 features, got {feature_count}"
print(f"Phase 2 outputs present. Feature count: {feature_count}. infea = {feature_count + 2}")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `applymap` (pandas) | `map` (pandas 2.x) | Pandas 2.0 | Fixed in Plan 01-01 — no further action |
| `eigsh` from `scipy.sparse.linalg.eigen.arpack` | `from scipy.sparse.linalg import eigsh` | Scipy 1.8+ | Fixed in existing codebase |
| Hardcoded `/root/autodl-tmp/` paths | Config-driven paths via `.conf` | Phase 1 | INFRA-01 requirement |

**Deprecated/outdated in original codebase:**
- `adj` (Laplacian-based graph) is loaded but commented out in `loadGraph()`. Only `adjgat` (Struc2Vec embedding) is used. The `adj` file must still exist but is not an active input to the model.
- Weighted loss (`w1*loss_regress + w2*loss_class`) was tested and reverted to simple sum (`loss_regress + loss_class`). The commented code in the training script confirms this.

## Open Questions

1. **Exact `features/` subdirectory name written by Phase 2**
   - What we know: Tests confirm `save_feature_csvs()` writes to `tmp_path / 'features' / 'ROC_5.csv'`
   - What's unclear: Whether `build_pipeline.py` uses exactly `features/` or a dated subdirectory like `Alpha_360_2018-01-01_2024-12-31/` (the CN config uses `Alpha_360_2021-06-04_2024-01-30/`)
   - Recommendation: Verify with `ls data/Stock_SP500_2018-2024/` after running `build_pipeline.py`, then update config accordingly. The safe fallback is to check `feature_engineering.py`'s `save_feature_csvs()` implementation for the exact output path.

2. **Survivorship bias handling (Claude's Discretion)**
   - What we know: yfinance downloads current S&P500 constituents; stocks delisted 2018-2024 are absent
   - What's unclear: Whether the thesis narrative needs to explicitly acknowledge this or simply note it as a limitation
   - Recommendation: Note in thesis as a known limitation; do not attempt to retrieve historical constituent lists (out of scope per CONTEXT.md deferred section)

3. **`cuda` config on macOS with MPS**
   - What we know: `torch.cuda.is_available()` returns `False` on macOS; training falls back to CPU
   - What's unclear: Whether MPS acceleration (`torch.backends.mps.is_available()`) should be exploited
   - Recommendation: Leave as-is; CPU training with 500 stocks × 69 features × 50 epochs is slow but feasible for thesis purposes. Thesis scope does not require optimized training speed.

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >=7.0,<8 |
| Config file | none (no pytest.ini; project root used implicitly) |
| Quick run command | `pytest tests/test_model_training.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| MODEL-01 | Config file exists at correct path with all required keys | unit | `pytest tests/test_model_training.py::test_sp500_config_exists -x` | ❌ Wave 0 |
| MODEL-01 | Config `alpha_360_dir` points to a directory with >= 60 CSV files | unit | `pytest tests/test_model_training.py::test_config_alpha_360_dir_valid -x` | ❌ Wave 0 |
| MODEL-01 | Config date range keys present and parseable | unit | `pytest tests/test_model_training.py::test_config_keys_complete -x` | ❌ Wave 0 |
| MODEL-02 | StockDataset loads test split with correct shapes (no NaN, infea=71) | unit | `pytest tests/test_model_training.py::test_stockdataset_sp500_shapes -x` | ❌ Wave 0 |
| MODEL-02 | Stockformer forward pass smoke test (random input, correct output shape) | unit | `pytest tests/test_model_training.py::test_stockformer_forward_pass -x` | ❌ Wave 0 |
| MODEL-02 | Checkpoint file exists and loadable after training completes | integration | manual — requires full training run | manual-only |
| MODEL-02 | Inference script produces regression_pred_last_step.csv in output_dir | integration | manual — requires checkpoint | manual-only |

Note: TEST-03 from REQUIREMENTS.md ("Unit test for model forward pass — smoke test with random input") maps directly to `test_stockformer_forward_pass` above. This is the Phase 3 contribution to TEST-03.

### Sampling Rate

- **Per task commit:** `pytest tests/test_model_training.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** All unit tests green + manual verification that checkpoint and prediction CSV exist before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_model_training.py` — covers MODEL-01 (config validation) and MODEL-02 (dataset shapes, forward pass)
- [ ] No new conftest.py fixtures needed — `tests/conftest.py` already has `project_root` fixture

## Sources

### Primary (HIGH confidence)

- Direct code read: `MultiTask_Stockformer_train.py` — full training loop, argparse wiring, checkpoint save/load, TensorBoard setup, `test_res()` output CSV paths
- Direct code read: `lib/Multitask_Stockformer_utils.py` — `StockDataset.__init__()`, `infea` computation, `metric()`, `save_to_csv()`
- Direct code read: `lib/graph_utils.py` — `loadGraph()` only loads `adjgat`; `adj` load is commented out
- Direct code read: `config/Multitask_Stock.conf` — exact INI structure, all required keys
- Direct code read: `tests/test_data_pipeline.py` — confirms 69 features and `features/` subdirectory path
- Direct code read: `tests/conftest.py` — existing fixtures and pytest setup
- Direct code read: `scripts/build_pipeline.py` — sentinel files, step order, output paths
- Direct code read: `requirements.txt` — pinned versions for all dependencies

### Secondary (MEDIUM confidence)

- `tests/test_data_pipeline.py` line 150: `tmp_path / 'features' / 'ROC_5.csv'` — infers `features/` as the exact subdirectory name, but the actual `feature_engineering.py` implementation was not read to confirm this directly

### Tertiary (LOW confidence)

- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all dependencies read directly from `requirements.txt` and confirmed in source code
- Architecture: HIGH — config structure, training loop, dataset loading, inference pattern all read from source
- Pitfalls: HIGH for import side-effect and path pitfalls (code-verified); MEDIUM for GPU/MPS pitfall (logic-verified)

**Research date:** 2026-03-11
**Valid until:** 2026-06-11 (stable codebase; no external dependencies changing)
