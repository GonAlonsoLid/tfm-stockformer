# Phase 1: Infrastructure - Research

**Researched:** 2026-03-10
**Domain:** Python environment portability, config-driven path management, requirements pinning
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| INFRA-01 | All hardcoded `/root/autodl-tmp/` paths replaced with config-driven paths so the pipeline runs on any machine | Exact file locations and line numbers catalogued; fix pattern confirmed (add keys to `.conf` `[file]` section, pass through `args`) |
| INFRA-02 | Working `requirements.txt` with pinned versions enabling a clean local install from scratch | Venv audited; two missing packages identified (`PyWavelets`, `tensorboard`); `torch` commented out — exact additions documented |
| INFRA-03 | Setup documentation that reproduces the environment from a fresh clone | Smoke test command defined; 30-minute onboarding goal feasible with a single `SETUP.md` covering venv creation, pip install, and smoke test |
</phase_requirements>

---

## Summary

The codebase was built and trained on a single AutoDL cloud GPU server at `/root/autodl-tmp/`. Every path in every file is hardcoded to that absolute prefix. This is the primary portability blocker: the training script, the data preprocessing scripts, and the post-processing script all fail immediately on any other machine — before a single line of model code runs.

The `requirements.txt` is incomplete in two directions: (1) `torch` is commented out entirely, so pip-installing from the file does not install the core ML framework; (2) two transitive dependencies that `pytorch-wavelets` and `torch.utils.tensorboard` require at import time — `PyWavelets` and `tensorboard` — are absent from the file. All other listed packages import successfully against the local venv.

The fix strategy is mechanical and low-risk: move every hardcoded path into `config/Multitask_Stock.conf` under the `[file]` section, thread new keys through `args` in `MultiTask_Stockformer_train.py`, update the three data processing scripts to accept CLI arguments or read from environment, add the missing packages to `requirements.txt` with pinned versions, add a `.gitignore`, and write a `SETUP.md` with a four-step onboarding sequence.

**Primary recommendation:** Fix paths file-by-file against the exact line catalogue below, update `requirements.txt` with the three missing entries, and ship a `SETUP.md` with a smoke test command that does not require training data (import model, load config, instantiate `Stockformer`).

---

## Standard Stack

### Core (already in project, versions verified in venv)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Python | 3.9 (venv) | Runtime | Original development Python; venv present |
| torch | 2.8.0 (venv installed) | Model training | Core DL framework; must be in requirements.txt |
| pytorch-wavelets | 1.3.0 | DWT disentanglement | Used in `StockDataset.disentangle()` and `lib/` |
| PyWavelets | 1.6.0 | Required by pytorch-wavelets | `pytorch_wavelets` imports `pywt` at module load |
| tensorboard | (latest stable) | TensorBoard `SummaryWriter` | `torch.utils.tensorboard` imports `tensorboard` at module load |
| numpy | 1.24.4 | Array I/O, `.npz` files | Pinned; compatible with torch 2.8 |
| pandas | 2.3.3 | CSV I/O, time-series | Installed version; `.map()` replaces deprecated `.applymap()` |
| scikit-learn | 1.1.2 | `normalize`, graph utils | Pinned in existing requirements.txt |
| scipy | 1.9.3 | Sparse matrices | Pinned in existing requirements.txt |
| statsmodels | 0.14.0 | Factor neutralization | Pinned in existing requirements.txt |
| matplotlib | 3.7.1 | Plotting | Pinned in existing requirements.txt |
| networkx | 3.2.1 | Graph construction | Pinned in existing requirements.txt |
| tqdm | 4.62.3 | Progress bars | Pinned in existing requirements.txt |
| configparser | stdlib | `.conf` parsing | Already used throughout codebase |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| python-dotenv | optional | `.env` file loading | Only if env var approach chosen over `.conf` for base data dir |

**Recommendation:** Do not add `python-dotenv`. The project already uses `configparser` and `args`. Thread new keys through the existing `.conf` pattern — no new infrastructure needed.

### Updated `requirements.txt` additions

```bash
# ADD these three lines to requirements.txt:
torch==2.8.0          # was commented out
PyWavelets==1.6.0     # missing transitive dep of pytorch-wavelets
tensorboard           # missing dep of torch.utils.tensorboard (pin after checking: latest compatible with torch 2.8)
```

**torch install note:** The existing requirements.txt comment references `torch==2.0.1+cu117` (CUDA 11.7, Linux). The local venv has `torch==2.8.0` (CPU/MPS, macOS). The correct approach is:
- Pin `torch==2.8.0` as the CPU/MPS-compatible version for local development
- Add a comment explaining how to substitute `torch==2.8.0+cu118` or later for GPU training
- Do NOT rely on a CUDA build for the `requirements.txt` that must work on any machine

---

## Architecture Patterns

### Config-driven path pattern (already established, must be completed)

The project already uses `configparser` + `argparse` throughout `MultiTask_Stockformer_train.py`. Every path in `config/Multitask_Stock.conf` under `[file]` is exposed via `args.*`. The fix for INFRA-01 is to extend this existing pattern to cover the currently bypassed paths.

**What:** Add missing path keys to `.conf` `[file]` section; add corresponding `parser.add_argument()` calls; replace hardcoded strings with `args.*` references.

**When to use:** Every file that currently contains `/root/autodl-tmp/` as a literal string.

### Pattern 1: Config file path extension

```python
# In config/Multitask_Stock.conf [file] section — ADD:
alpha_360_dir = ./data/Stock_CN_2021-06-04_2024-01-30/Alpha_360_2021-06-04_2024-01-30
output_dir = ./output/Multitask_output_2021-06-04_2024-01-30
tensorboard_dir = ./runs/Multitask_Stockformer/Stock_CN_2021-06-04_2024-01-30

# In MultiTask_Stockformer_train.py — ADD these argparse entries:
parser.add_argument('--alpha_360_dir', default=config['file']['alpha_360_dir'])
parser.add_argument('--output_dir', default=config['file']['output_dir'])
parser.add_argument('--tensorboard_dir', default=config['file']['tensorboard_dir'])
```

### Pattern 2: Replace hardcoded strings with args references

```python
# BEFORE (MultiTask_Stockformer_train.py line 88):
tensorboard_folder = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/runs/Multitask_Stockformer/Stock_CN_2021-06-04_2024-01-30'

# AFTER:
tensorboard_folder = args.tensorboard_dir
```

```python
# BEFORE (lib/Multitask_Stockformer_utils.py line 112):
path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30/Alpha_360_2021-06-04_2024-01-30'

# AFTER:
path = args.alpha_360_dir
```

```python
# BEFORE (MultiTask_Stockformer_train.py lines 241-246):
save_to_csv('/root/autodl-tmp/.../output/Multitask_output_.../classification/classification_pred_last_step.csv', ...)

# AFTER:
save_to_csv(os.path.join(args.output_dir, 'classification', 'classification_pred_last_step.csv'), ...)
# (directory creation must be added before the save call)
```

### Pattern 3: Data processing scripts — CLI argument approach

`data_Interception.py` and `Stockformer_data_preprocessing_script.py` are standalone scripts (not imported by the training loop). They currently hardcode paths at module level. The fix is to convert their hardcoded variables to CLI `argparse` arguments with sensible defaults:

```python
# Stockformer_data_preprocessing_script.py — BEFORE:
directory = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30'
sys.path.append('/root/autodl-tmp/Stockformer/Stockformer_run/GraphEmbedding')

# AFTER:
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./data/Stock_CN_2021-06-04_2024-01-30')
parser.add_argument('--ge_path', default=None, help='Path to GraphEmbedding library if not pip-installed')
args = parser.parse_args()
directory = args.data_dir
if args.ge_path:
    sys.path.append(args.ge_path)
```

### Anti-Patterns to Avoid

- **Hardcoded base prefix with string concatenation:** Do not replace `/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/` with a new hardcoded base. All paths must originate from `.conf` or CLI args.
- **Environment variables as primary mechanism:** The existing codebase uses `.conf` files. Adding `os.environ.get()` calls creates a second configuration system without advantage.
- **Relative paths in data processing scripts without a clear CWD contract:** Scripts in `data_processing_script/` are run from the project root. Use `./data/...` relative paths and document the required CWD in `SETUP.md`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config management | Custom config parser | `configparser` (already used) | stdlib, already wired in |
| Path joining | String concatenation | `os.path.join()` | Handles OS-specific separators, already used in codebase |
| Directory creation | Custom mkdir logic | `os.makedirs(path, exist_ok=True)` | Already used in the codebase for log/model dirs |
| Package version pinning | Custom lock format | `requirements.txt` with `==` versions | pip standard; no new tooling needed |

---

## Common Pitfalls

### Pitfall 1: Forgetting the `alpha_360_dir` path inside `StockDataset`

**What goes wrong:** `lib/Multitask_Stockformer_utils.py` line 112 hardcodes the Alpha_360 bonus data directory inside the `StockDataset.__init__` constructor. This path is NOT in the config and NOT in `args`. Even after fixing all paths in `MultiTask_Stockformer_train.py`, training still fails because `StockDataset` ignores `args.alpha_360_dir`.

**Why it happens:** `StockDataset.__init__` receives `args` as a parameter, but the original code never added `alpha_360_dir` to the config, so it was just hardcoded inside the class.

**How to avoid:** After adding `alpha_360_dir` to `.conf` and `argparse`, verify `StockDataset.__init__` reads `args.alpha_360_dir` instead of the literal string. This is the single most likely path fix to be missed.

**Warning signs:** `FileNotFoundError: [Errno 2] No such file or directory: '/root/autodl-tmp/...'` raised inside `StockDataset.__init__` at line ~113 even when all other paths succeed.

### Pitfall 2: `PyWavelets` missing from `requirements.txt`

**What goes wrong:** `pip install -r requirements.txt` succeeds, but `python MultiTask_Stockformer_train.py --config ...` crashes immediately with `ModuleNotFoundError: No module named 'pywt'` — before the model even initializes.

**Why it happens:** `pytorch-wavelets` imports `pywt` at module load time. `pywt` is provided by the `PyWavelets` package, which is a transitive dependency not declared in `requirements.txt`.

**How to avoid:** Add `PyWavelets==1.6.0` to `requirements.txt`. Verified: `PyWavelets 1.6.0` installs successfully and resolves the import error.

### Pitfall 3: `tensorboard` missing from `requirements.txt`

**What goes wrong:** Same symptom as Pitfall 2 but for `torch.utils.tensorboard`: `ModuleNotFoundError: No module named 'tensorboard'`.

**Why it happens:** `torch.utils.tensorboard` requires the standalone `tensorboard` package, which is not bundled with `torch`.

**How to avoid:** Add `tensorboard` to `requirements.txt`. Pin after checking current version compatible with `torch==2.8.0`.

### Pitfall 4: `applymap` deprecation crash in `results_data_processing.py`

**What goes wrong:** `results_data_processing.py` line 15 calls `data.astype(str).applymap(...)`. With `pandas==2.3.3` (installed), `applymap` was removed in pandas 2.1. This raises `AttributeError: 'DataFrame' object has no attribute 'applymap'`.

**Why it happens:** `requirements.txt` pins pandas at `1.x`-era assumption but the venv has pandas 2.3.3.

**How to avoid:** Replace `applymap` with `map` in `results_data_processing.py`. This is a one-line fix. Documented in STATE.md as a known blocker.

### Pitfall 5: Output directory not created before `save_to_csv`

**What goes wrong:** When `output_dir` is moved to a config value pointing to a new location, `save_to_csv` will fail with `FileNotFoundError` if the classification/regression subdirectories do not exist.

**Why it happens:** The training script creates `log/` and `cpt/` directories automatically but assumes the output directory pre-exists (it was hardcoded to an existing server path).

**How to avoid:** Add `os.makedirs(os.path.join(args.output_dir, 'classification'), exist_ok=True)` and `os.makedirs(os.path.join(args.output_dir, 'regression'), exist_ok=True)` in `test_res()` before the `save_to_csv` calls.

### Pitfall 6: `torch` commented out means no core ML after clean install

**What goes wrong:** A developer clones the repo, runs `pip install -r requirements.txt`, and then gets `ModuleNotFoundError: No module named 'torch'` — the most fundamental dependency.

**Why it happens:** The original requirements.txt has `# torch==2.0.1+cu117` (commented out) because the CUDA build was platform-specific.

**How to avoid:** Uncomment and update the torch line with the CPU-compatible build. Add a comment for the GPU alternative.

---

## Code Examples

Verified patterns from codebase inspection:

### Complete hardcoded path inventory (all occurrences)

```
File: MultiTask_Stockformer_train.py
  Line 88:  tensorboard_folder = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/runs/Multitask_Stockformer/Stock_CN_2021-06-04_2024-01-30'
  Line 241: save_to_csv('/root/autodl-tmp/.../output/Multitask_output_2021-06-04_2024-01-30/classification/classification_pred_last_step.csv', ...)
  Line 242: save_to_csv('/root/autodl-tmp/.../output/Multitask_output_2021-06-04_2024-01-30/classification/classification_label_last_step.csv', ...)
  Line 245: save_to_csv('/root/autodl-tmp/.../output/Multitask_output_2021-06-04_2024-01-30/regression/regression_pred_last_step.csv', ...)
  Line 246: save_to_csv('/root/autodl-tmp/.../output/Multitask_output_2021-06-04_2024-01-30/regression/regression_label_last_step.csv', ...)

File: lib/Multitask_Stockformer_utils.py
  Line 112: path = '/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30/Alpha_360_2021-06-04_2024-01-30'
  (Line 111 is commented-out predecessor: '/root/.../Alpha_360_2021-02-01_2023-12-29')

File: data_processing_script/stockformer_input_data_processing/results_data_processing.py
  Line 35:  detail_data = pd.read_csv('/root/autodl-tmp/.../data/Stock_CN_2021-06-04_2024-01-30/label.csv', ...)
  Lines 45-46: folder_paths dict with '/root/autodl-tmp/.../output/Multitask_output_2021-06-04_2024-01-30/regression/'
               and '/root/autodl-tmp/.../output/Multitask_output_2021-06-04_2024-01-30/classification/'

File: data_processing_script/stockformer_input_data_processing/data_Interception.py
  Lines 26-30: label_source_path, alpha_source_dir, target_base_dir all hardcoded to /root/autodl-tmp/...

File: data_processing_script/stockformer_input_data_processing/Stockformer_data_preprocessing_script.py
  Line 8:   directory = '/root/autodl-tmp/.../data/Stock_CN_2021-06-04_2024-01-30'
  Line 53:  sys.path.append('/root/autodl-tmp/Stockformer/Stockformer_run/GraphEmbedding')
```

### Updated `config/Multitask_Stock.conf` `[file]` section

```ini
[file]
traffic = ./data/Stock_CN_2021-06-04_2024-01-30/flow.npz
indicator = ./data/Stock_CN_2021-06-04_2024-01-30/trend_indicator.npz
adj = ./data/Stock_CN_2021-06-04_2024-01-30/corr_adj.npy
adjgat = ./data/Stock_CN_2021-06-04_2024-01-30/128_corr_struc2vec_adjgat.npy
model = ./cpt/STOCK/saved_model_Multitask_2021-06-04_2024-01-30
log = ./log/STOCK/log_Multitask_2021-06-04_2024-01-30
alpha_360_dir = ./data/Stock_CN_2021-06-04_2024-01-30/Alpha_360_2021-06-04_2024-01-30
output_dir = ./output/Multitask_output_2021-06-04_2024-01-30
tensorboard_dir = ./runs/Multitask_Stockformer/Stock_CN_2021-06-04_2024-01-30
```

### Updated `requirements.txt`

```text
# Platform-independent requirements for tfm-stockformer
# Run from project root with: pip install -r requirements.txt
# For GPU training substitute: torch==2.8.0+cu118 (Linux, CUDA 11.8)

torch==2.8.0
PyWavelets==1.6.0
tensorboard
scikit-learn==1.1.2
numpy==1.24.4
scipy==1.9.3
matplotlib==3.7.1
tqdm==4.62.3
statsmodels==0.14.0
pytorch-wavelets==1.3.0
networkx==3.2.1
pandas>=2.0,<3.0
```

**Note on pandas:** The requirements.txt pinned `pandas==1.x` but the venv has `2.3.3`. Rather than pin to a version that then breaks on `applymap`, use `pandas>=2.0,<3.0` after fixing the `applymap` bug. Alternatively pin to `pandas==2.3.3` to match the venv exactly.

### Smoke test (INFRA-03 — no training data required)

```bash
# From project root with venv active:
python3 - <<'EOF'
import configparser, argparse, sys

# Test 1: Config loads
config = configparser.ConfigParser()
config.read('config/Multitask_Stock.conf')
assert 'file' in config, "Config missing [file] section"
assert 'alpha_360_dir' in config['file'], "Config missing alpha_360_dir — INFRA-01 incomplete"
print("PASS: config loads and has all required keys")

# Test 2: Core imports work (validates INFRA-02)
import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from torch.utils.tensorboard import SummaryWriter
import numpy as np, pandas as pd
print(f"PASS: imports OK — torch {torch.__version__}, pandas {pd.__version__}")

# Test 3: Model instantiates (no data needed)
sys.path.insert(0, '.')
from Stockformermodel.Multitask_Stockformer_models import Stockformer
model = Stockformer(L=2, h=1, d=128, s=1, w='sym2', j=1, num_nodes=300, num_classes=2)
print("PASS: Stockformer instantiates successfully")

print("\nAll smoke tests passed. Environment is ready.")
EOF
```

### `.gitignore` content

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
venv/
.venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# ML artifacts (generated, do not commit large binaries)
cpt/
runs/
log/
output/

# Data (large; sourced externally)
data/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/settings.json
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `applymap()` (pandas 1.x) | `map()` (pandas 2.x) | pandas 2.0 (Apr 2023), removed in 2.1 | One-line fix in `results_data_processing.py` |
| `torch==2.0.1+cu117` (CUDA-only) | `torch==2.8.0` (CPU/MPS local) | Venv already upgraded | Must un-comment and update in requirements.txt |
| `pywt` bundled assumption | Explicit `PyWavelets` package | pytorch-wavelets 1.3.0 has always required it | Must add to requirements.txt |

**Deprecated/outdated:**
- `requirements.txt` platform comment `# platform: linux-64`: Remove; requirements must be platform-agnostic
- `.ipynb_checkpoints/` committed: Remove from version control; add to `.gitignore`
- `Stockformer_raw` class: Dead code, noted but out of scope for Phase 1

---

## Open Questions

1. **`ge` (GraphEmbedding/Struc2Vec) package for `Stockformer_data_preprocessing_script.py`**
   - What we know: `Stockformer_data_preprocessing_script.py` imports `ge` via `sys.path.append` to the remote server path. The `ge` library is not in `requirements.txt` and cannot be pip-installed under that name.
   - What's unclear: The actual package name on PyPI (candidates: `graphembedding`, `ge`, or a GitHub-only repo). This script is NOT part of Phase 1 or 2 scope as written — it regenerates graph embeddings for a new universe, which is Phase 2 work.
   - Recommendation: For Phase 1, convert the hardcoded `sys.path.append` to a CLI argument (`--ge_path`) so the script can be called portably. Resolve the actual pip package name during Phase 2 when the script will actually be run.

2. **`pandas` version pinning strategy**
   - What we know: `venv` has pandas 2.3.3; `requirements.txt` has no pandas line; `applymap` is broken in pandas 2.x.
   - What's unclear: Whether to pin to `pandas==2.3.3` (match venv exactly) or use `pandas>=2.0,<3.0` (looser).
   - Recommendation: Fix the `applymap` bug first, then pin `pandas==2.3.3` to match the tested venv exactly. Looser pins risk unexpected breakage in later Python environments.

3. **`tensorboard` version to pin**
   - What we know: `tensorboard` is absent from both venv and requirements.txt; `torch.utils.tensorboard` requires it.
   - What's unclear: The precise version compatible with `torch==2.8.0`.
   - Recommendation: Pin as `tensorboard>=2.14,<3` (torch 2.x series works with tensorboard 2.x). Verify during task execution by checking PyPI for the latest 2.x release.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (not yet installed; Wave 0 gap) |
| Config file | none — see Wave 0 |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| INFRA-01 | Running training script without `/root/autodl-tmp/` raises no path error | smoke | `python3 scripts/smoke_test.py` | No — Wave 0 |
| INFRA-02 | `pip install -r requirements.txt` completes without conflicts | integration | `pip install -r requirements.txt --dry-run` | No — Wave 0 |
| INFRA-03 | Smoke test (import model, load config) passes within 30 minutes of cloning | smoke | `python3 scripts/smoke_test.py` | No — Wave 0 |

**Note on INFRA-01:** The success criterion is "does not raise a path error on a machine without `/root/autodl-tmp/`". The smoke test achieves this by running with config-relative paths on the current machine — which has no such directory. If the smoke test passes, INFRA-01 is verified.

### Sampling Rate

- **Per task commit:** `python3 scripts/smoke_test.py`
- **Per wave merge:** `pytest tests/ -v` (once test infrastructure exists from Phase 7)
- **Phase gate:** Smoke test green before marking Phase 1 complete

### Wave 0 Gaps

- [ ] `scripts/smoke_test.py` — covers INFRA-01 and INFRA-03 (model instantiation + config key check)
- [ ] `tests/` directory — does not exist; no pytest infrastructure yet
- [ ] Framework install: `pip install pytest` — add to requirements.txt or document as dev dependency

---

## Sources

### Primary (HIGH confidence)

- Direct code inspection: `MultiTask_Stockformer_train.py` lines 88, 241-246 — hardcoded path inventory
- Direct code inspection: `lib/Multitask_Stockformer_utils.py` line 112 — `StockDataset` bypassed config
- Direct code inspection: `data_processing_script/stockformer_input_data_processing/results_data_processing.py` lines 35, 45-46 — hardcoded paths
- Direct code inspection: `data_processing_script/stockformer_input_data_processing/data_Interception.py` lines 26-30 — hardcoded paths
- Direct code inspection: `data_processing_script/stockformer_input_data_processing/Stockformer_data_preprocessing_script.py` lines 8, 53 — hardcoded paths
- Live venv test: `from pytorch_wavelets import DWT1DForward` → `ModuleNotFoundError: No module named 'pywt'` — confirmed PyWavelets missing
- Live venv test: `from torch.utils.tensorboard import SummaryWriter` → `ModuleNotFoundError: No module named 'tensorboard'` — confirmed tensorboard missing
- Live venv test: After `pip install PyWavelets`, DWT forward pass with shape `[4, 1, 20]` → xl shape `[4, 1, 11]` confirms pytorch-wavelets functional
- `.planning/codebase/CONCERNS.md` — full tech debt catalogue and bug list
- `.planning/codebase/STACK.md` — package versions in venv

### Secondary (MEDIUM confidence)

- `pandas` changelog: `applymap` deprecated 2.0, `map` replacement — consistent with observed `AttributeError` in pandas 2.3.3 venv
- `pytorch-wavelets` GitHub: last release 2021, requires `pywt` — consistent with observed import failure

### Tertiary (LOW confidence)

- `tensorboard` version compatibility with `torch==2.8.0`: not verified against PyPI; recommended `>=2.14` based on torch 2.x release notes pattern

---

## Metadata

**Confidence breakdown:**
- Hardcoded path locations: HIGH — inspected every file, exact line numbers recorded
- Requirements gaps: HIGH — live venv testing confirmed `pywt` and `tensorboard` missing
- Fix pattern (config extension): HIGH — existing `configparser`+`argparse` pattern verified in training script
- `tensorboard` pin version: LOW — transitive; recommend checking PyPI at task execution time
- `pandas` version: MEDIUM — `applymap` bug confirmed by codebase analysis; pin strategy is a judgment call

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable tooling; requirements only change if new packages added)
