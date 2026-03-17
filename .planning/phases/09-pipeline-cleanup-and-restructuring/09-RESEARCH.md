# Phase 9: Pipeline Cleanup and Restructuring - Research

**Researched:** 2026-03-17
**Domain:** Python project restructuring, CLI orchestration, documentation
**Confidence:** HIGH

## Summary

Phase 9 is a purely structural cleanup phase with zero ML changes. All work falls into four
tightly defined buckets: (1) move `data_processing_script/sp500_pipeline/` to `scripts/sp500_pipeline/`
and update the one path reference in `build_pipeline.py`; (2) integrate `build_alpha360.py` as step 5
in `build_pipeline.py` with a `--config` interface; (3) add local-file fallback to `download_ohlcv.py`'s
`get_sp500_tickers()`; (4) rewrite the README `## File Description` section into a Quick Start
reproduction guide and delete all references to the Chinese-market legacy scripts.

The legacy `data_processing_script/` tree that remains after the move (three subdirectories:
`volume_and_price_factor_construction/`, `stockformer_input_data_processing/`, and the now-empty
`data_processing_script/` root) is deleted entirely. No code in any currently-active script imports
from these directories.

**Primary recommendation:** Execute all four buckets as independent, sequentially ordered tasks.
Each task has a single clear deliverable and can be verified by running the test suite plus a
`--help` smoke-check on the CLI.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Legacy Code Removal**
- Delete the entire `data_processing_script/` tree — all directories and files
- Before deleting, move `data_processing_script/sp500_pipeline/` to `scripts/sp500_pipeline/`
  and update `build_pipeline.py`'s `PIPELINE_DIR` to point to the new location
- `results_data_processing.py` (inside `stockformer_input_data_processing/`) is deleted —
  its functionality is superseded by `scripts/run_backtest.py` and `scripts/compute_ic.py`
- Directories being deleted:
  - `data_processing_script/volume_and_price_factor_construction/` (Qlib TA notebooks — obsolete)
  - `data_processing_script/stockformer_input_data_processing/` (original Chinese-market scripts)
  - `data_processing_script/sp500_pipeline/` (moved to `scripts/sp500_pipeline/` first)
  - `data_processing_script/` itself (now empty after moves)

**Pipeline Integration**
- Integrate `build_alpha360.py` as step 5 in `build_pipeline.py` so one command runs the full
  data preparation pipeline: `download → normalize → serialize → graph → alpha360`
- `build_pipeline.py` is updated to accept `--config config/Multitask_Stock_SP500.conf` as the
  primary interface, deriving `data_dir` and `alpha_360_dir` from the config
- The existing `--data_dir`, `--start`, `--end`, `--force` flags remain for backward compatibility
  when running without a config file
- Step 5 sentinel: check that `features/` directory contains exactly 360 CSV files
- Invocation after this phase: `python scripts/build_pipeline.py --config config/Multitask_Stock_SP500.conf`

**Ticker Sourcing**
- `download_ohlcv.py` checks for an existing `tickers.txt` in `data_dir` **before** fetching
  from Wikipedia
- If `tickers.txt` exists: read tickers from file, skip Wikipedia scrape entirely
- If `tickers.txt` does not exist: fetch from Wikipedia as before, then save to file
- Rationale: avoids network dependency and Wikipedia format fragility on re-runs;
  first run still auto-populates the file for future runs

**End-to-End Documentation**
- Update README.md (existing file) with a new "Quick Start" or "Pipeline Reproduction" section
  containing the full command sequence from fresh clone to backtest output:
  1. Install dependencies
  2. `python scripts/build_pipeline.py --config config/Multitask_Stock_SP500.conf`
  3. `python MultiTask_Stockformer_train.py --config config/Multitask_Stock_SP500.conf`
  4. `python scripts/run_inference.py --config config/Multitask_Stock_SP500.conf`
  5. `python scripts/compute_ic.py`
  6. `python scripts/run_backtest.py`
  7. `streamlit run app.py`
- README should note that step 3 (training) requires GPU for reasonable runtime (Kaggle recommended);
  all other steps run on CPU
- Remove references to `data_processing_script/` from README (currently describes Chinese-market
  legacy scripts)

### Claude's Discretion
- Exact README section structure and wording
- Whether to add `--help` output snippets to README
- Internal refactoring within `build_pipeline.py` (how config parsing is implemented)
- Whether to add `__init__.py` to `scripts/sp500_pipeline/` after the move

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| argparse | stdlib | CLI argument parsing | Already used in all pipeline scripts |
| configparser | stdlib | `.conf` file parsing | Already used in `build_alpha360.py` and training script |
| os / pathlib | stdlib | Path manipulation | Used throughout project |
| shutil | stdlib | Directory copy/delete for legacy removal | stdlib, no dependencies |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| subprocess | stdlib | Subprocess invocation for steps 1-4 | Already used in `build_pipeline.py` for steps 1-4 |

**No new dependencies are introduced in this phase.** All work uses stdlib and existing project
imports.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Direct function import for step 5 | subprocess call | Function import avoids spawning a new Python process and gives cleaner error propagation; CONTEXT.md explicitly prefers direct import |
| `--config` only interface | Keep `--data_dir` only | `--config` is the unified interface; `--data_dir`/`--start`/`--end` kept for backward compatibility |

## Architecture Patterns

### Existing Project Structure (relevant paths)

```
scripts/
├── build_pipeline.py      # orchestrator — PIPELINE_DIR and STEPS modified here
├── build_alpha360.py      # step 5 — imported directly, not subprocess
├── run_inference.py
├── run_backtest.py
├── compute_ic.py
└── smoke_test.py

data_processing_script/    # DELETED in this phase (after move)
├── sp500_pipeline/        # MOVED to scripts/sp500_pipeline/
│   ├── download_ohlcv.py  # ticker fallback logic added here
│   ├── normalize_split.py
│   ├── serialize_arrays.py
│   └── graph_embedding.py
├── stockformer_input_data_processing/  # DELETED (Chinese-market scripts)
└── volume_and_price_factor_construction/  # DELETED (Qlib notebooks)

config/
└── Multitask_Stock_SP500.conf  # provides traffic, alpha_360_dir
```

**After phase 9:**

```
scripts/
├── build_pipeline.py      # updated: PIPELINE_DIR, STEPS, --config flag
├── build_alpha360.py      # unchanged
├── sp500_pipeline/        # moved from data_processing_script/
│   ├── download_ohlcv.py  # updated: local tickers.txt fallback
│   ├── normalize_split.py
│   ├── serialize_arrays.py
│   └── graph_embedding.py
├── run_inference.py
├── run_backtest.py
├── compute_ic.py
└── smoke_test.py
```

### Pattern 1: Sentinel-Based Idempotent Step

**What:** Each pipeline step checks for an output file before running; if it exists the step is
skipped. The existing STEPS list uses `(script_name, sentinel)` tuples.

**When to use:** Step 5 follows the same tuple pattern. The sentinel for Alpha360 is a directory
count: `features/` contains exactly 360 CSV files.

**Current STEPS list (lines 44-49 of `scripts/build_pipeline.py`):**
```python
STEPS = [
    ("download_ohlcv.py",   "tickers.txt"),
    ("normalize_split.py",  "split_indices.json"),
    ("serialize_arrays.py", "flow.npz"),
    ("graph_embedding.py",  "128_corr_struc2vec_adjgat.npy"),
]
```

**Step 5 sentinel approach** — the sentinel check requires special handling because it is a
directory count, not a single file. The `_sentinel_exists` helper checks `os.path.exists(sentinel)`,
which does not work for "360 CSV files in features/". The step 5 runner must use a custom sentinel
check rather than the generic helper:

```python
def _alpha360_done(features_dir: str) -> bool:
    if not os.path.isdir(features_dir):
        return False
    csv_count = sum(1 for f in os.listdir(features_dir) if f.endswith(".csv"))
    return csv_count == 360
```

### Pattern 2: Config-Driven Path Derivation

**What:** `build_alpha360.main()` already parses the `.conf` file using `configparser` and derives
`data_dir` from `cfg["file"]["alpha_360_dir"]` parent. The same derivation must be replicated in
`build_pipeline.py` when `--config` is passed.

**Key config keys used:**
- `cfg["file"]["traffic"]` — path is `data_dir/flow.npz`; `data_dir` = parent of `traffic`'s
  parent folder (i.e., `Path(traffic).parent` directly gives `data_dir` since traffic =
  `./data/Stock_SP500_2018-01-01_2024-01-01/flow.npz`)
- `cfg["file"]["alpha_360_dir"]` — already points to `data_dir/features`

**Derivation in `build_pipeline.py`:**
```python
# Source: config/Multitask_Stock_SP500.conf structure
import configparser
from pathlib import Path

cfg = configparser.ConfigParser()
cfg.read(args.config)
traffic_path = Path(cfg["file"]["traffic"]).resolve()
data_dir = str(traffic_path.parent)            # e.g., .../Stock_SP500_2018-01-01_2024-01-01
alpha_360_dir = str(Path(cfg["file"]["alpha_360_dir"]).resolve())
# start/end can be inferred from data_dir name or kept as defaults
```

### Pattern 3: Direct Function Import for Step 5

**What:** Instead of `subprocess.run([sys.executable, "build_alpha360.py", ...])`, step 5 imports
and calls `build_alpha360.main()` directly.

**Why:** Cleaner error propagation (exceptions surface immediately), no process spawn overhead,
consistent with how tests already call `build_alpha360.main(config_path, data_dir)`.

**Implementation:**
```python
# In build_pipeline.py — step 5 runner
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_alpha360 import main as build_alpha360_main

def run_alpha360_step(config_path: str, features_dir: str, force: bool) -> None:
    if not force and _alpha360_done(features_dir):
        print(f"[SKIP] build_alpha360  (sentinel: 360 CSVs already in features/)")
        return
    print("\n[RUN]  build_alpha360")
    build_alpha360_main(config_path=config_path)
    print(f"[DONE] build_alpha360")
```

### Pattern 4: Ticker Local-File Fallback

**What:** Modify `get_sp500_tickers()` signature to accept optional `data_dir` parameter (or
modify `main()` to check before calling the function). The CONTEXT.md shows the check as a one-liner
at the top of the function.

**Current `get_sp500_tickers()` does not accept `data_dir`.** The cleaner approach is to modify
`main()` to branch before calling `get_sp500_tickers()`:

```python
# In download_ohlcv.py main()
tickers_txt_path = os.path.join(args.data_dir, "tickers.txt")
if os.path.exists(tickers_txt_path):
    logger.info("Reading tickers from existing %s (skipping Wikipedia scrape)", tickers_txt_path)
    with open(tickers_txt_path) as f:
        tickers = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d tickers from file.", len(tickers))
else:
    logger.info("Fetching S&P 500 ticker list from Wikipedia...")
    tickers = get_sp500_tickers()
    logger.info("Found %d tickers.", len(tickers))
```

Note: `tickers.txt` is written at the end of `main()` in the same file (lines 253-256). On first
run the file does not exist so Wikipedia is used and the file is created. On subsequent runs the
file exists and is read directly.

### Anti-Patterns to Avoid

- **Changing PIPELINE_DIR to an absolute path at module level:** Keep it relative to `_SCRIPT_DIR`
  as currently done; after the move the relative path changes from `"../data_processing_script/sp500_pipeline"` to `"sp500_pipeline"` (same directory as `build_pipeline.py`)
- **Removing `--data_dir`, `--start`, `--end` flags:** These must remain for backward compatibility
  per the locked decisions
- **Using `shutil.rmtree` before verifying the move completed:** Move first, verify new location,
  then delete old tree
- **Importing `build_alpha360` without adjusting `sys.path`:** The import must work from project
  root (`python scripts/build_pipeline.py`); since both files are in `scripts/`, a `sys.path.insert`
  pointing to `_SCRIPT_DIR` is sufficient

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Config file parsing | Custom `.conf` reader | `configparser.ConfigParser()` | Already used in project; handles sections, keys, defaults |
| Directory tree deletion | Custom recursive delete | `shutil.rmtree()` | Handles non-empty directories atomically |
| Directory copy (backup) | Custom recursive copy | `shutil.copytree()` | Already used in `build_alpha360.py` |
| Step counting CSV files | Custom glob | `os.listdir()` + list comprehension | Already used in `build_alpha360.py` validation block |

**Key insight:** This phase reuses every stdlib tool already present in the codebase. No new
patterns are introduced.

## Common Pitfalls

### Pitfall 1: PIPELINE_DIR Path After Move

**What goes wrong:** After moving `sp500_pipeline/` into `scripts/`, the PIPELINE_DIR calculation
in `build_pipeline.py` still points to `../data_processing_script/sp500_pipeline` and raises
`FileNotFoundError` when any step runs.

**Why it happens:** `PIPELINE_DIR` is computed at module load time (lines 37-40 of
`build_pipeline.py`). The current value is:
```python
PIPELINE_DIR = os.path.normpath(
    os.path.join(_SCRIPT_DIR, "..", "data_processing_script", "sp500_pipeline")
)
```
**How to avoid:** After move, update to:
```python
PIPELINE_DIR = os.path.join(_SCRIPT_DIR, "sp500_pipeline")
```
**Warning signs:** `python scripts/build_pipeline.py --data_dir ...` raises `FileNotFoundError`
or `subprocess.CalledProcessError` on step 1.

### Pitfall 2: Step 5 Sentinel Type Mismatch

**What goes wrong:** The generic `_sentinel_exists(data_dir, sentinel)` uses `os.path.exists()`.
If step 5 is added to STEPS with a string sentinel like `"features"`, it will check for a file
named `"features"` rather than counting 360 CSVs inside the directory.

**Why it happens:** The sentinel contract in the existing code is file existence, not directory
content count.

**How to avoid:** Handle step 5 outside the `for` loop over STEPS, or special-case it with a
custom check function. Do not put step 5 in the STEPS list with a simple string sentinel.

**Warning signs:** Pipeline always shows `[SKIP] build_alpha360` even when `features/` is empty
or contains fewer than 360 files.

### Pitfall 3: Config Derivation When --config and --data_dir Are Both Provided

**What goes wrong:** If a user passes both `--config` and `--data_dir`, the two sources may
conflict (config says one path, `--data_dir` says another).

**Why it happens:** The locked decision says `--data_dir` is kept for backward compatibility but
`--config` is the primary interface. No precedence rule is explicitly specified.

**How to avoid:** When `--config` is provided, derive `data_dir` from config and ignore
`--data_dir` (or emit a warning that `--config` takes precedence). Document this in `--help`.

**Warning signs:** Data written to wrong directory; pipeline completes but model fails to find
its input files.

### Pitfall 4: Ticker Fallback When tickers.txt Was Written By a Previous Partial Run

**What goes wrong:** A previous run of `download_ohlcv.py` failed after writing a partial
`tickers.txt` (fewer tickers than expected). On re-run, the fallback reads the partial file,
downloads OHLCV for fewer tickers, and subsequent steps silently operate on a smaller universe.

**Why it happens:** `tickers.txt` is written at the END of `main()` (line 255), after all
Parquets are saved. A crash before that point means no `tickers.txt` exists, so this pitfall
requires an intermediate partial write to occur — unlikely given the current code, but possible
if the file is pre-placed by the user manually.

**How to avoid:** No code change needed; just document in README that a manually placed
`tickers.txt` is authoritative. The current sentinel logic in `build_pipeline.py` already
skips step 1 if `tickers.txt` exists, which is consistent.

### Pitfall 5: README Reference Removal Incomplete

**What goes wrong:** Old references to `data_processing_script/` remain in the README after
the update, causing confusion for readers after the directory is deleted.

**Why it happens:** The README has two relevant sections: "File Description" (lines 38-96) and
"How to Run" (lines 99-103). Both reference the Chinese-market original structure.

**How to avoid:** Search README.md for `data_processing_script` and `Stockformer_train.py` (the
old training script name) and remove or replace each occurrence.

## Code Examples

Verified patterns from project source code:

### Updated PIPELINE_DIR (after move)
```python
# scripts/build_pipeline.py — after sp500_pipeline/ is moved into scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(_SCRIPT_DIR, "sp500_pipeline")  # was: "../data_processing_script/sp500_pipeline"
```

### Config-Driven data_dir Derivation
```python
# Derived from existing config structure (config/Multitask_Stock_SP500.conf line 28):
#   traffic = ./data/Stock_SP500_2018-01-01_2024-01-01/flow.npz
import configparser
from pathlib import Path

def _data_dir_from_config(config_path: str) -> tuple:
    """Returns (data_dir, alpha_360_dir) derived from config file."""
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    traffic = cfg["file"]["traffic"]
    data_dir = str(Path(traffic).resolve().parent)
    alpha_360_dir = str(Path(cfg["file"]["alpha_360_dir"]).resolve())
    return data_dir, alpha_360_dir
```

### Step 5 Alpha360 Sentinel Check
```python
def _alpha360_done(features_dir: str) -> bool:
    """Return True if features/ contains exactly 360 CSV files."""
    if not os.path.isdir(features_dir):
        return False
    return sum(1 for f in os.listdir(features_dir) if f.endswith(".csv")) == 360
```

### Ticker Local-File Fallback (in download_ohlcv.py main())
```python
# Add at the start of the download block in main(), before the Wikipedia call
tickers_txt_path = os.path.join(args.data_dir, "tickers.txt")
if os.path.exists(tickers_txt_path):
    logger.info("Reading tickers from existing %s (skipping Wikipedia scrape)", tickers_txt_path)
    with open(tickers_txt_path) as f:
        tickers = [line.strip() for line in f if line.strip()]
    logger.info("Loaded %d tickers from file.", len(tickers))
    raw_data = download_ohlcv_batched(tickers, start=args.start, end=args.end)
else:
    logger.info("Fetching S&P 500 ticker list from Wikipedia...")
    tickers = get_sp500_tickers()
    # ... existing flow continues
```

### Legacy Tree Deletion (after move is confirmed)
```python
# Verify new location exists before deleting old
import shutil
assert os.path.isdir("scripts/sp500_pipeline"), "Move must complete before deletion"
shutil.rmtree("data_processing_script")
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Steps 1-4 in `data_processing_script/sp500_pipeline/` | Steps 1-4 in `scripts/sp500_pipeline/` | Phase 9 | All pipeline scripts co-located in `scripts/` |
| Alpha360 built as separate manual command | Alpha360 integrated as step 5 in `build_pipeline.py` | Phase 9 | Single command `build_pipeline.py --config` runs full pipeline |
| Wikipedia scrape on every re-run | Local `tickers.txt` cache checked first | Phase 9 | Re-runs are network-independent |
| README describes Chinese-market original code | README Quick Start describes S&P500 pipeline | Phase 9 | Reproducibility for any fresh clone |

**Deprecated/outdated after this phase:**
- `data_processing_script/volume_and_price_factor_construction/` — Qlib TA notebooks,
  superseded by Phase 8 Alpha360 builder
- `data_processing_script/stockformer_input_data_processing/` — Chinese-market scripts,
  superseded by the full S&P500 pipeline
- README "How to Run" section referencing `python Multitask_Stockformer_models.py` — the
  training script is now `MultiTask_Stockformer_train.py`

## Open Questions

1. **Whether to add `__init__.py` to `scripts/sp500_pipeline/`**
   - What we know: `build_pipeline.py` invokes step scripts via `subprocess.run`, not via import;
     no test currently imports from `scripts/sp500_pipeline/` directly
   - What's unclear: whether future test additions may import from this package
   - Recommendation: Add an empty `__init__.py` — low cost, makes the directory a proper package,
     consistent with `scripts/__init__.py` already present

2. **`--start` and `--end` derivation when `--config` is provided**
   - What we know: The config file does not contain `start` / `end` date fields; they appear in
     the `data_dir` folder name (`Stock_SP500_2018-01-01_2024-01-01`) but parsing folder names
     is fragile
   - What's unclear: Whether `--start` and `--end` should retain their argparse defaults when
     `--config` is used, or be derived from the folder name
   - Recommendation: Keep argparse defaults (`2018-01-01` / `2024-01-01`); a user can override
     with explicit flags. No config-based derivation needed.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing) |
| Config file | none — run from project root |
| Quick run command | `pytest tests/ -x -q` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map

Phase 9 has no new requirement IDs in REQUIREMENTS.md (it is structural, not functional). The
tests relevant to this phase are existing tests that must remain green after the refactor:

| Existing Test | Behavior | Verification |
|---------------|----------|-------------|
| `tests/test_phase1_infra.py` | Path and config tests | Must pass after PIPELINE_DIR change |
| `tests/test_build_alpha360.py` | Alpha360 build contract | Must pass — no changes to build_alpha360.py |
| `tests/` (full suite) | All project tests | Must be all-green after deletions and moves |

**New tests for Phase 9 changes:**
| Behavior | Test Type | Suggested File |
|----------|-----------|----------------|
| `PIPELINE_DIR` resolves to a valid directory | unit | `tests/test_pipeline.py` |
| `build_pipeline.py --config` parses `data_dir` from config correctly | unit | `tests/test_pipeline.py` |
| `_alpha360_done()` returns True only when exactly 360 CSVs present | unit | `tests/test_pipeline.py` |
| `download_ohlcv.py` reads tickers from file when `tickers.txt` exists | unit | `tests/test_download_ohlcv.py` |
| `download_ohlcv.py` calls Wikipedia when `tickers.txt` absent | unit (mock) | `tests/test_download_ohlcv.py` |

### Sampling Rate
- **Per task commit:** `pytest tests/ -x -q`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pipeline.py` — covers PIPELINE_DIR update, config derivation, alpha360 sentinel
- [ ] `tests/test_download_ohlcv.py` — covers ticker local-file fallback logic

*(Existing `tests/conftest.py` provides shared fixtures; no new fixtures required for these unit tests.)*

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `scripts/build_pipeline.py` (lines 37-49) — PIPELINE_DIR and STEPS list
- Direct code inspection of `scripts/build_alpha360.py` (lines 131-145) — `main()` signature and config parsing
- Direct code inspection of `config/Multitask_Stock_SP500.conf` (lines 28-34) — key names and path structure
- Direct code inspection of `data_processing_script/sp500_pipeline/download_ohlcv.py` (lines 201-263) — existing `main()` flow and `tickers.txt` write location
- Direct code inspection of `README.md` (lines 38-103) — sections requiring update

### Secondary (MEDIUM confidence)
- `.planning/phases/09-pipeline-cleanup-and-restructuring/09-CONTEXT.md` — user decisions and code context notes

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all stdlib, all already in use
- Architecture: HIGH — patterns read directly from existing source files
- Pitfalls: HIGH — derived from actual code structure (sentinel type, path computation)
- Validation: HIGH — existing pytest infrastructure confirmed

**Research date:** 2026-03-17
**Valid until:** 2026-04-17 (stable internal refactor — no external dependency changes)
