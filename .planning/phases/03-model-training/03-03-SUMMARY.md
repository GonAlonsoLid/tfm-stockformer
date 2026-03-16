---
phase: 03-model-training
plan: "03"
subsystem: model-training
tags: [stockformer, inference, standalone-script, predictions, csv]

# Dependency graph
requires:
  - phase: 03-model-training
    plan: "02"
    provides: "config/Multitask_Stock_SP500.conf — training config with all paths"
  - phase: 02-data-pipeline
    provides: "Phase 2 artifacts: flow.npz, trend_indicator.npz, adjgat.npy, features/ CSVs"
provides:
  - "scripts/run_inference.py — standalone inference script producing prediction CSVs"
  - "MODEL-02 inference tests both passing (xpassed) in pytest"
affects: [04-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-phase argparse: pre-parse for --config, then register INI values as defaults"
    - "Numpy-array inference loop (no DataLoader) — mirrors training script pattern"
    - "Standalone import pattern: sys.path.insert to project root, import only from lib/ and Stockformermodel/"

key-files:
  created:
    - scripts/run_inference.py
  modified: []

key-decisions:
  - "loadGraph() returns adjgat directly (not a tuple) — plan interface docs were outdated; actual lib/graph_utils.py returns only adjgat"
  - "Inference loop uses raw numpy arrays from dataset attributes (XL, XH, indicator_X, TE, Y, bonus_X) rather than DataLoader — consistent with training script pattern"
  - "Two-phase argparse: pre-parser uses parse_known_args() so --config is captured before INI values are available for defaults"
  - "--config is required (not optional) to force explicit config path and prevent silent misconfiguration"
  - "configparser normalises keys to lowercase: config[data][t1] not T1"

patterns-established:
  - "Standalone inference script: no module-level side effects, no log file creation, no SummaryWriter"
  - "Phase 2 artifact validation before model load: clear error messages pointing user to data pipeline"

requirements-completed: [MODEL-02]

# Metrics
duration: 2min
completed: 2026-03-12
---

# Phase 3 Plan 03: Standalone Inference Script Summary

**Standalone `scripts/run_inference.py` loading trained Stockformer checkpoint via two-phase argparse with numpy-array inference loop, producing regression and classification prediction CSVs**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-12T10:40:38Z
- **Completed:** 2026-03-12T10:42:12Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments

- Created `scripts/run_inference.py` as a fully standalone script (277 lines)
- Two-phase argparse: `--config` (required) + `--checkpoint` (optional, overrides config model path)
- No imports from `MultiTask_Stockformer_train.py` — only `lib/` and `Stockformermodel/` used
- Mirrors the training script's numpy-array batch loop (consistent data access pattern)
- Phase 2 artifact validation with actionable error messages before attempting data load
- Produces four output CSVs in `{output_dir}/regression/` and `{output_dir}/classification/`
- Both inference tests (`test_inference_script_exists`, `test_inference_script_args`) xpassed
- Full test suite: 5 xpassed, 1 xfailed — all green

## Task Commits

Each task was committed atomically:

1. **Task 1: Create scripts/run_inference.py** — `e9d1743` (feat)

## Files Created/Modified

- `scripts/run_inference.py` — standalone inference script; two-phase argparse, numpy-array inference loop, saves prediction CSVs

## Decisions Made

- **loadGraph deviation:** `lib/graph_utils.py` returns `adjgat` directly (a single array), not a tuple `(adj, adjgat)` as described in the plan's interface docs. The plan's interface comment was based on an older version of the function. Used the actual return value. This is an auto-fix under Rule 1 (bug in plan docs vs. actual code).
- **Numpy-array loop instead of DataLoader:** The training script accesses dataset attributes (`.XL`, `.XH`, etc.) directly as numpy arrays and slices them per batch — it does NOT use PyTorch DataLoader. The inference script follows the same pattern for consistency and correctness.
- **config key case:** `configparser` normalises INI keys to lowercase, so `T1`/`T2` from the `[data]` section are accessed as `config["data"]["t1"]` and `config["data"]["t2"]`.
- **--config is required:** Made `--config` a required argument (not optional) to prevent silent misconfiguration where no config is loaded. The pre-parser uses `required=True`; error message is clear.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Stale interface docs] loadGraph returns adjgat directly, not (adj, adjgat) tuple**
- **Found during:** Task 1 implementation (reading lib/graph_utils.py)
- **Issue:** Plan interface docs stated `loadGraph(args)` returns `(adj, adjgat)`. Actual implementation returns only `adjgat` (adj loading is commented out in the function body).
- **Fix:** Used `adjgat = loadGraph(args)` (single assignment) matching the actual training script at line 413.
- **Files modified:** scripts/run_inference.py only (plan docs not modified)
- **Commit:** e9d1743 (included in task commit)

**2. [Rule 1 - Stale interface] DataLoader not used by training script**
- **Found during:** Task 1 implementation (reading MultiTask_Stockformer_train.py main block)
- **Issue:** Plan action section referenced `DataLoader(test_dataset, ...)` pattern. The actual training script accesses dataset numpy attributes directly and manually slices them in a batch loop.
- **Fix:** Implemented the same numpy-array batch loop pattern as the training script.
- **Commit:** e9d1743 (included in task commit)

## Issues Encountered

None beyond the two interface-doc deviations above, both auto-fixed.

## User Setup Required

None. Inference script runs once Phase 2 data artifacts and a trained checkpoint exist at the configured paths.

## Next Phase Readiness

- MODEL-02 complete: `scripts/run_inference.py` is ready for use by Phase 4 (Evaluation)
- Phase 4 can read `regression_pred_last_step.csv` and `classification_pred_last_step.csv` from `{output_dir}/`
- Actual inference requires Phase 2 data pipeline to have run and a trained checkpoint to exist

---
*Phase: 03-model-training*
*Completed: 2026-03-12*
