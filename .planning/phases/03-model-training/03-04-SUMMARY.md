---
phase: 03-model-training
plan: "04"
subsystem: model-training
tags: [stockformer, smoke-test, inference, validation, checkpoint]

# Dependency graph
requires:
  - phase: 03-model-training
    plan: "02"
    provides: "config/Multitask_Stock_SP500.conf — training config with all paths"
  - phase: 03-model-training
    plan: "03"
    provides: "scripts/run_inference.py — standalone inference script"
  - phase: 02-data-pipeline
    provides: "Phase 2 artifacts: flow.npz, trend_indicator.npz, adjgat.npy, features/ CSVs"
provides:
  - "End-to-end validation: 2-epoch smoke test + inference run with real Phase 2 data"
  - "Trained checkpoint at cpt/STOCK/saved_model_Multitask_SP500_2018-2024 (post user action)"
  - "Prediction CSV at output/Multitask_output_SP500_2018-2024/regression/regression_pred_last_step.csv (post user action)"
affects: [04-evaluation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Human-verify checkpoint gate: user runs long-running commands (training, pipeline) and confirms artifacts"

key-files:
  created: []
  modified: []

key-decisions:
  - "Phase 2 pipeline (build_pipeline.py) must be run before any training smoke test — data/Stock_SP500_2018-2024/ did not exist on this machine"
  - "Task 1 verification confirmed Phase 2 data absent; user must run build_pipeline.py before proceeding"

patterns-established:
  - "Verification-first pattern: check all prereqs before attempting any long-running training command"

requirements-completed: []

# Metrics
duration: 1min
completed: 2026-03-12
---

# Phase 3 Plan 04: End-to-End Smoke Test and Human Verification Summary

**Human-verify checkpoint gate for 2-epoch training smoke test and inference run — awaiting Phase 2 pipeline execution and user confirmation**

## Performance

- **Duration:** ~1 min
- **Started:** 2026-03-12T10:44:31Z
- **Completed:** 2026-03-12T10:45:00Z (checkpoint — awaiting human action)
- **Tasks:** 1 of 2 (Task 1 ran; Task 2 is a human-verify checkpoint)
- **Files created:** 0

## Accomplishments

- Ran Task 1 Phase 2 output verification — confirmed `data/Stock_SP500_2018-2024/` is absent on this machine
- Identified that `scripts/build_pipeline.py` must be run before smoke test can proceed
- All supporting scripts and config files are present: `scripts/run_inference.py`, `config/Multitask_Stock_SP500.conf`

## Task Commits

No new code was written in this plan (verification-only tasks). Task 1 produced no file changes.

## Files Created/Modified

None — plan 03-04 is a verification and human-verify gate plan.

## Decisions Made

- Phase 2 pipeline must run on local machine first. The `data/` directory does not exist — `flow.npz`, `trend_indicator.npz`, `corr_adj.npy`, `128_corr_struc2vec_adjgat.npy`, and `features/` are all absent.
- Struc2Vec graph embeddings (via `build_pipeline.py`) require 10-20 minutes; this is a human-action gate.

## Deviations from Plan

None — Task 1 ran as specified. The missing data is the expected pre-condition that the task was designed to detect.

## Issues Encountered

Phase 2 data pipeline has not been run on this machine. The `data/Stock_SP500_2018-2024/` directory does not exist. This is not a bug — it is the expected blocking condition that Task 1 was designed to surface. The user must run the Phase 2 pipeline before proceeding with the smoke test.

## User Setup Required

**Before proceeding with Task 2, run the Phase 2 data pipeline:**

```bash
cd /Users/gonzaloalonsolidon/Desktop/Repos/Cursor/tfm-stockformer
python scripts/build_pipeline.py
```

This will:
- Download S&P500 OHLCV data via yfinance
- Compute TA features (69 columns) and save feature CSVs
- Build `flow.npz`, `trend_indicator.npz`, `corr_adj.npy`
- Run Struc2Vec to produce `128_corr_struc2vec_adjgat.npy`
- Expected time: 10-30 minutes depending on hardware and download speed

After the pipeline completes, re-run Task 1 verification, then proceed with the smoke test steps in Task 2.

## Next Phase Readiness

- Phase 3 is NOT complete — awaiting human verification (Task 2)
- Once user confirms smoke test passed and prediction CSV exists, Phase 3 will be complete
- Phase 4 (Evaluation) requires `output/Multitask_output_SP500_2018-2024/regression/regression_pred_last_step.csv`

---
*Phase: 03-model-training*
*Completed: 2026-03-12 (partial — checkpoint pending)*
