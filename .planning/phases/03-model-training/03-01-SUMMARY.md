---
phase: 03-model-training
plan: "01"
subsystem: testing

tags: [pytest, xfail, model-training, stockformer, inference]

requires:
  - phase: 02-data-pipeline
    provides: "StockDataset interface contract and NPZ/CSV file-format specifications that tests will eventually validate"

provides:
  - "tests/test_model_training.py with six xfail stubs covering MODEL-01 and MODEL-02"
  - "Wave 0 scaffold: automated feedback available from first task of every Phase 3 plan"

affects: [03-02, 03-03, phase-3-verify]

tech-stack:
  added: []
  patterns:
    - "xfail(strict=False) stubs — all Phase 3 tests start as stubs; plans 03-02/03-03 turn them green"
    - "Imports inside test body — model/script imports are deferred so the file loads without errors when artefacts are absent"

key-files:
  created:
    - tests/test_model_training.py
  modified: []

key-decisions:
  - "All Phase 3 test stubs use xfail(strict=False) — suite passes before implementation lands (consistent with Phase 2 precedent in STATE.md)"
  - "test_stockformer_forward_pass imports Stockformer inside test body to avoid ModuleNotFoundError at collection time"

patterns-established:
  - "Wave 0 pattern: create xfail stubs first, green them in later plans — ensures automated feedback is always available"

requirements-completed: [MODEL-01, MODEL-02]

duration: 2min
completed: 2026-03-11
---

# Phase 3 Plan 01: Model Training Test Scaffold Summary

**Six xfail stubs in tests/test_model_training.py covering config validation, dataset loading, Stockformer forward pass shape, and inference script argument checks for MODEL-01 and MODEL-02**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-11T19:23:58Z
- **Completed:** 2026-03-11T19:25:09Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `tests/test_model_training.py` with all six stubs required by 03-VALIDATION.md
- All stubs use `@pytest.mark.xfail(strict=False)` so `pytest tests/test_model_training.py -x -q` exits 0 immediately
- Function names match 03-VALIDATION.md exactly: `test_config_file_exists`, `test_config_fields_present`, `test_dataset_loads`, `test_stockformer_forward_pass`, `test_inference_script_exists`, `test_inference_script_args`
- Implementation imports deferred inside test bodies — file loads cleanly even when model/script artefacts are absent

## Task Commits

1. **Task 1: Write xfail test stubs for MODEL-01 and MODEL-02** - `c207a11` (test)

## Files Created/Modified

- `tests/test_model_training.py` - Six xfail stubs covering MODEL-01 config/dataset and MODEL-02 forward pass/inference

## Decisions Made

- All stubs use `strict=False` consistent with the Phase 2 precedent already recorded in STATE.md
- `test_stockformer_forward_pass` defers the `from Stockformermodel...` import inside the test body to avoid a collection-time ModuleNotFoundError
- `test_config_fields_present` explicitly checks all 24 required keys across four INI sections ([file], [data], [train], [param]) as specified in the plan

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

The full suite (`pytest tests/ -q`) shows 4 pre-existing failures in `test_phase1_infra.py` for torch/pywt/pytorch_wavelets imports — these failures exist before this plan and are caused by the local environment lacking those packages. They are out of scope for this plan. The plan target `pytest tests/test_model_training.py -x -q` exits 0 with 6 xfailed.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Wave 0 scaffold complete — automated feedback is available from the first task of every Phase 3 plan
- Plan 03-02 can now implement `config/Multitask_Stock_SP500.conf` and turn `test_config_file_exists` and `test_config_fields_present` green
- Plan 03-03 can implement `scripts/run_inference.py` and turn `test_inference_script_exists` and `test_inference_script_args` green
- `test_stockformer_forward_pass` and `test_dataset_loads` require full model/data artefacts on disk

---
*Phase: 03-model-training*
*Completed: 2026-03-11*
