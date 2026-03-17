---
phase: 09-pipeline-cleanup-and-restructuring
plan: "04"
subsystem: documentation
tags: [cleanup, readme, legacy-deletion, documentation]
dependency_graph:
  requires: [09-03]
  provides: [clean-repo, onboarding-docs]
  affects: [README.md]
tech_stack:
  added: []
  patterns: [sentinel-based-idempotency, quick-start-documentation]
key_files:
  created: []
  modified:
    - README.md
  deleted:
    - data_processing_script/ (entire tree)
decisions:
  - "data_processing_script/ deleted in full — sp500_pipeline/ subdir was already moved to scripts/ in 09-02; Chinese-market subdirs were fully obsolete"
  - "README.md Quick Start section uses 7-step numbered sequence matching CONTEXT.md locked command list"
  - "GPU note placed in Step 3 (training only); all other steps documented as CPU-runnable"
metrics:
  duration: "~5 min"
  completed: "2026-03-17"
  tasks_completed: 3
  files_modified: 1
  files_deleted: 1
---

# Phase 09 Plan 04: Legacy Cleanup and README Rewrite Summary

**One-liner:** Deleted `data_processing_script/` legacy tree and rewrote README with Quick Start section listing all 7 reproduction commands and GPU note.

## What Was Built

This plan finalized the Phase 9 cleanup. The legacy `data_processing_script/` directory (containing Chinese-market Qlib notebooks and superseded preprocessing scripts) was deleted entirely. The `README.md` was rewritten to replace the outdated File Description and How to Run sections with a structured File Structure tree and a Quick Start section that guides any reader from fresh clone to Streamlit dashboard in seven steps.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Delete data_processing_script/ tree and verify test suite | 2bdec58 | data_processing_script/ (deleted) |
| 2 | Rewrite README.md with Quick Start section | 01125db | README.md |
| 3 | Human verify README accuracy and legacy deletion | — (checkpoint) | — |

## Verification Results

- `ls data_processing_script/` returns "No such file or directory"
- `grep "data_processing_script" README.md` returns no matches
- `grep "## Quick Start" README.md` matches
- `grep "GPU required" README.md` matches
- `python -m pytest tests/ -x -q` passes (full suite green after deletion)
- Human reviewer approved README accuracy and confirmed --help flags

## Deviations from Plan

None — plan executed exactly as written.

## Decisions Made

1. **data_processing_script/ deleted in full** — The `sp500_pipeline/` subdir was already moved to `scripts/` in Plan 09-02. The remaining subdirectories (`stockformer_input_data_processing/`, `volume_and_price_factor_construction/`) contained only Chinese-market Qlib notebooks and scripts with no relevance to the S&P500 pipeline.

2. **7-step Quick Start sequence** — Commands match the locked sequence from CONTEXT.md exactly: install, build_pipeline.py, train, run_inference.py, compute_ic.py, run_backtest.py, streamlit.

3. **GPU note placement** — Applied only to Step 3 (model training) to avoid misleading users about other steps, which all run on CPU.

## Phase 9 Completion

With Plan 09-04 complete, Phase 9 (Pipeline Cleanup and Restructuring) is fully done:

| Plan | Name | Status |
|------|------|--------|
| 09-01 | Wave 0 stubs and test infrastructure | Complete |
| 09-02 | Move sp500_pipeline and fix ticker fallback | Complete |
| 09-03 | Wire --config into build_pipeline.py and Alpha360 integration | Complete |
| 09-04 | Delete legacy tree and rewrite README | Complete |

The repository is now clean: no Chinese-market legacy code, config-driven paths throughout, canonical script locations under `scripts/`, and onboarding documentation sufficient to reproduce the full pipeline from a fresh clone.

## Self-Check: PASSED
