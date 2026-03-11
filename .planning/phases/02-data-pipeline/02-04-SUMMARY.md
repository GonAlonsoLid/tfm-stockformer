---
phase: 02-data-pipeline
plan: "04"
subsystem: data
tags: [numpy, pandas, normalization, train-val-test-split, npz, serialization]

requires:
  - phase: 02-03
    provides: feature CSVs with cross-sectional normalization baked in; label.csv with forward returns

provides:
  - normalize_split.py with cross_sectional_normalize() and split_by_date() exported
  - serialize_arrays.py with save_model_arrays() exported
  - split_indices.json (train_end, val_end) for reproducible train/val/test splits
  - flow.npz [T, N] raw forward returns for StockDataset regression target
  - trend_indicator.npz [T, N] binary 0/1 for StockDataset classification target

affects: [03-graph-embeddings, 04-model-training, 05-backtesting]

tech-stack:
  added: []
  patterns:
    - "Cross-sectional z-score normalization: per-row across N stocks, no time leakage"
    - "Date-ordered split by integer position: 75/12.5/12.5 train/val/test"
    - "NPZ output contract: np.savez(path, result=array) matching StockDataset interface"

key-files:
  created:
    - data_processing_script/sp500_pipeline/normalize_split.py
    - data_processing_script/sp500_pipeline/serialize_arrays.py
  modified:
    - tests/test_data_pipeline.py

key-decisions:
  - "normalize_split.py main() writes only split_indices.json — no re-normalization of feature CSVs (already done in 02-03)"
  - "flow.npz stores raw (not normalized) forward returns — label is a raw return signal, not a feature"
  - "trend_indicator values are (label > 0).astype(int32) — binary classification target matching StockDataset contract"

patterns-established:
  - "cross_sectional_normalize() exported for reuse: any module needing row-wise z-score can import from normalize_split"
  - "NPZ key must be 'result': np.load(file)['result'] is the StockDataset interface — all arrays serialized with this key"

requirements-completed: [DATA-04, DATA-05]

duration: 3min
completed: 2026-03-11
---

# Phase 2 Plan 4: Normalize-Split and Array Serialization Summary

**Date-ordered 75/12.5/12.5 train-val-test split with per-row cross-sectional normalization and flow.npz/trend_indicator.npz serialization matching StockDataset contract**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-11T10:06:15Z
- **Completed:** 2026-03-11T10:09:30Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Implemented `cross_sectional_normalize()` with per-row z-score (no time leakage) exported for reuse
- Implemented `split_by_date()` producing exact 750/125/125 split on 1000-row DataFrames by integer position
- Implemented `save_model_arrays()` producing `flow.npz` (raw forward returns) and `trend_indicator.npz` (binary 0/1)
- All DATA-04 and DATA-05 array tests pass; test_graph_embedding_shape remains xfail as expected

## Task Commits

Each task was committed atomically:

1. **Task 1 RED: failing tests** - `39953b9` (test)
2. **Task 1 GREEN: normalize_split.py + serialize_arrays.py** - `9e04fee` (feat)

_Note: TDD tasks have separate RED (test) and GREEN (implementation) commits_

## Files Created/Modified

- `data_processing_script/sp500_pipeline/normalize_split.py` - cross_sectional_normalize(), split_by_date(), main() writing split_indices.json
- `data_processing_script/sp500_pipeline/serialize_arrays.py` - save_model_arrays(), main()
- `tests/test_data_pipeline.py` - DATA-04 and DATA-05 xfail stubs replaced with real test implementations

## Decisions Made

- normalize_split.py main() writes only split_indices.json — no re-normalization of CSVs (already normalized in 02-03)
- flow.npz stores raw (not normalized) forward returns — label is the regression signal, normalization would distort it
- trend_indicator uses int32 with values (data > 0).astype(int32) to match StockDataset binary classification contract

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- Numerical pipeline complete: label.csv, feature CSVs, flow.npz, trend_indicator.npz all ready on disk
- split_indices.json provides reproducible train/val/test boundaries for Phase 3 and beyond
- Only remaining Phase 2 deliverable: Struc2Vec graph embeddings (Plan 02-05)

---
*Phase: 02-data-pipeline*
*Completed: 2026-03-11*

## Self-Check: PASSED

- normalize_split.py: FOUND
- serialize_arrays.py: FOUND
- 02-04-SUMMARY.md: FOUND
- Commit 39953b9 (RED): FOUND
- Commit 9e04fee (GREEN): FOUND
