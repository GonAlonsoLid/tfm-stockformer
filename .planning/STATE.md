---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: Phase 4 context gathered
last_updated: "2026-03-14T13:46:19.098Z"
last_activity: 2026-03-12 — Plan 03-03 executed; standalone inference script created
progress:
  total_phases: 7
  completed_phases: 3
  total_plans: 12
  completed_plans: 12
  percent: 63
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-09)

**Core value:** A working, reproducible end-to-end pipeline that trains the Stockformer on S&P500 data, generates portfolios, and lets the user interactively explore backtest results.
**Current focus:** Phase 3 - Model Training

## Current Position

Phase: 3 of 7 (Model Training)
Plan: 3 of 4 in current phase
Status: Phase 3 plan 03-03 complete; scripts/run_inference.py created, MODEL-02 inference tests green
Last activity: 2026-03-12 — Plan 03-03 executed; standalone inference script created

Progress: [███████░░░] 63%

## Performance Metrics

**Velocity:**
- Total plans completed: 3
- Average duration: ~4 min
- Total execution time: ~11 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-infrastructure | 2 | ~10 min | ~5 min |
| 02-data-pipeline | 2 | ~6 min | ~3 min |

**Recent Trend:**
- Last 5 plans: 01-01 (6 min), 01-02 (~5 min), 02-01 (~1 min), 02-02 (~5 min)
- Trend: Steady

*Updated after each plan completion*
| Phase 02-data-pipeline P03 | 6 | 2 tasks | 2 files |
| Phase 02-data-pipeline P04 | 3 | 2 tasks | 3 files |
| Phase 02-data-pipeline P05 | 7 | 2 tasks | 4 files |
| Phase 02-data-pipeline P06 | 2 | 2 tasks | 2 files |
| Phase 03-model-training P02 | 3 | 1 tasks | 1 files |
| Phase 03-model-training P03 | 2 | 1 tasks | 1 files |
| Phase 03-model-training P04 | 1 | 1 tasks | 0 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Setup]: Preserve original model architecture — thesis contribution is adaptation, not architectural novelty
- [Setup]: Use yfinance only — free tier, sufficient for daily OHLCV at thesis scope
- [Setup]: Replace Qlib Alpha-360 with custom TA features — Qlib targets Chinese market conventions
- [Setup]: Config-driven paths — highest-impact portability fix; removes `/root/autodl-tmp/` hardcodes
- [01-01]: scikit-learn pinned as >=1.1.2 (not ==) to support Python 3.9 venv and Python 3.11+ simultaneously
- [01-01]: .ipynb_checkpoints/ excluded from path-grep test — Jupyter-generated cache, not source code
- [01-02]: SETUP.md extended to seven sections (beyond plan minimum of four) to include optional test suite, project structure, and troubleshooting
- [02-01]: xfail stubs use strict=False so suite collects without failing until implementation lands
- [02-01]: feature_matrix_fixture shape [280, 5] = 300 days minus 20-day warmup for longest TA window
- [Phase 02-02]: download_ohlcv.py named without leading digit — Python module names cannot start with a digit
- [Phase 02-02]: yfinance import guarded with try/except so module imports without yfinance installed (tests run offline)
- [Phase 02-data-pipeline]: Pure-pandas TA instead of pandas_ta: pandas_ta requires Python >=3.12 but project venv is Python 3.9
- [Phase 02-data-pipeline]: Normalization wired at save_feature_csvs() — CSVs on disk are pre-normalized per DATA-03 spec
- [Phase 02-data-pipeline]: normalize_split.py main() writes only split_indices.json — no re-normalization of CSVs (already done in 02-03)
- [Phase 02-data-pipeline]: flow.npz stores raw (not normalized) forward returns — label is regression signal, normalization would distort it
- [Phase 02-data-pipeline]: NPZ key 'result' mandatory for StockDataset interface: np.savez(path, result=array)
- [Phase 02-data-pipeline]: graph_embedding.py calls ge.Struc2Vec directly (not legacy script) to keep correlation threshold consistent between corr_adj.npy and data.edgelist
- [Phase 02-data-pipeline]: test_graph_embedding_shape validates file-format contract only (shape [N,128]) without requiring ge installed — keeps CI fast
- [Phase 02-data-pipeline]: build_pipeline.py uses sentinel files for idempotent reruns; --force overrides all sentinels
- [Phase 02-data-pipeline]: TA feature expansion to 69 columns: all pure-pandas, no pandas_ta/TA-Lib, satisfies Phase 3 Alpha-360 replacement prereq
- [Phase 02-data-pipeline]: test_feature_count_for_phase3 added as machine-verifiable Phase 3 gate (>= 60 features)
- [Phase 03-model-training]: alpha_360_dir points to features/ subdirectory (not parent) to avoid pandas parse errors on .npz/.npy files
- [Phase 03-model-training]: max_epoch=50 (down from CN config 100) to reduce training wall time for S&P500 run
- [Phase 03-03]: loadGraph() returns adjgat directly (not a tuple) — plan interface docs were outdated; actual lib/graph_utils.py returns only adjgat
- [Phase 03-03]: Inference loop uses raw numpy arrays from dataset attributes (not DataLoader) — consistent with training script pattern
- [Phase 03-model-training]: Phase 2 pipeline (build_pipeline.py) must be run before any training smoke test — data/Stock_SP500_2018-2024/ did not exist on this machine

### Pending Todos

None yet.

### Blockers/Concerns

- Struc2Vec graph embeddings must be regenerated for S&P500 universe — required deliverable of Phase 2
- pytorch-wavelets 1.3.0 patched and working (no further action needed in Phase 1)
- `applymap` → `map` pandas 2.x bug fixed in results_data_processing.py (Plan 01-01 — resolved)

## Session Continuity

Last session: 2026-03-14T13:46:19.094Z
Stopped at: Phase 4 context gathered
Resume file: .planning/phases/04-evaluation/04-CONTEXT.md
