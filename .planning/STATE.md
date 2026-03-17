---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: Completed 09-04-PLAN.md
last_updated: "2026-03-17T11:49:57.279Z"
last_activity: 2026-03-14 — Plan 04-02 executed; standalone evaluation script created
progress:
  total_phases: 9
  completed_phases: 7
  total_plans: 26
  completed_plans: 25
  percent: 67
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-09)

**Core value:** A working, reproducible end-to-end pipeline that trains the Stockformer on S&P500 data, generates portfolios, and lets the user interactively explore backtest results.
**Current focus:** Phase 3 - Model Training

## Current Position

Phase: 4 of 7 (Evaluation)
Plan: 2 of 2 in current phase
Status: Phase 4 plan 04-02 complete; scripts/compute_ic.py implemented, EVAL-01 and EVAL-02 satisfied, all 6 tests xpassing
Last activity: 2026-03-14 — Plan 04-02 executed; standalone evaluation script created

Progress: [████████░░] 67%

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
| Phase 04-evaluation P01 | 1 | 1 tasks | 1 files |
| Phase 04-evaluation P02 | 5 | 2 tasks | 1 files |
| Phase 05-portfolio-backtesting P01 | 1 | 1 tasks | 1 files |
| Phase 05-portfolio-backtesting P03 | 15 | 2 tasks | 1 files |
| Phase 06-interface P01 | 5 | 2 tasks | 4 files |
| Phase 06-interface P02 | 10 | 1 tasks | 1 files |
| Phase 08-alpha360-feature-replacement P01 | 2 | 1 tasks | 1 files |
| Phase 08-alpha360-feature-replacement P02 | 5 | 1 tasks | 1 files |
| Phase 09-pipeline-cleanup-and-restructuring P01 | 85s | 2 tasks | 2 files |
| Phase 09-pipeline-cleanup-and-restructuring P02 | 6 | 2 tasks | 6 files |
| Phase 09 P04 | 5 | 3 tasks | 2 files |

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
- [Phase 04-evaluation]: xfail(strict=False) pattern for TDD Red phase tests — consistent with project convention from 02-01; imports inside test bodies prevent module-level ImportError
- [Phase 04-02]: spearmanr(...).statistic used (not .correlation) — scipy 1.7+ changed return API
- [Phase 04-02]: np.std(ddof=1) for ICIR denominator — sample std is academically correct for IC time series
- [Phase 04-02]: f1_score(average="macro") — class distribution 47.6/52.4% nearly balanced; macro is academically conservative
- [Phase 04-02]: main() accepts optional output_dir kwarg for programmatic smoke test invocation without spawning subprocess
- [Phase 04-02]: Pearson IC included as bonus column alongside Spearman — provides thesis reviewers additional correlation context
- [Phase 05-portfolio-backtesting]: xfail(strict=False) Wave 0 stubs with imports inside test bodies — consistent with Phase 02-01 and 04-01 convention; keeps CI GREEN before scripts/run_backtest.py exists
- [Phase 05-portfolio-backtesting]: yfinance import guarded with try/except and _YF_AVAILABLE sentinel — pure functions remain importable without yfinance in test environments
- [Phase 05-portfolio-backtesting]: derive_date_index uses bdate_range first then falls back to label.csv + split_indices.json for US-holiday-aware date alignment
- [Phase 06-interface]: xfail(strict=False) with imports inside test bodies for Phase 6 — consistent with Phase 02-01 and 04-01 convention; keeps CI GREEN before app.py exists
- [Phase 06-interface]: streamlit<=1.50 pinned: last version supporting Python 3.9; plotly>=5.18 ensures go.Heatmap zmid support
- [Phase 06-interface]: cumprod().shift(1, fill_value=1.0) for equity curve starting at 1.0 — matches test contract and standard equity curve convention
- [Phase 06-interface]: build_heatmap() slices tickers[:k] internally — allows callers to pass longer list with explicit k; satisfies test_heatmap_top_k_filter
- [Phase 06-interface]: format_metrics_table() returns numeric DataFrame without fillna to preserve Streamlit NumberColumn formatting
- [Phase 08-alpha360-feature-replacement]: xfail(strict=False) with imports inside test bodies for Phase 8 Wave 0 — consistent with Phase 02-01, 04-01, 05-01, 06-01 convention; keeps CI GREEN before scripts/build_alpha360.py exists
- [Phase 08-alpha360-feature-replacement]: build_alpha360.main(config_path, data_dir) interface locked in test file docstring — Plan 02 executor must implement this exact signature
- [Phase 08-alpha360-feature-replacement]: tqdm fallback implemented as context-manager-compatible class (not plain function) for 'with tqdm(...) as pbar:' pattern
- [Phase 08-alpha360-feature-replacement]: features_dir cleared after backup before writing 360 new CSVs to ensure exact count of 360
- [Phase 09-pipeline-cleanup-and-restructuring]: xfail(strict=False) pattern maintained for Wave 0 stubs — consistent with all prior phases
- [Phase 09-pipeline-cleanup-and-restructuring]: test_download_ohlcv.py imports from scripts.sp500_pipeline (post-move path) so tests go green after Plan 09-02 without modification
- [Phase 09-pipeline-cleanup-and-restructuring]: scripts/sp500_pipeline/ is now canonical location for pipeline step scripts; data_processing_script/sp500_pipeline/ retained until Plan 09-04 cleanup
- [Phase 09-pipeline-cleanup-and-restructuring]: download_ohlcv.py ticker fallback reads tickers.txt from args.data_dir before calling Wikipedia scrape; first run still writes the file
- [Phase 09]: data_processing_script/ deleted in full — sp500_pipeline/ already moved in 09-02; Chinese-market subdirs fully obsolete
- [Phase 09]: README.md Quick Start uses 7-step sequence matching CONTEXT.md locked command list with GPU note on training step only

### Roadmap Evolution

- Phase 8 added: Alpha360 Feature Replacement
- Phase 9 added: Pipeline cleanup and restructuring

### Pending Todos

None yet.

### Blockers/Concerns

- Struc2Vec graph embeddings must be regenerated for S&P500 universe — required deliverable of Phase 2
- pytorch-wavelets 1.3.0 patched and working (no further action needed in Phase 1)
- `applymap` → `map` pandas 2.x bug fixed in results_data_processing.py (Plan 01-01 — resolved)

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 1 | Add backtest_positions.csv output to scripts/run_backtest.py | 2026-03-14 | bec131a | [1-add-backtest-positions-csv-output-to-scr](.planning/quick/1-add-backtest-positions-csv-output-to-scr/) |

## Session Continuity

Last session: 2026-03-17T11:49:57.276Z
Stopped at: Completed 09-04-PLAN.md
Resume file: None
