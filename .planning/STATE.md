# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-09)

**Core value:** A working, reproducible end-to-end pipeline that trains the Stockformer on S&P500 data, generates portfolios, and lets the user interactively explore backtest results.
**Current focus:** Phase 1 - Infrastructure

## Current Position

Phase: 1 of 7 (Infrastructure)
Plan: 2 of 3 in current phase
Status: Phase 1 plans 01-01 and 01-02 complete; ready for Phase 2
Last activity: 2026-03-10 — Plans 01-01 and 01-02 executed; INFRA-01, INFRA-02, INFRA-03 all verified

Progress: [██░░░░░░░░] 10%

## Performance Metrics

**Velocity:**
- Total plans completed: 2
- Average duration: ~5 min
- Total execution time: ~10 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-infrastructure | 2 | ~10 min | ~5 min |

**Recent Trend:**
- Last 5 plans: 01-01 (6 min), 01-02 (~5 min)
- Trend: Fast (infrastructure + docs plans)

*Updated after each plan completion*

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

### Pending Todos

None yet.

### Blockers/Concerns

- Struc2Vec graph embeddings must be regenerated for S&P500 universe — required deliverable of Phase 2
- pytorch-wavelets 1.3.0 patched and working (no further action needed in Phase 1)
- `applymap` → `map` pandas 2.x bug fixed in results_data_processing.py (Plan 01-01 — resolved)

## Session Continuity

Last session: 2026-03-10
Stopped at: Completed 01-02-PLAN.md — INFRA-03 human-verified; Phase 1 infrastructure complete
Resume file: None
