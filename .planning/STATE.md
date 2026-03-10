# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-09)

**Core value:** A working, reproducible end-to-end pipeline that trains the Stockformer on S&P500 data, generates portfolios, and lets the user interactively explore backtest results.
**Current focus:** Phase 1 - Infrastructure

## Current Position

Phase: 1 of 7 (Infrastructure)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-03-10 — Roadmap created; 25 requirements mapped across 7 phases

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: none yet
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Setup]: Preserve original model architecture — thesis contribution is adaptation, not architectural novelty
- [Setup]: Use yfinance only — free tier, sufficient for daily OHLCV at thesis scope
- [Setup]: Replace Qlib Alpha-360 with custom TA features — Qlib targets Chinese market conventions
- [Setup]: Config-driven paths — highest-impact portability fix; removes `/root/autodl-tmp/` hardcodes

### Pending Todos

None yet.

### Blockers/Concerns

- pytorch-wavelets 1.3.0 may need patching for PyTorch >= 2.0 — surface this during Phase 1 environment setup
- `applymap` → `map` pandas 2.x bug in `results_data_processing.py` — fix during Phase 1 path cleanup
- Struc2Vec graph embeddings must be regenerated for S&P500 universe — required deliverable of Phase 2

## Session Continuity

Last session: 2026-03-10
Stopped at: Roadmap created; ready to plan Phase 1
Resume file: None
