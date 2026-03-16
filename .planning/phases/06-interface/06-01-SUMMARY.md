---
phase: 06-interface
plan: 01
subsystem: testing
tags: [streamlit, plotly, tdd, xfail, pytest, dark-theme]

# Dependency graph
requires:
  - phase: 05-portfolio-backtesting
    provides: backtest outputs (backtest_daily_returns.csv, backtest_summary.csv, regression_pred_last_step.csv) that the synthetic fixtures mirror
provides:
  - 7 xfail test stubs for UI-01 through UI-04 in tests/test_app.py
  - 3 synthetic fixtures in tests/conftest.py for app UI testing
  - Streamlit dark theme config in .streamlit/config.toml
  - Pinned streamlit and plotly dependencies in requirements.txt
affects: [06-02, 06-03, 06-04]

# Tech tracking
tech-stack:
  added: [streamlit>=1.35<=1.50, plotly>=5.18<7]
  patterns: [xfail(strict=False) TDD RED phase with imports inside test bodies to avoid ImportError before app.py exists]

key-files:
  created:
    - tests/test_app.py
    - .streamlit/config.toml
  modified:
    - tests/conftest.py
    - requirements.txt

key-decisions:
  - "xfail(strict=False) with imports inside test bodies — consistent with Phase 02-01 and 04-01 convention; keeps CI GREEN before app.py exists"
  - "streamlit<=1.50 pins last version supporting Python 3.9; plotly>=5.18 ensures go.Heatmap zmid support"
  - "Synthetic fixtures use fixed np.random seeds (7, 13) for deterministic test data"

patterns-established:
  - "Phase 6 test pattern: all app.py imports inside test bodies behind xfail(strict=False) markers"
  - "Streamlit dark theme: #0E1117 background, #1E2130 secondary, #4C9BE8 primary, #E5E7EB text"

requirements-completed: [UI-01, UI-02, UI-03, UI-04]

# Metrics
duration: 5min
completed: 2026-03-16
---

# Phase 6 Plan 01: Wave 0 Test Scaffold Summary

**7 xfail TDD stubs for Streamlit app (UI-01 to UI-04) with synthetic fixtures, dark theme config, and pinned streamlit/plotly dependencies**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-16T11:42:25Z
- **Completed:** 2026-03-16T11:47:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Created tests/test_app.py with 7 xfail stubs covering all 4 UI requirements (UI-01 through UI-04)
- Added 3 synthetic fixtures (backtest_daily_returns_fixture, backtest_summary_fixture, regression_pred_fixture) to conftest.py
- Created .streamlit/config.toml with dark theme (6 color tokens)
- Pinned streamlit>=1.35,<=1.50 and plotly>=5.18,<7 in requirements.txt; suite exits 0

## Task Commits

Each task was committed atomically:

1. **Task 1: Create tests/test_app.py with xfail stubs + augment conftest.py** - `e536574` (test)
2. **Task 2: Add .streamlit/config.toml + pin streamlit and plotly** - `1f00789` (chore)

## Files Created/Modified

- `tests/test_app.py` - 7 xfail test stubs covering UI-01 (imports/constants), UI-02 (equity chart), UI-03 (metrics table), UI-04 (heatmap)
- `tests/conftest.py` - Added Phase 6 fixtures section with 3 synthetic data fixtures
- `.streamlit/config.toml` - Streamlit dark theme with primaryColor #4C9BE8, backgroundColor #0E1117
- `requirements.txt` - Added streamlit and plotly version pins

## Decisions Made

- xfail(strict=False) with imports inside test bodies — consistent with Phase 02-01 and 04-01 convention; keeps CI GREEN before app.py exists
- streamlit<=1.50: last version supporting Python 3.9; 1.51 dropped Python 3.9 support
- plotly>=5.18: confirmed go.Heatmap zmid support; <7 avoids breaking API changes
- Synthetic fixtures use fixed np.random seeds for deterministic test data

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- TDD RED phase complete: 7 xfail stubs are in place, CI stays green throughout development
- app.py skeleton can now be built in 06-02 against these test contracts
- Streamlit theme config ready for use when app.py is created

---
*Phase: 06-interface*
*Completed: 2026-03-16*
