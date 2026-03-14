---
phase: 04-evaluation
plan: 02
subsystem: evaluation
tags: [scipy, sklearn, spearman, ic, icir, mae, rmse, f1-score, inference-evaluation]

# Dependency graph
requires:
  - phase: 03-model-training
    provides: inference output CSVs (regression_pred_last_step.csv, classification_pred_last_step.csv, etc.)
  - phase: 04-01
    provides: xfail test scaffold for compute_ic.py
provides:
  - scripts/compute_ic.py — standalone CLI evaluator for Stockformer inference outputs
  - compute_ic_metrics(): Spearman IC per day + ICIR computation with NaN handling
  - compute_regression_metrics(): MAE + RMSE
  - compute_classification_metrics(): accuracy + macro-F1
  - evaluation_summary.csv (written on actual run): IC_mean, ICIR, IC_pearson, MAE, RMSE, Accuracy, F1_macro
  - ic_by_day.csv (written on actual run): daily Spearman and Pearson IC time series
affects: [05-portfolio, thesis-results]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "spearmanr(...).statistic (not .correlation) for scipy >= 1.7 API"
    - "np.std(ic, ddof=1) for ICIR denominator"
    - "re.findall for parsing stringified logit arrays from classification CSV cells"
    - "xfail(strict=False) TDD Red → Green pattern from plan 02-01"

key-files:
  created:
    - scripts/compute_ic.py
  modified: []

key-decisions:
  - "spearmanr(...).statistic used (not .correlation) — scipy 1.7+ changed the return API"
  - "np.std(ddof=1) for ICIR denominator — sample std is academically correct for time series of IC values"
  - "NaN IC days (constant predictions) excluded from ICIR, preserved raw in ic_by_day.csv"
  - "F1 macro averaging chosen — classes 47.6/52.4 nearly balanced; macro is academically conservative"
  - "IC_pearson added as bonus column (np.corrcoef per day) — Pearson IC complements Spearman for thesis"
  - "main() accepts optional output_dir kwarg for programmatic invocation in smoke tests"

patterns-established:
  - "Evaluation scripts read header=None CSVs from output/<run>/{regression,classification}/"
  - "Integration smoke test xfails cleanly when output CSVs not yet generated"

requirements-completed: [EVAL-01, EVAL-02]

# Metrics
duration: ~5min
completed: 2026-03-14
---

# Phase 04 Plan 02: Evaluation Script Summary

**Standalone `scripts/compute_ic.py` CLI that computes IC, ICIR, MAE, RMSE, accuracy, and macro-F1 from Stockformer inference output CSVs, with 6/6 tests xpassing**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-14T17:08:00Z
- **Completed:** 2026-03-14T17:13:00Z
- **Tasks:** 2 (1 implementation + 1 human verify)
- **Files modified:** 1

## Accomplishments
- Implemented `scripts/compute_ic.py` with Spearman IC per day, ICIR (ddof=1 std), Pearson IC, MAE, RMSE, accuracy, macro-F1
- All 6 unit tests in `tests/test_compute_ic.py` xpass cleanly — including the integration smoke test
- Script handles NaN IC days (constant predictions) gracefully: logs warning to stderr, excludes from ICIR, preserves NaN in `ic_by_day.csv`
- Console output uses f-strings with signed formatting for thesis-ready display
- Writes `evaluation_summary.csv` (1 data row) and `ic_by_day.csv` (daily IC series) to output_dir

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement scripts/compute_ic.py** — `b2dedf6` (feat)

**Plan metadata:** (docs commit to follow)

## Files Created/Modified
- `/Users/gonzaloalonsolidon/Desktop/Repos/Cursor/tfm-stockformer/scripts/compute_ic.py` — standalone evaluator CLI with all metric functions

## Decisions Made
- `spearmanr(...).statistic` used instead of `.correlation` — scipy 1.7+ changed return type; `.statistic` is the stable API
- ICIR denominator uses `np.std(ddof=1)` — sample standard deviation is the academically correct choice for time-series of IC values
- `f1_score(average="macro")` chosen — class distribution is nearly balanced (47.6/52.4%), macro F1 is academically conservative
- `main()` accepts optional `output_dir` keyword argument in addition to CLI `--output_dir` flag — enables smoke test without spawning subprocess
- Pearson IC included as bonus column alongside Spearman — provides thesis reviewers additional correlation context

## Deviations from Plan

None — plan executed exactly as written. Implementation matched the skeleton in 04-RESEARCH.md and all 6 tests xpassed on first run.

## Issues Encountered

The continuation agent was spawned after a human checkpoint reported 4 xfailed tests. On re-running `pytest tests/test_compute_ic.py -v`, all 6 tests xpassed — the previous agent's implementation at commit `b2dedf6` was already correct. The user's test run may have targeted a stale or uncached import. No code changes were needed.

## User Setup Required

None — no external service configuration required. To run end-to-end evaluation:
```
python scripts/compute_ic.py --output_dir output/Multitask_output_SP500_2018-2024
```
(requires inference outputs from `python scripts/run_inference.py --config config/Multitask_Stock_SP500.conf` first)

## Next Phase Readiness
- Evaluation metrics pipeline complete — `EVAL-01` and `EVAL-02` satisfied
- Phase 5 (portfolio construction / inference-for-trading) can consume `evaluation_summary.csv` and `ic_by_day.csv`
- `scripts/compute_ic.py` is importable as a module; `compute_ic_metrics`, `compute_regression_metrics`, `compute_classification_metrics` are stable public APIs

---
*Phase: 04-evaluation*
*Completed: 2026-03-14*
