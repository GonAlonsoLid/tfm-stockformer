# Phase 4: Evaluation - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Score the model's test-set predictions with IC (Information Coefficient) and ICIR, and ensure MAE, RMSE, accuracy, and F1 are all reported in a single evaluation command. No new training, no portfolio construction — pure post-hoc scoring of the CSVs produced by inference.

</domain>

<decisions>
## Implementation Decisions

### Script location and interface
- Standalone script: `scripts/compute_ic.py`
- Entry point: `python scripts/compute_ic.py --output_dir output/Multitask_output_SP500_2018-2024`
- Reads `regression_pred_last_step.csv` and `regression_label_last_step.csv` from the output_dir directly
- No config file required — CSVs contain everything needed
- No integration with run_inference.py — explicitly kept separate

### Scope of compute_ic.py
- Reports ALL evaluation metrics in one shot: IC, ICIR, MAE, RMSE, accuracy, F1
- Single command gives the complete results table for the thesis
- Does NOT split by step (step 1 / step 2) — only global test-period summary
- Single run at a time (no multi-run comparison)

### Output format
- Console: formatted summary table with all metrics
- File 1: `evaluation_summary.csv` — one row with IC_mean, ICIR, MAE, RMSE, accuracy, F1
- File 2: `ic_by_day.csv` — one row per trading day with date index and IC value (for time-series plots in thesis)
- Both CSVs saved into the same `output_dir`

### F1 metric
- Add F1 computation for the classification head (up/down direction prediction)
- Binary F1 (macro or weighted — Claude's discretion on which is more appropriate for balanced classes)
- Classification CSVs already exist: `classification_pred_last_step.csv` and `classification_label_last_step.csv`

### IC definition
- IC = Spearman rank correlation between predicted returns and realized returns
- Computed **per day** across all stocks (standard cross-sectional quant definition)
- ICIR = mean(daily IC) / std(daily IC) over the test period
- (Pearson IC as additional column is Claude's discretion — not required by user)

### Claude's Discretion
- Whether to also compute Pearson IC alongside Spearman
- F1 averaging strategy (macro vs weighted)
- Exact table formatting (tabulate library or manual formatting)
- How to handle days with NaN IC (e.g., if all stocks have same predicted rank)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `lib/Multitask_Stockformer_utils.py:metric()`: Already computes acc, mae, rmse, mape from numpy arrays — can reuse or replicate this logic in compute_ic.py
- `scripts/run_inference.py`: Pattern reference for how to read output_dir and construct CSV paths
- `output/Multitask_output_SP500_2018-2024/regression/regression_pred_last_step.csv`: Raw float matrix, no header, shape (n_test_days, n_stocks)
- `output/Multitask_output_SP500_2018-2024/classification/classification_pred_last_step.csv`: Classification predictions, same shape

### Established Patterns
- Output CSVs: headerless float matrices — rows = trading days, cols = stocks; need `pd.read_csv(..., header=None)`
- Metric function signature: `metric(reg_pred, reg_label, class_pred, class_label)` → returns `(acc, mae, rmse, mape)`
- scipy.stats.spearmanr is available (scipy already in requirements)

### Integration Points
- Phase 5 (Portfolio & Backtesting) reads from the same `regression_pred_last_step.csv` — no changes to file format needed
- `ic_by_day.csv` output may be consumed by Phase 6 (Streamlit) for rolling IC chart (deferred to Phase 6)

</code_context>

<specifics>
## Specific Ideas

- The `ic_by_day.csv` is explicitly produced with thesis plots in mind — user needs to chart IC over time in the results chapter
- Run pattern: `python scripts/run_inference.py --config ... && python scripts/compute_ic.py --output_dir ...`

</specifics>

<deferred>
## Deferred Ideas

- Rolling 20-day IC chart in Streamlit — Phase 6 (VIZ-01 in v2 requirements)
- IC distribution histogram — v2 (VIZ-02)
- Per-sector IC breakdown — v2 (VIZ-03)
- Multi-run comparison table for ablations — not needed for thesis scope

</deferred>

---

*Phase: 04-evaluation*
*Context gathered: 2026-03-14*
