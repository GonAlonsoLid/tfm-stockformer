# Phase 3: Model Training - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning (pending Phase 2 feature expansion prerequisite)

<domain>
## Phase Boundary

Create a new `.conf` config file targeting S&P500 data, wire it to Phase 2 outputs, train the Stockformer end-to-end, and produce prediction CSVs on the test period. Architecture is preserved as-is — thesis contribution is adaptation, not architectural novelty.

</domain>

<decisions>
## Implementation Decisions

### S&P500 config file
- Config filename: `config/Multitask_Stock_SP500.conf`
- Data directory: `./data/Stock_SP500_2018-2024/` (mirrors `./data/Stock_CN_2021-2024/` convention)
- Date range: 2018-01-01 to 2024-12-31
- Train/val/test ratios: keep 0.75/0.125/0.125 (matches Phase 2 split)
- All other hyperparameters (`layers`, `heads`, `dims`, `wave`, `level`) kept from original config

### Training duration
- `max_epoch = 50` as default in config (down from original 100)
- Overridable via `--max_epoch` CLI flag at runtime (e.g., `--max_epoch 100` for full run)
- `batch_size = 12` unchanged
- No early stopping — run full max_epoch, save best checkpoint (lowest val MAE)

### Inference script
- Create `scripts/run_inference.py` as a standalone script separate from the training script
- Reads config via `--config` flag (same as training script)
- Optional `--checkpoint` flag overrides `model_file` from config; defaults to config path
- Output structure unchanged: `output_dir/classification/` and `output_dir/regression/` CSVs
  (classification_pred_last_step.csv, regression_pred_last_step.csv, and label counterparts)

### Feature CSV mapping (alpha_360_dir)
- Phase 2's TA feature CSVs are used directly as `alpha_360_dir` — no adapter step needed
- Format confirmed: one CSV per feature, rows=dates, cols=stock tickers (wide format)
- `infea = num_feature_CSVs + 2` (StockDataset computes this dynamically at load time)
- **PREREQUISITE**: Phase 2 must be extended to ~60-80 features before Phase 3 runs (see below)

### Phase 2 feature expansion (prerequisite for Phase 3)
- Current Phase 2 output: 16 TA features — insufficient to replace Alpha-360's richness
- Decision: expand Phase 2 to ~60-80 features by adding more TA windows + new indicators
- New indicators to add: ATR, OBV, Stochastic, Williams %R, CCI, Donchian channels, plus additional RSI/MACD/ROC window variants
- This requires a new Phase 2 plan (02-06) before Phase 3 can execute
- Thesis narrative: "We replace China-specific Alpha-360 factors with ~70 universal TA indicators"

### Claude's Discretion
- Exact list of 60-80 features (researcher determines the full set from standard US TA indicators)
- How to handle stocks dropped from S&P500 during 2018-2024 (survivorship bias — researcher investigates)
- TensorBoard log directory naming convention for S&P500 runs
- Exact `cpt/` and `log/` subdirectory naming

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `config/Multitask_Stock.conf`: Template for the new S&P500 config — copy and update paths/dates
- `MultiTask_Stockformer_train.py`: Training script, already config-driven via argparse; no structural changes needed
- `lib/Multitask_Stockformer_utils.py:StockDataset`: Loads `alpha_360_dir` CSVs directly; confirmed compatible with Phase 2 wide-format feature CSVs
- `lib/graph_utils.py`: Loads `corr_adj.npy` and `adjgat` — Phase 2 produces these files

### Established Patterns
- Config-driven paths: all file paths come from `.conf` via argparse — no hardcoding needed
- `StockDataset` computes `infea` dynamically: `bonus_all.shape[-1] + 2` — no manual dimension editing
- Checkpoint saving: saves to `args.model_file` whenever val MAE improves; inference loads from same path

### Integration Points
- Phase 2 outputs needed by config:
  - `flow.npz` → `[file] traffic`
  - `trend_indicator.npz` → `[file] indicator`
  - `corr_adj.npy` → `[file] adj`
  - `128_corr_struc2vec_adjgat.npy` → `[file] adjgat`
  - TA feature CSV directory → `[file] alpha_360_dir`
- Phase 4 (Evaluation) reads from `output_dir/regression/regression_pred_last_step.csv`

</code_context>

<specifics>
## Specific Ideas

- `--max_epoch` override matters: user wants to be able to run `python MultiTask_Stockformer_train.py --config config/Multitask_Stock_SP500.conf --max_epoch 10` for quick sanity checks
- Phase 2 extension should target ~70 features to give the model enough signal richness for a thesis-quality evaluation
- Thesis narrative framing: adaptation story centers on replacing Chinese-market Qlib factors with universally applicable US TA indicators

</specifics>

<deferred>
## Deferred Ideas

- US Alpha-360 equivalent with 200+ features — major engineering effort, out of thesis scope
- Cross-sectional price-volume features (relative ranks, turnover proxies) — consider for v2 if 70 TA features underperform
- Walk-forward cross-validation — explicitly out of scope per REQUIREMENTS.md

</deferred>

---

*Phase: 03-model-training*
*Context gathered: 2026-03-11*
