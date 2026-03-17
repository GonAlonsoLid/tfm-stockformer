# Phase 9: Pipeline Cleanup and Restructuring - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Remove legacy code that became obsolete after the Alpha360 migration, consolidate all
active pipeline scripts under `scripts/`, integrate `build_alpha360.py` into
`build_pipeline.py` as a unified end-to-end command, improve ticker sourcing resilience,
and update the README with the full reproduction command sequence.

This phase does NOT add new ML capabilities, retrain the model, or change any data
contracts. It is purely structural cleanup and ergonomics improvement for reproducibility.

</domain>

<decisions>
## Implementation Decisions

### Legacy Code Removal
- Delete the **entire `data_processing_script/` tree** — all directories and files
- Before deleting, move `data_processing_script/sp500_pipeline/` to `scripts/sp500_pipeline/`
  and update `build_pipeline.py`'s `PIPELINE_DIR` to point to the new location
- `results_data_processing.py` (inside `stockformer_input_data_processing/`) is **deleted** —
  its functionality is superseded by `scripts/run_backtest.py` and `scripts/compute_ic.py`
- Directories being deleted:
  - `data_processing_script/volume_and_price_factor_construction/` (Qlib TA notebooks — obsolete)
  - `data_processing_script/stockformer_input_data_processing/` (original Chinese-market scripts)
  - `data_processing_script/sp500_pipeline/` (moved to `scripts/sp500_pipeline/` first)
  - `data_processing_script/` itself (now empty after moves)

### Pipeline Integration
- Integrate `build_alpha360.py` as **step 5** in `build_pipeline.py` so one command
  runs the full data preparation pipeline:
  `download → normalize → serialize → graph → alpha360`
- `build_pipeline.py` is updated to accept `--config config/Multitask_Stock_SP500.conf`
  as the **primary interface**, deriving `data_dir` and `alpha_360_dir` from the config
- The existing `--data_dir`, `--start`, `--end`, `--force` flags remain for backward
  compatibility when running without a config file
- Step 5 sentinel: check that `features/` directory contains exactly 360 CSV files
- Invocation after this phase: `python scripts/build_pipeline.py --config config/Multitask_Stock_SP500.conf`

### Ticker Sourcing
- `download_ohlcv.py` checks for an existing `tickers.txt` in `data_dir` **before**
  fetching from Wikipedia
- If `tickers.txt` exists: read tickers from file, skip Wikipedia scrape entirely
- If `tickers.txt` does not exist: fetch from Wikipedia as before, then save to file
- Rationale: avoids network dependency and Wikipedia format fragility on re-runs;
  first run still auto-populates the file for future runs

### End-to-End Documentation
- Update **README.md** (existing file) with a new "Quick Start" or "Pipeline Reproduction"
  section containing the full command sequence from fresh clone to backtest output:
  1. Install dependencies
  2. `python scripts/build_pipeline.py --config config/Multitask_Stock_SP500.conf`
  3. `python MultiTask_Stockformer_train.py --config config/Multitask_Stock_SP500.conf`
  4. `python scripts/run_inference.py --config config/Multitask_Stock_SP500.conf`
  5. `python scripts/compute_ic.py`
  6. `python scripts/run_backtest.py`
  7. `streamlit run app.py`
- README should note that step 3 (training) requires GPU for reasonable runtime (Kaggle
  recommended); all other steps run on CPU
- Remove references to `data_processing_script/` from README (currently describes the
  Chinese-market legacy scripts)

### Claude's Discretion
- Exact README section structure and wording
- Whether to add `--help` output snippets to README
- Internal refactoring within `build_pipeline.py` (how config parsing is implemented)
- Whether to add `__init__.py` to `scripts/sp500_pipeline/` after the move

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pipeline Orchestrator
- `scripts/build_pipeline.py` — Current orchestrator; PIPELINE_DIR path must be updated after move; step list must have Alpha360 added as step 5
- `scripts/build_alpha360.py` — Step 5 to integrate; reads config via argparse --config; entry point is `main(config_path, data_dir)`

### Config Interface
- `config/Multitask_Stock_SP500.conf` — Authoritative config; provides `traffic`, `alpha_360_dir`, and all path references; `data_dir` is derived from the parent of `traffic` path

### Scripts Being Moved
- `data_processing_script/sp500_pipeline/download_ohlcv.py` — Contains `get_sp500_tickers()` function; needs local-tickers.txt fallback logic added
- `data_processing_script/sp500_pipeline/normalize_split.py` — Move only, no logic changes
- `data_processing_script/sp500_pipeline/serialize_arrays.py` — Move only, no logic changes
- `data_processing_script/sp500_pipeline/graph_embedding.py` — Move only, no logic changes

### Documentation Target
- `README.md` — Existing file; currently describes Chinese-market setup; section on `data_processing_script/` must be updated/removed

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `build_alpha360.main(config_path, data_dir)` — Callable interface already exists; step 5 integration calls this directly (no subprocess needed)
- `build_pipeline.py` STEPS list — Simple list of `(script_name, sentinel)` tuples; adding step 5 follows the same pattern

### Established Patterns
- Sentinel-based idempotency: each step checks for its output file before running; step 5 sentinel = 360 CSV files in `features/`
- `subprocess.run()` for each step — step 5 may call `build_alpha360.main()` directly as a Python function import instead of subprocess (cleaner)
- `argparse` with `formatter_class=ArgumentDefaultsHelpFormatter` — follow same style

### Integration Points
- `build_pipeline.py` imports `PIPELINE_DIR` as the scripts location — this is the single path to update after move
- `download_ohlcv.py`'s `get_sp500_tickers()` — add local-file check at the top of this function
- README.md `## File Description` section — primary target for content removal/replacement

</code_context>

<specifics>
## Specific Ideas

- After move, `scripts/sp500_pipeline/` sits alongside `build_pipeline.py` — makes the dependency explicit by co-location
- Unified `--config` interface means a user only needs to know one flag to run the full pipeline, matching the training script interface
- The ticker fallback logic is a one-liner: check `os.path.exists(os.path.join(data_dir, "tickers.txt"))` before the Wikipedia call

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope

</deferred>

---

*Phase: 09-pipeline-cleanup-and-restructuring*
*Context gathered: 2026-03-17*
