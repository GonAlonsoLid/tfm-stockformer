---
plan: 09-03
phase: 09-pipeline-cleanup-and-restructuring
status: complete
wave: 2
---

## Summary

Extended `scripts/build_pipeline.py` with `--config` flag, `_data_dir_from_config()`, `_alpha360_done()` sentinel, and `run_alpha360_step()` integrating `build_alpha360.main()` as step 5. All 5 `test_pipeline.py` tests xpassed.

## What Was Built

- `_data_dir_from_config(config_path)` — reads `.conf` file, derives `data_dir` from `cfg["file"]["traffic"]` parent and `alpha_360_dir` from `cfg["file"]["alpha_360_dir"]`
- `_alpha360_done(features_dir)` — returns True if exactly 360 CSV files exist in `features/`
- `run_alpha360_step(config_path, features_dir, force)` — runs step 5 via direct `from build_alpha360 import main` import; skips if sentinel satisfied
- `--config` argparse flag that overrides `--data_dir` when provided
- Step 5 call after the STEPS for-loop when `--config` is present; info message when not

## Key Files

- `scripts/build_pipeline.py` — all changes already present from initial file creation

## Test Results

- `tests/test_pipeline.py` — 5/5 xpassed
- `tests/test_download_ohlcv.py` — 2/2 xpassed (from plan 09-02)
- Pre-existing: `test_torch_importable` fails because torch is not installed in dev environment (unrelated to this phase)

## Verification

```
python3 -c "from scripts.build_pipeline import _data_dir_from_config; d, a = _data_dir_from_config('config/Multitask_Stock_SP500.conf'); print(d, a)"
# → .../Stock_SP500_2018-01-01_2024-01-01 .../features

python3 scripts/build_pipeline.py --help
# → shows --config, --data_dir, --start, --end, --force
```

## Self-Check: PASSED
