---
phase: 3
slug: model-training
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-11
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `tests/test_data_pipeline.py` (extended with model tests) |
| **Quick run command** | `python -m pytest tests/test_model_training.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -q` |
| **Estimated runtime** | ~10 seconds (unit/smoke tests only — full training excluded) |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_model_training.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 3-01-01 | 01 | 1 | MODEL-01 | unit | `python -m pytest tests/test_model_training.py::test_config_file_exists -xq` | ❌ W0 | ⬜ pending |
| 3-01-02 | 01 | 1 | MODEL-01 | unit | `python -m pytest tests/test_model_training.py::test_config_fields_present -xq` | ❌ W0 | ⬜ pending |
| 3-02-01 | 02 | 2 | MODEL-01 | smoke | `python -m pytest tests/test_model_training.py::test_dataset_loads -xq` | ❌ W0 | ⬜ pending |
| 3-03-01 | 03 | 3 | MODEL-02 | unit | `python -m pytest tests/test_model_training.py::test_inference_script_exists -xq` | ❌ W0 | ⬜ pending |
| 3-03-02 | 03 | 3 | MODEL-02 | unit | `python -m pytest tests/test_model_training.py::test_inference_script_args -xq` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_model_training.py` — stubs for MODEL-01, MODEL-02
- [ ] `tests/conftest.py` — update with Phase 3 fixtures (config path, tmp output dir)

*Existing framework (pytest) already installed — no new framework needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Training completes without error for 2 epochs | MODEL-01 | Requires real GPU/CPU time + Phase 2 data on disk | Run: `python MultiTask_Stockformer_train.py --config config/Multitask_Stock_SP500.conf --max_epoch 2` — must exit 0 with checkpoint written |
| TensorBoard logs written to log dir | MODEL-01 | File system artifact, not unit-testable | `ls runs/SP500/` after training smoke test |
| Inference CSV covers full test period | MODEL-02 | Requires trained checkpoint on disk | Run: `python scripts/run_inference.py --config config/Multitask_Stock_SP500.conf` — check output CSVs have test-period dates |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
