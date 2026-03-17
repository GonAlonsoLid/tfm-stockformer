---
phase: 8
slug: alpha360-feature-replacement
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-17
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (project standard) |
| **Config file** | `pytest.ini` or `pyproject.toml` — check project root |
| **Quick run command** | `pytest tests/test_build_alpha360.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~30 seconds (unit tests with tmp_path fixtures) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_build_alpha360.py -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd:verify-work`:** Full suite must be green + manual validation snippet confirms `feature count=360, shape=(1449,478), NaN=0`
- **Max feedback latency:** ~30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 8-01-01 | 01 | 0 | ALPHA360-01..05 | unit stubs | `pytest tests/test_build_alpha360.py -x` | ❌ W0 | ⬜ pending |
| 8-02-01 | 02 | 1 | ALPHA360-01 | integration smoke | `pytest tests/test_build_alpha360.py::test_build_alpha360_creates_360_csvs -x` | ❌ W0 | ⬜ pending |
| 8-02-02 | 02 | 1 | ALPHA360-02 | unit | `pytest tests/test_build_alpha360.py::test_feature_csv_shape_and_no_nan -x` | ❌ W0 | ⬜ pending |
| 8-02-03 | 02 | 1 | ALPHA360-03 | unit | `pytest tests/test_build_alpha360.py::test_first_row_date -x` | ❌ W0 | ⬜ pending |
| 8-02-04 | 02 | 1 | ALPHA360-04 | unit | `pytest tests/test_build_alpha360.py::test_backup_created -x` | ❌ W0 | ⬜ pending |
| 8-02-05 | 02 | 1 | ALPHA360-05 | unit | `pytest tests/test_build_alpha360.py::test_column_order_matches_tickers -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_build_alpha360.py` — stubs/implementations for ALPHA360-01 through ALPHA360-05; uses `tmp_path` fixtures with synthetic Parquet data (no full dataset required)

*No framework install needed — pytest already present in project.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| IC improvement after retraining | Success Criterion 4 | Requires full Kaggle training run (hours) | Run `scripts/compute_ic.py` after retraining; expect IC mean > 0.01 |
| `.DS_Store` not counted in infea | macOS dev only | OS artifact, not present on Kaggle | Verify `len([f for f in os.listdir(feat_dir) if f.endswith('.csv')]) == 360` |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
