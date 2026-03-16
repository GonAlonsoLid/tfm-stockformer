---
phase: 4
slug: evaluation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >=7.0,<8 |
| **Config file** | none — discovered by pytest from project root |
| **Quick run command** | `pytest tests/test_compute_ic.py -x -q` |
| **Full suite command** | `pytest tests/ -x -q` |
| **Estimated runtime** | ~5 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_compute_ic.py -x -q`
- **After every plan wave:** Run `pytest tests/ -x -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 0 | EVAL-01 | unit | `pytest tests/test_compute_ic.py::test_ic_known_correlation -x` | ❌ W0 | ⬜ pending |
| 4-01-02 | 01 | 0 | EVAL-01 | unit | `pytest tests/test_compute_ic.py::test_ic_nan_handling -x` | ❌ W0 | ⬜ pending |
| 4-01-03 | 01 | 0 | EVAL-01 | unit | `pytest tests/test_compute_ic.py::test_icir_formula -x` | ❌ W0 | ⬜ pending |
| 4-01-04 | 01 | 0 | EVAL-02 | unit | `pytest tests/test_compute_ic.py::test_mae_rmse_zero_for_perfect -x` | ❌ W0 | ⬜ pending |
| 4-01-05 | 01 | 0 | EVAL-02 | unit | `pytest tests/test_compute_ic.py::test_f1_perfect_classification -x` | ❌ W0 | ⬜ pending |
| 4-01-06 | 01 | 1 | EVAL-01, EVAL-02 | smoke | `pytest tests/test_compute_ic.py::test_smoke_actual_output -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_compute_ic.py` — stubs for EVAL-01 and EVAL-02 (script doesn't exist yet; tests go in Wave 0 before implementation)

*conftest.py already exists with shared fixtures — no gap there.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Console table formatting looks readable | EVAL-01 | Visual layout check | Run `python compute_ic.py --output_dir output/Multitask_output_SP500_2018-2024/` and verify table alignment |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
