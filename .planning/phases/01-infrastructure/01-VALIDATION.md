---
phase: 1
slug: infrastructure
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | tests/conftest.py — Wave 0 installs |
| **Quick run command** | `pytest tests/test_phase1_infra.py -v` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~15 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_phase1_infra.py -v`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 1-01-01 | 01 | 1 | INFRA-01 | smoke | `python -c "from MultiTask_Stockformer_train import *"` | ❌ W0 | ⬜ pending |
| 1-01-02 | 01 | 1 | INFRA-01 | unit | `pytest tests/test_phase1_infra.py::test_no_hardcoded_paths -v` | ❌ W0 | ⬜ pending |
| 1-02-01 | 02 | 1 | INFRA-02 | smoke | `pip install -r requirements.txt --dry-run` | ❌ W0 | ⬜ pending |
| 1-02-02 | 02 | 1 | INFRA-02 | unit | `pytest tests/test_phase1_infra.py::test_imports -v` | ❌ W0 | ⬜ pending |
| 1-03-01 | 03 | 2 | INFRA-03 | manual | Follow SETUP.md from scratch | N/A | ⬜ pending |
| 1-03-02 | 03 | 2 | INFRA-03 | smoke | `python -c "from Stockformermodel.Multitask_Stockformer_models import *; print('OK')"` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_phase1_infra.py` — stubs for INFRA-01, INFRA-02, INFRA-03
- [ ] `tests/conftest.py` — shared fixtures (project root path, config loading)
- [ ] `pytest` — install if not present in requirements.txt

*Wave 0 creates test file stubs before any implementation tasks run.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Developer follows SETUP.md and can import model within 30 minutes | INFRA-03 | Requires fresh environment; hard to automate fully | Clone repo, follow SETUP.md step-by-step, run smoke test command, verify completes in <30 min |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
