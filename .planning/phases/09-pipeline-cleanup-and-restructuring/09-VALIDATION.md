---
phase: 9
slug: pipeline-cleanup-and-restructuring
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-17
---

# Phase 9 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing) |
| **Config file** | none — run from project root |
| **Quick run command** | `pytest tests/ -x -q` |
| **Full suite command** | `pytest tests/ -v` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x -q`
- **After every plan wave:** Run `pytest tests/ -v`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** ~10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 9-01-01 | 01 | 0 | Wave 0 | unit stub | `pytest tests/test_pipeline.py -x -q` | ❌ W0 | ⬜ pending |
| 9-01-02 | 01 | 0 | Wave 0 | unit stub | `pytest tests/test_download_ohlcv.py -x -q` | ❌ W0 | ⬜ pending |
| 9-02-01 | 02 | 1 | Move sp500_pipeline | unit | `pytest tests/test_pipeline.py -x -q` | ✅ | ⬜ pending |
| 9-02-02 | 02 | 1 | PIPELINE_DIR update | unit | `pytest tests/test_pipeline.py -x -q` | ✅ | ⬜ pending |
| 9-02-03 | 02 | 1 | Ticker fallback | unit | `pytest tests/test_download_ohlcv.py -x -q` | ✅ | ⬜ pending |
| 9-02-04 | 02 | 1 | build_pipeline --config | unit | `pytest tests/test_pipeline.py -x -q` | ✅ | ⬜ pending |
| 9-02-05 | 02 | 1 | Step 5 integration | unit | `pytest tests/test_pipeline.py -x -q` | ✅ | ⬜ pending |
| 9-02-06 | 02 | 1 | Legacy deletion | smoke | `pytest tests/ -x -q` | ✅ | ⬜ pending |
| 9-02-07 | 02 | 1 | README update | manual | see manual table | N/A | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_pipeline.py` — stubs for PIPELINE_DIR update, config derivation, `_alpha360_done()` sentinel
- [ ] `tests/test_download_ohlcv.py` — stubs for ticker local-file fallback (file exists and file absent cases)

*Existing `tests/conftest.py` provides shared fixtures; no new fixtures required for these unit tests.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| README Quick Start commands are accurate and complete | End-to-end docs | Command correctness requires human judgment | Read README.md Quick Start section; verify all 7 steps are present, order is correct, GPU note is included |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
