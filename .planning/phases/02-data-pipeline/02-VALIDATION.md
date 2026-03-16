---
phase: 2
slug: data-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-10
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | pytest.ini or pyproject.toml |
| **Quick run command** | `pytest tests/test_data_pipeline.py -x -q` |
| **Full suite command** | `pytest tests/ -q` |
| **Estimated runtime** | ~30 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_data_pipeline.py -x -q`
- **After every plan wave:** Run `pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 30 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 2-01-01 | 01 | 1 | DATA-01 | unit | `pytest tests/test_data_pipeline.py::test_download -x -q` | ❌ W0 | ⬜ pending |
| 2-01-02 | 01 | 1 | DATA-01 | unit | `pytest tests/test_data_pipeline.py::test_clean -x -q` | ❌ W0 | ⬜ pending |
| 2-02-01 | 02 | 2 | DATA-02 | unit | `pytest tests/test_data_pipeline.py::test_features -x -q` | ❌ W0 | ⬜ pending |
| 2-03-01 | 03 | 3 | DATA-03 | unit | `pytest tests/test_data_pipeline.py::test_normalize -x -q` | ❌ W0 | ⬜ pending |
| 2-03-02 | 03 | 3 | DATA-03 | unit | `pytest tests/test_data_pipeline.py::test_no_leakage -x -q` | ❌ W0 | ⬜ pending |
| 2-04-01 | 04 | 4 | DATA-04 | unit | `pytest tests/test_data_pipeline.py::test_arrays -x -q` | ❌ W0 | ⬜ pending |
| 2-04-02 | 04 | 4 | DATA-04 | unit | `pytest tests/test_data_pipeline.py::test_shapes -x -q` | ❌ W0 | ⬜ pending |
| 2-05-01 | 05 | 5 | DATA-05 | unit | `pytest tests/test_data_pipeline.py::test_struc2vec -x -q` | ❌ W0 | ⬜ pending |
| 2-05-02 | 05 | 5 | DATA-05 | integration | `pytest tests/test_data_pipeline.py::test_pipeline_e2e -x -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_data_pipeline.py` — stubs for DATA-01 through DATA-05
- [ ] `tests/conftest.py` — shared fixtures (sample OHLCV data, mock yfinance responses)
- [ ] pytest already installed from Phase 1 requirements.txt

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| yfinance rate limit handling | DATA-01 | Requires live network + 429 response | Run downloader for 500+ tickers and verify retry logic triggers |
| Struc2Vec embedding visual quality | DATA-05 | Embedding quality is subjective | Check embedding clusters align with known sector groupings |
| Parquet file integrity check | DATA-01 | End-to-end pipeline run needed | Load parquet, check no NaN trading days, verify date range |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
