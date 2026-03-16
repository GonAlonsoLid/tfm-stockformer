---
phase: 6
slug: interface
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-16
---

# Phase 6 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x |
| **Config file** | `tests/conftest.py` (existing) |
| **Quick run command** | `python -m pytest tests/test_app.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_app.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 6-01-01 | 01 | 0 | UI-01 | unit stub | `python -m pytest tests/test_app.py -x -q` | ❌ W0 | ⬜ pending |
| 6-01-02 | 01 | 1 | UI-01 | unit | `python -m pytest tests/test_app.py::test_app_imports -x -q` | ❌ W0 | ⬜ pending |
| 6-01-03 | 01 | 1 | UI-01 | unit | `python -m pytest tests/test_app.py::test_sidebar_defaults -x -q` | ❌ W0 | ⬜ pending |
| 6-02-01 | 02 | 1 | UI-02 | unit | `python -m pytest tests/test_app.py::test_equity_chart_shape -x -q` | ❌ W0 | ⬜ pending |
| 6-02-02 | 02 | 1 | UI-02 | unit | `python -m pytest tests/test_app.py::test_equity_chart_starts_at_one -x -q` | ❌ W0 | ⬜ pending |
| 6-02-03 | 02 | 1 | UI-03 | unit | `python -m pytest tests/test_app.py::test_metrics_table_columns -x -q` | ❌ W0 | ⬜ pending |
| 6-02-04 | 02 | 1 | UI-04 | unit | `python -m pytest tests/test_app.py::test_heatmap_zmid -x -q` | ❌ W0 | ⬜ pending |
| 6-02-05 | 02 | 1 | UI-04 | unit | `python -m pytest tests/test_app.py::test_heatmap_top_k_filter -x -q` | ❌ W0 | ⬜ pending |
| 6-03-01 | 03 | 2 | UI-01..04 | integration | `python -m pytest tests/test_app.py -q` | ❌ W0 | ⬜ pending |

*Actual test function names (7 total): `test_app_imports`, `test_sidebar_defaults`, `test_equity_chart_shape`, `test_equity_chart_starts_at_one`, `test_metrics_table_columns`, `test_heatmap_zmid`, `test_heatmap_top_k_filter`*

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_app.py` — stubs for UI-01, UI-02, UI-03, UI-04
- [ ] Streamlit + Plotly installed: `pip install "streamlit>=1.35,<=1.50" plotly`

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Live stdout log scrolls during pipeline run | UI-01 | Requires browser interaction + real subprocess | Launch `streamlit run app.py`, click Run, observe scrolling spinner log |
| Date range picker filters displayed equity curve | UI-01 | Requires browser interaction | Select different date range, verify chart updates to show only the selected window |
| Empty state shows placeholder instructions | UI-01 | Requires browser render | Launch fresh app, verify main area shows instructions before running |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
