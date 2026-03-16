---
phase: 5
slug: portfolio-backtesting
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-03-14
---

# Phase 5 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (no config file — auto-discovery) |
| **Config file** | none — `python -m pytest tests/` from project root |
| **Quick run command** | `python -m pytest tests/test_backtest.py -x -q` |
| **Full suite command** | `python -m pytest tests/ -q` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `python -m pytest tests/test_backtest.py -x -q`
- **After every plan wave:** Run `python -m pytest tests/ -q`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|-----------|-------------------|-------------|--------|
| 5-01-01 | 01 | 0 | PORT-01, PORT-02, PORT-03, BACK-01, BACK-02, BACK-03 | unit (xfail stubs) | `python -m pytest tests/test_backtest.py -x -q` | ❌ W0 | ⬜ pending |
| 5-02-01 | 02 | 1 | PORT-01, PORT-02, PORT-03 | unit | `python -m pytest tests/test_backtest.py::test_top_k_selection tests/test_backtest.py::test_equal_weight tests/test_backtest.py::test_transaction_cost -x -q` | ❌ W0 | ⬜ pending |
| 5-02-02 | 02 | 1 | BACK-01, BACK-02, BACK-03 | unit | `python -m pytest tests/test_backtest.py::test_cumulative_return tests/test_backtest.py::test_performance_metrics tests/test_backtest.py::test_alpha_beta -x -q` | ❌ W0 | ⬜ pending |
| 5-03-01 | 03 | 2 | PORT-01, PORT-02, PORT-03, BACK-01, BACK-02, BACK-03 | integration | `python -m pytest tests/ -q` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_backtest.py` — xfail stubs for PORT-01, PORT-02, PORT-03, BACK-01, BACK-02, BACK-03

*Existing conftest.py fixtures are sufficient as a base — no new fixtures needed.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| equity_curve.png visually shows portfolio vs SPY lines correctly | BACK-01 | Chart visual correctness cannot be automated | Run `python scripts/run_backtest.py --output_dir output/Multitask_output_SP500_2018-2024 --top_k 10` and inspect the saved PNG |
| Script CLI runs end-to-end without error on real data | All | Integration with yfinance requires network access | Run CLI with real output_dir and verify all 3 output files are created |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
