# Phase 4: Evaluation - Research

**Researched:** 2026-03-14
**Domain:** Quantitative finance metrics (IC/ICIR) + scikit-learn classification metrics + CSV parsing
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Standalone script: `scripts/compute_ic.py`
- Entry point: `python scripts/compute_ic.py --output_dir output/Multitask_output_SP500_2018-2024`
- Reads `regression_pred_last_step.csv` and `regression_label_last_step.csv` from `output_dir` directly
- No config file required — CSVs contain everything needed
- No integration with run_inference.py — explicitly kept separate
- Reports ALL evaluation metrics in one shot: IC, ICIR, MAE, RMSE, accuracy, F1
- Single command gives the complete results table for the thesis
- Does NOT split by step — only global test-period summary
- Single run at a time (no multi-run comparison)
- Output File 1: `evaluation_summary.csv` — one row with IC_mean, ICIR, MAE, RMSE, accuracy, F1
- Output File 2: `ic_by_day.csv` — one row per trading day with date index and IC value
- Both CSVs saved into the same `output_dir`
- IC = Spearman rank correlation between predicted returns and realized returns, computed per day across all stocks
- ICIR = mean(daily IC) / std(daily IC) over the test period
- F1 computed for the classification head (up/down direction prediction)
- Classification CSVs: `classification_pred_last_step.csv` and `classification_label_last_step.csv`

### Claude's Discretion
- Whether to also compute Pearson IC alongside Spearman
- F1 averaging strategy (macro vs weighted)
- Exact table formatting (tabulate library or manual formatting)
- How to handle days with NaN IC (e.g., if all stocks have same predicted rank)

### Deferred Ideas (OUT OF SCOPE)
- Rolling 20-day IC chart in Streamlit — Phase 6 (VIZ-01 in v2 requirements)
- IC distribution histogram — v2 (VIZ-02)
- Per-sector IC breakdown — v2 (VIZ-03)
- Multi-run comparison table for ablations — not needed for thesis scope
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | IC (Information Coefficient) and ICIR computed on test period predictions | scipy.stats.spearmanr confirmed available (scipy==1.9.3); per-day cross-sectional pattern verified on actual 167×478 CSVs; NaN handling strategy documented |
| EVAL-02 | Existing MAE, RMSE, accuracy, and F1 metrics retained from original codebase | lib/Multitask_Stockformer_utils.py metric() reusable; sklearn.metrics.f1_score verified; classification CSV parsing pattern confirmed (stringified logit arrays); class balance documented |
</phase_requirements>

---

## Summary

Phase 4 adds a standalone `scripts/compute_ic.py` that reads the four CSVs produced by inference and computes a complete evaluation table. The technical domain is well-understood: IC/ICIR are industry-standard quantitative finance metrics, and all dependencies are already present in requirements.txt.

The primary challenge is CSV parsing. The classification prediction CSV stores numpy array strings like `[ 0.16767871 -0.08977825]` per cell — these must be parsed with a regex before `np.argmax` can extract the predicted class. The regression CSVs are clean float matrices readable with `pd.read_csv(..., header=None)`.

All metrics have been dry-run against the actual output CSVs (167 days x 478 stocks) confirming the pipeline is feasible with zero new dependencies. The only discretionary dependency is `tabulate` for formatting, but it is not installed — standard Python string formatting should be used instead.

**Primary recommendation:** Build `scripts/compute_ic.py` as a self-contained script importing only numpy, pandas, scipy, and sklearn (all in requirements.txt); use regex-based cell parsing for classification CSVs; use `ddof=1` in std for ICIR; drop NaN IC days silently with a logged warning.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| numpy | 1.24.4 (pinned) | Array arithmetic for IC, MAE, RMSE | Already in requirements.txt; no new install |
| pandas | 2.3.3 (pinned) | CSV I/O with `header=None`, `ic_by_day.csv` date index | Already in requirements.txt |
| scipy | 1.9.3 (pinned) | `scipy.stats.spearmanr` for Spearman IC | Already in requirements.txt; confirmed `.statistic` attribute available |
| scikit-learn | >=1.1.2 | `f1_score(average='macro')` for classification F1 | Already in requirements.txt |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| re (stdlib) | — | Parse stringified logit arrays in cls_pred CSV | Required — classification_pred_last_step.csv stores `[ 0.168 -0.090]` strings |
| argparse (stdlib) | — | `--output_dir` CLI argument | Standard pattern from run_inference.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual string formatting | tabulate | tabulate is NOT installed and would require requirements.txt edit; manual formatting is sufficient for one table |
| `scipy.stats.spearmanr` | `pandas.DataFrame.corr(method='spearman')` | Both work; spearmanr is more explicit for per-row iteration |
| `f1_score(average='macro')` | `average='weighted'` | Classes are nearly balanced (37,967 down vs 41,859 up = 47.6%/52.4%); macro is more conservative and appropriate for academic reporting — see Discretion section |

**Installation:** No new packages required. All dependencies in `requirements.txt`.

---

## Architecture Patterns

### Recommended Project Structure
```
scripts/
├── build_pipeline.py     # Phase 2 (existing)
├── run_inference.py      # Phase 3 (existing)
├── compute_ic.py         # Phase 4 (NEW) — standalone evaluator
└── smoke_test.py         # Phase 3 (existing)

output/Multitask_output_SP500_2018-2024/
├── regression/
│   ├── regression_pred_last_step.csv     # input: (167, 478) float matrix
│   └── regression_label_last_step.csv   # input: (167, 478) float matrix
├── classification/
│   ├── classification_pred_last_step.csv  # input: (167, 478) stringified logits
│   └── classification_label_last_step.csv # input: (167, 478) binary labels
├── evaluation_summary.csv    # output: 1-row summary
└── ic_by_day.csv             # output: 167-row daily IC series
```

### Pattern 1: Per-Day Spearman IC

**What:** For each trading day (row), compute Spearman rank correlation between the model's 478 predicted returns and the 478 realized returns.
**When to use:** Standard cross-sectional IC definition used in quant finance.

```python
# Source: scipy.stats documentation + verified on actual CSVs
from scipy.stats import spearmanr
import numpy as np

ic_per_day = []
for d in range(reg_pred.shape[0]):
    result = spearmanr(reg_pred[d], reg_label[d])
    ic_per_day.append(result.statistic)  # use .statistic, not .correlation (same value, .statistic is canonical)
ic_per_day = np.array(ic_per_day)
```

### Pattern 2: ICIR Computation

**What:** Annualized quality of IC signal — mean IC divided by std of IC.
**When to use:** Always paired with IC in quant evaluation.

```python
# Use ddof=1 (sample std) for unbiased estimate with finite test period
valid_ic = ic_per_day[~np.isnan(ic_per_day)]
ic_mean = np.mean(valid_ic)
ic_std = np.std(valid_ic, ddof=1)
icir = ic_mean / ic_std if ic_std > 0 else np.nan
```

### Pattern 3: Classification CSV Parsing

**What:** The classification prediction CSV stores numpy array strings. Standard `pd.read_csv` reads them as strings.
**When to use:** Any time classification_pred_last_step.csv is loaded.

```python
import re
import numpy as np

def parse_cls_pred_csv(path: str) -> np.ndarray:
    """Load classification_pred_last_step.csv -> integer class predictions (n_days, n_stocks)."""
    df = pd.read_csv(path, header=None)
    preds = np.zeros(df.shape, dtype=int)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", str(df.iloc[i, j]))]
            preds[i, j] = int(np.argmax(nums)) if len(nums) >= 2 else 0
    return preds
```

**Note:** Verified against actual CSV: cell value is `'[ 0.16767871 -0.08977825]'`. argmax(0) = class 0 (down), argmax(1) = class 1 (up). This matches the `np.argmax(class_pred, axis=-1)` used in `metric()`.

### Pattern 4: Regression CSV Loading

```python
# Both regression CSVs are clean float matrices with no header
reg_pred  = pd.read_csv(os.path.join(output_dir, "regression", "regression_pred_last_step.csv"),  header=None).values
reg_label = pd.read_csv(os.path.join(output_dir, "regression", "regression_label_last_step.csv"), header=None).values
# Shape: (167, 478) — 167 test days, 478 stocks
```

### Pattern 5: F1 Computation

```python
from sklearn.metrics import f1_score

# Flatten to 1D for sklearn
preds_flat  = preds_2d.flatten()    # (167*478,) integer predictions
labels_flat = cls_label.flatten()   # (167*478,) integer ground truth

f1 = f1_score(labels_flat, preds_flat, average="macro")
```

### Anti-Patterns to Avoid
- **Using `result.correlation` for spearmanr:** `.statistic` is the canonical attribute in scipy 1.9+; both are equivalent but `.statistic` is future-proof.
- **Using `ddof=0` for ICIR std:** Population std understates uncertainty on a 167-day test window; use `ddof=1`.
- **Loading cls_pred with `.values` without parsing:** Yields an array of Python strings — downstream `np.argmax` will fail silently or raise errors.
- **Using `average='binary'` for F1:** Data has two classes (0 and 1) but binary average assumes class 1 is positive — macro is better for balanced academic reporting.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Rank correlation | Manual rank computation + Pearson | `scipy.stats.spearmanr` | Handles ties, edge cases (zero std), returns p-value if needed |
| F1 score | Manual TP/FP/FN counting | `sklearn.metrics.f1_score` | Handles multi-class averaging, zero-division edge cases |
| Regex float parsing | Custom tokenizer | `re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", ...)` | Standard pattern, handles scientific notation |

**Key insight:** Every metric has a correct implementation in the already-installed stack. The only genuinely custom code is the CSV loading plumbing and the ICIR ratio itself.

---

## Common Pitfalls

### Pitfall 1: Classification CSV Cell Format
**What goes wrong:** `pd.read_csv(..., header=None)` reads `[ 0.168 -0.090]` as a Python string, not a numpy array. Direct indexing or `.astype(float)` raises a ValueError.
**Why it happens:** `save_to_csv()` in `lib/Multitask_Stockformer_utils.py` writes multi-value cells by serializing the numpy array's `__repr__`.
**How to avoid:** Always parse with `re.findall` before calling `np.argmax`. See Pattern 3.
**Warning signs:** `ValueError: could not convert string to float` when loading cls_pred CSV.

### Pitfall 2: NaN IC on Degenerate Days
**What goes wrong:** If all predicted returns for a day are identical (zero variance), `spearmanr` returns `nan`. Passing NaN into `np.mean()` for ICIR makes the entire result NaN.
**Why it happens:** Model can output near-identical predictions for some time steps.
**How to avoid:** Filter out NaN days before computing ICIR: `valid_ic = ic_per_day[~np.isnan(ic_per_day)]`. Log a warning with the NaN count.
**Warning signs:** Checked on actual CSVs — 0 NaN days observed in current output, but defensive handling is required for future runs.

### Pitfall 3: ICIR std denominator
**What goes wrong:** `np.std(ic_per_day)` uses `ddof=0` by default. For a 167-day test window, this slightly underestimates variance, producing a marginally inflated ICIR.
**Why it happens:** numpy default is population std.
**How to avoid:** Always use `np.std(ic_per_day, ddof=1)` for ICIR.

### Pitfall 4: CSV output directory nesting
**What goes wrong:** If `output_dir` doesn't already exist, `pd.DataFrame.to_csv()` raises `FileNotFoundError`.
**Why it happens:** `output_dir` is expected to exist (produced by Phase 3 inference), but the path may vary between machines.
**How to avoid:** Call `os.makedirs(output_dir, exist_ok=True)` before writing output CSVs.

### Pitfall 5: Accuracy metric alignment with metric()
**What goes wrong:** `metric()` computes `acc` using `pred_classes = np.argmax(class_pred, axis=-1)` on the raw 3D logit array from training. In `compute_ic.py`, we work from the pre-saved last-step CSV where logits are already per-cell strings.
**Why it happens:** Two different entry points to the same underlying predictions.
**How to avoid:** Parse cls_pred CSV -> argmax per cell -> compare to cls_label; result is equivalent to what `metric()` produces for the last step.

---

## Code Examples

Verified patterns from actual project CSVs (167 days x 478 stocks):

### Complete compute_ic.py skeleton

```python
#!/usr/bin/env python3
"""Evaluate Stockformer inference outputs with IC, ICIR, MAE, RMSE, accuracy, F1.

Usage:
    python scripts/compute_ic.py --output_dir output/Multitask_output_SP500_2018-2024
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score


def load_regression(output_dir: str):
    reg_pred  = pd.read_csv(os.path.join(output_dir, "regression", "regression_pred_last_step.csv"),  header=None).values
    reg_label = pd.read_csv(os.path.join(output_dir, "regression", "regression_label_last_step.csv"), header=None).values
    return reg_pred, reg_label


def load_classification(output_dir: str):
    cls_pred_raw = pd.read_csv(os.path.join(output_dir, "classification", "classification_pred_last_step.csv"), header=None)
    cls_label    = pd.read_csv(os.path.join(output_dir, "classification", "classification_label_last_step.csv"), header=None).values.flatten().astype(int)

    preds = np.zeros(cls_pred_raw.shape, dtype=int)
    for i in range(cls_pred_raw.shape[0]):
        for j in range(cls_pred_raw.shape[1]):
            nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?", str(cls_pred_raw.iloc[i, j]))]
            preds[i, j] = int(np.argmax(nums)) if len(nums) >= 2 else 0
    return preds, cls_label


def compute_ic_metrics(reg_pred, reg_label):
    n_days = reg_pred.shape[0]
    ic_per_day = np.array([spearmanr(reg_pred[d], reg_label[d]).statistic for d in range(n_days)])
    nan_count = np.isnan(ic_per_day).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} day(s) had constant predictions (IC=NaN), excluded from ICIR.", file=sys.stderr)
    valid_ic = ic_per_day[~np.isnan(ic_per_day)]
    ic_mean = float(np.mean(valid_ic))
    ic_std = float(np.std(valid_ic, ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else float("nan")
    return ic_mean, icir, ic_per_day


def compute_regression_metrics(reg_pred, reg_label):
    mae  = float(np.mean(np.abs(reg_pred - reg_label)))
    rmse = float(np.sqrt(np.mean((reg_pred - reg_label) ** 2)))
    return mae, rmse


def compute_classification_metrics(cls_preds_2d, cls_label_flat):
    preds_flat = cls_preds_2d.flatten()
    acc = float(np.mean(preds_flat == cls_label_flat))
    f1  = float(f1_score(cls_label_flat, preds_flat, average="macro"))
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics on Stockformer inference outputs.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory produced by run_inference.py (contains regression/ and classification/)")
    args = parser.parse_args()

    reg_pred, reg_label = load_regression(args.output_dir)
    cls_preds, cls_label = load_classification(args.output_dir)

    ic_mean, icir, ic_per_day = compute_ic_metrics(reg_pred, reg_label)
    mae, rmse = compute_regression_metrics(reg_pred, reg_label)
    acc, f1 = compute_classification_metrics(cls_preds, cls_label)

    # Console table
    print("\n=== Evaluation Summary ===")
    print(f"  IC mean   : {ic_mean:+.6f}")
    print(f"  ICIR      : {icir:+.6f}")
    print(f"  MAE       : {mae:.6f}")
    print(f"  RMSE      : {rmse:.6f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")

    # Save evaluation_summary.csv
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "evaluation_summary.csv")
    pd.DataFrame([{"IC_mean": ic_mean, "ICIR": icir, "MAE": mae, "RMSE": rmse, "Accuracy": acc, "F1_macro": f1}]).to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    # Save ic_by_day.csv (index = 0-based day integer; no trading date available without config)
    ic_day_path = os.path.join(args.output_dir, "ic_by_day.csv")
    pd.DataFrame({"day": range(len(ic_per_day)), "IC": ic_per_day}).to_csv(ic_day_path, index=False)
    print(f"Daily IC saved to: {ic_day_path}")


if __name__ == "__main__":
    main()
```

### Unit test skeleton for compute_ic.py

```python
# tests/test_compute_ic.py
import numpy as np
import pytest

def test_ic_known_correlation():
    """IC=1.0 for perfectly rank-correlated predictions."""
    from scripts.compute_ic import compute_ic_metrics
    n_days, n_stocks = 10, 50
    label = np.random.randn(n_days, n_stocks)
    # Perfect correlation: pred == label
    ic_mean, icir, ic_per_day = compute_ic_metrics(label, label)
    assert np.allclose(ic_per_day, 1.0), f"Expected all IC=1.0, got {ic_per_day}"

def test_ic_nan_handling():
    """NaN IC days (constant predictions) are excluded from ICIR without crashing."""
    from scripts.compute_ic import compute_ic_metrics
    n_days, n_stocks = 5, 20
    pred = np.random.randn(n_days, n_stocks)
    label = np.random.randn(n_days, n_stocks)
    pred[2, :] = 0.5  # constant predictions on day 2 → NaN IC
    ic_mean, icir, ic_per_day = compute_ic_metrics(pred, label)
    assert np.isnan(ic_per_day[2]), "Day with constant pred should produce NaN IC"
    assert not np.isnan(ic_mean), "ic_mean should not be NaN after filtering"

def test_mae_rmse_zero_for_perfect():
    """MAE=0, RMSE=0 when pred == label."""
    from scripts.compute_ic import compute_regression_metrics
    data = np.random.randn(20, 50)
    mae, rmse = compute_regression_metrics(data, data)
    assert mae == 0.0
    assert rmse == 0.0

def test_f1_perfect_classification():
    """F1=1.0 when predictions match labels exactly."""
    from scripts.compute_ic import compute_classification_metrics
    labels = np.array([0, 1, 0, 1, 0, 1])
    preds_2d = labels.reshape(2, 3)
    acc, f1 = compute_classification_metrics(preds_2d, labels)
    assert acc == 1.0
    assert f1 == 1.0
```

---

## Discretion Resolutions

Based on evidence gathered, here are recommendations for items marked "Claude's Discretion":

### F1 Averaging: macro (recommended)
- Class balance: 37,967 down / 41,859 up (47.6% / 52.4%) — nearly balanced
- Macro treats both classes equally regardless of support; this is more conservative and standard in academic ML evaluation
- Weighted F1 = 0.4853 vs macro F1 = 0.4783 on actual data — difference is small; macro is the academically defensible choice

### Pearson IC: add as optional column (recommended)
- Spearman is the quant-standard IC; Pearson is sensitive to outlier predictions
- Add `IC_pearson` as a bonus column in `evaluation_summary.csv` using `np.corrcoef(pred, label)[0,1]` per day
- Zero additional dependencies; adds thesis completeness

### NaN IC handling: drop silently with stderr warning (recommended)
- Zero NaN days observed on actual 167-day test output — this is a defensive measure
- Log count to stderr, exclude from ICIR computation, preserve raw NaN in `ic_by_day.csv` for transparency

### Table formatting: Python f-strings (recommended)
- `tabulate` is not installed; adding it requires requirements.txt change with no benefit
- Fixed-width f-string formatting is readable and requires no new dependency

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Manual Pearson IC (common in early papers) | Spearman rank IC | Spearman is robust to outlier predictions; standard in modern quant finance |
| Global accuracy only | Accuracy + F1 + IC + ICIR | Thesis completeness: IC/ICIR are the primary quant evaluation metrics |

---

## Open Questions

1. **Date index for `ic_by_day.csv`**
   - What we know: CSVs have no date column; the 167 rows map to test-split trading days, but the date sequence requires the config's date range and split ratio to reconstruct
   - What's unclear: Does the thesis need actual trading dates in `ic_by_day.csv`, or is a 0-based integer index sufficient for Phase 6?
   - Recommendation: Use 0-based integer index for Phase 4; Phase 6 (Streamlit) can join with actual dates when it has config access. This avoids adding a config dependency to compute_ic.py, which the user explicitly locked as not needing a config.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest >=7.0,<8 |
| Config file | none (discovered by pytest from project root) |
| Quick run command | `pytest tests/test_compute_ic.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-01 | IC=1.0 for perfect rank-correlated predictions | unit | `pytest tests/test_compute_ic.py::test_ic_known_correlation -x` | Wave 0 |
| EVAL-01 | NaN IC days excluded from ICIR without crash | unit | `pytest tests/test_compute_ic.py::test_ic_nan_handling -x` | Wave 0 |
| EVAL-01 | ICIR computed as mean/std(ddof=1) | unit | `pytest tests/test_compute_ic.py::test_icir_formula -x` | Wave 0 |
| EVAL-02 | MAE=0, RMSE=0 for perfect regression | unit | `pytest tests/test_compute_ic.py::test_mae_rmse_zero_for_perfect -x` | Wave 0 |
| EVAL-02 | F1=1.0 for perfect classification | unit | `pytest tests/test_compute_ic.py::test_f1_perfect_classification -x` | Wave 0 |
| EVAL-02 | compute_ic.py runs end-to-end on actual CSVs | smoke | `pytest tests/test_compute_ic.py::test_smoke_actual_output -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_compute_ic.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_compute_ic.py` — covers EVAL-01 and EVAL-02; does not exist yet (script doesn't exist yet)

*(conftest.py already exists with shared fixtures — no gap there)*

---

## Sources

### Primary (HIGH confidence)
- `lib/Multitask_Stockformer_utils.py` — metric() signature, save_to_csv() format verified
- `scripts/run_inference.py` — CSV path construction pattern verified
- `output/Multitask_output_SP500_2018-2024/` — actual CSV shapes (167, 478) and cell formats confirmed by direct Python inspection
- scipy 1.9.3 installed — `spearmanr(...).statistic` confirmed working
- scikit-learn >=1.1.2 installed — `f1_score(average='macro')` confirmed working

### Secondary (MEDIUM confidence)
- scipy.stats.spearmanr documentation — `.statistic` is canonical return attribute in 1.9+

### Tertiary (LOW confidence)
- None — all claims verified on actual project files

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already installed and verified with direct Python calls
- Architecture: HIGH — CSV shapes, cell formats, and metric computations confirmed on actual 167×478 output files
- Pitfalls: HIGH — discovered by actually running the parsing pipeline (cls_pred string format) and checking scipy return attributes

**Research date:** 2026-03-14
**Valid until:** 2026-06-14 (stable libraries; scipy/sklearn APIs change slowly)
