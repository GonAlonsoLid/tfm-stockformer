# Testing Patterns

**Analysis Date:** 2026-03-09

## Test Framework

**Runner:**
- None detected. No `pytest`, `unittest`, `nose`, or any other testing framework is configured or installed in `requirements.txt`.
- No `pytest.ini`, `setup.cfg`, `pyproject.toml`, `.flake8`, or any test configuration file is present.

**Assertion Library:**
- None configured.

**Run Commands:**
```bash
# No test commands defined. There are no test files in this project.
```

## Test File Organization

**Location:**
- No test files exist. Running `find . -name "test_*.py" -o -name "*_test.py"` returns no results.

**Naming:**
- No test naming convention established.

**Structure:**
```
# No test directory structure exists.
```

## Test Structure

**Suite Organization:**
- Not applicable. No test suites exist.

**Patterns:**
- No setup, teardown, or assertion patterns.

## Mocking

**Framework:**
- None.

**Patterns:**
- No mocking patterns exist in the codebase.

**What to Mock:**
- Not established.

## Fixtures and Factories

**Test Data:**
- No test fixtures or factory functions exist.
- Real data files (`.npz`, `.npy`, `.csv`) are used directly during training runs; there is no synthetic data generation for testing purposes.

**Location:**
- No fixture directory.

## Coverage

**Requirements:**
- None enforced. No coverage configuration or reporting.

**View Coverage:**
```bash
# Not configured.
```

## Test Types

**Unit Tests:**
- None exist.

**Integration Tests:**
- None exist.

**E2E Tests:**
- None exist.

## Validation Approach (Substitute for Testing)

Although no automated tests exist, the project uses several manual validation patterns:

**Training-Time Evaluation:**
- The `res()` function in `MultiTask_Stockformer_train.py` computes per-step and average metrics (acc, MAE, RMSE, MAPE) on the validation split after each epoch.
- The `test_res()` function applies the same metric computation on the held-out test split.
- Both functions are defined in `MultiTask_Stockformer_train.py` and use `metric()` from `lib/Multitask_Stockformer_utils.py`.

**Metric Computation:**
```python
# From lib/Multitask_Stockformer_utils.py
def metric(reg_pred, reg_label, class_pred, class_label):
    # Computes acc (classification), mae, rmse, mape (regression)
    return acc, mae, rmse, mape
```

**Log-Based Verification:**
- All metric results are written to a log file via `log_string()` and also printed to stdout.
- TensorBoard scalars are recorded for `Val/Average_Accuracy`, `Val/Average_MAE`, `Val/Average_RMSE`, `Val/Average_MAPE`, and `training loss`.

**Data Processing Validation:**
- Data preprocessing scripts print confirmation messages after each step (e.g., `print('Data read successfully.')`, `print('Flow array saved as npz successfully.')`) but do not assert correctness of outputs.

**Notebook-Based Exploration:**
- Jupyter notebooks in `data_processing_script/volume_and_price_factor_construction/` serve as interactive validation for factor construction and backtest analysis:
  - `1_stock_data_consolidation.ipynb`
  - `2_data_preprocessing.ipynb`
  - `3_qlib_factor_construction.ipynb`
  - `5_factor_verification.ipynb`
  - `Backtest/Backtest.ipynb`

## Recommendations for Adding Tests

If tests are to be introduced, the following areas have the highest value and lowest coupling:

1. **`lib/Multitask_Stockformer_utils.py`** - `metric()`, `masked_mae()`, `disentangle()`, and `generate_temporal_embeddings()` are pure or near-pure functions that can be unit tested with synthetic numpy arrays.
2. **`lib/graph_utils.py`** - `laplacian()` and `loadGraph()` are small and testable with a mock adjacency matrix.
3. **`data_processing_script/stockformer_input_data_processing/data_Interception.py`** - `filter_date_range()` and `save_filtered_data()` accept file paths and can be tested with temporary CSV fixtures.
4. **`Stockformermodel/Multitask_Stockformer_models.py`** - Model `forward()` passes can be smoke-tested with random tensors of correct shapes to verify no runtime shape errors.

Suggested framework: `pytest` with `numpy.testing` for array assertions. Install with `pip install pytest`.

---

*Testing analysis: 2026-03-09*
