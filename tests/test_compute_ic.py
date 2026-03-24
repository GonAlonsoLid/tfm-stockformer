"""
Unit tests for scripts/compute_ic.py — written BEFORE implementation (TDD Red phase).

All tests that require the not-yet-existing scripts/compute_ic.py are protected with
xfail(strict=False) so the suite collects cleanly and does not fail CI until Plan 02
delivers the implementation.

Covers requirements: EVAL-01 (IC/ICIR metrics), EVAL-02 (MAE, RMSE, accuracy, F1).
"""
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# EVAL-01: Information Coefficient metrics
# ---------------------------------------------------------------------------


def test_ic_known_correlation():
    """compute_ic_metrics(label, label) must return ic_per_day all equal to 1.0."""
    try:
        from scripts.compute_ic import compute_ic_metrics
    except ImportError:
        pytest.skip("scripts/compute_ic.py not found — Plan 02 not yet executed")

    np.random.seed(0)
    n_days, n_stocks = 10, 50
    label = np.random.randn(n_days, n_stocks)

    ic_mean, icir, ic_per_day = compute_ic_metrics(label, label)

    assert np.allclose(ic_per_day, 1.0), (
        f"Expected perfect rank correlation (1.0) for all days; got {ic_per_day}"
    )


def test_ic_nan_handling():
    """
    When one day's predictions are constant (zero variance), ic_per_day[that_day]
    must be NaN; ic_mean must NOT be NaN (NaN days are excluded before averaging).
    """
    try:
        from scripts.compute_ic import compute_ic_metrics
    except ImportError:
        pytest.skip("scripts/compute_ic.py not found — Plan 02 not yet executed")

    np.random.seed(1)
    n_days, n_stocks = 5, 20
    pred = np.random.randn(n_days, n_stocks)
    label = np.random.randn(n_days, n_stocks)

    # Day 2: constant predictions -> Spearman correlation is undefined (NaN)
    pred[2, :] = 0.5

    ic_mean, icir, ic_per_day = compute_ic_metrics(pred, label)

    assert np.isnan(ic_per_day[2]), (
        f"Expected NaN for day 2 (constant predictions); got {ic_per_day[2]}"
    )
    assert not np.isnan(ic_mean), (
        f"ic_mean must not be NaN when NaN days are excluded; got {ic_mean}"
    )


def test_icir_formula():
    """
    ICIR = mean(ic_valid) / std(ic_valid, ddof=1).
    Verified on the returned values: assert icir == approx(ic_mean / std(valid_ic, ddof=1)).
    """
    try:
        from scripts.compute_ic import compute_ic_metrics
    except ImportError:
        pytest.skip("scripts/compute_ic.py not found — Plan 02 not yet executed")

    np.random.seed(2)
    n_days, n_stocks = 10, 50
    pred = np.random.randn(n_days, n_stocks)
    label = np.random.randn(n_days, n_stocks)

    ic_mean, icir, ic_per_day = compute_ic_metrics(pred, label)

    valid_ic = ic_per_day[~np.isnan(ic_per_day)]
    assert len(valid_ic) > 1, "Need at least 2 valid IC values to compute ICIR"

    expected_icir = float(np.mean(valid_ic)) / float(np.std(valid_ic, ddof=1))

    assert icir == pytest.approx(expected_icir, rel=1e-5), (
        f"ICIR mismatch: expected {expected_icir}, got {icir}"
    )


# ---------------------------------------------------------------------------
# EVAL-02: Regression metrics
# ---------------------------------------------------------------------------


def test_mae_rmse_zero_for_perfect():
    """compute_regression_metrics(data, data) must return (mae=0.0, rmse=0.0)."""
    try:
        from scripts.compute_ic import compute_regression_metrics
    except ImportError:
        pytest.skip("scripts/compute_ic.py not found — Plan 02 not yet executed")

    np.random.seed(3)
    data = np.random.randn(20, 50)

    mae, rmse = compute_regression_metrics(data, data)

    assert mae == pytest.approx(0.0, abs=1e-10), f"Expected MAE=0.0 for perfect predictions; got {mae}"
    assert rmse == pytest.approx(0.0, abs=1e-10), f"Expected RMSE=0.0 for perfect predictions; got {rmse}"


# ---------------------------------------------------------------------------
# EVAL-02: Classification metrics
# ---------------------------------------------------------------------------


def test_f1_perfect_classification():
    """
    compute_classification_metrics(preds_2d, labels_flat) must return (acc=1.0, f1=1.0)
    when predictions match labels exactly.
    """
    try:
        from scripts.compute_ic import compute_classification_metrics
    except ImportError:
        pytest.skip("scripts/compute_ic.py not found — Plan 02 not yet executed")

    labels = np.array([0, 1, 0, 1, 0, 1])
    preds_2d = labels.reshape(2, 3)

    acc, f1 = compute_classification_metrics(preds_2d, labels)

    assert acc == pytest.approx(1.0), f"Expected accuracy=1.0; got {acc}"
    assert f1 == pytest.approx(1.0), f"Expected F1=1.0; got {f1}"


# ---------------------------------------------------------------------------
# Integration smoke test (requires Plan 02 deliverable + inference outputs)
# ---------------------------------------------------------------------------


def test_smoke_actual_output(tmp_path):
    """
    Imports main() from scripts.compute_ic and verifies it exits cleanly.
    Allowed to xfail if output CSVs are not present or compute_ic.py does not exist.
    This is the integration gate for Plan 02's deliverable.
    """
    try:
        from scripts.compute_ic import main
    except ImportError:
        pytest.skip("scripts/compute_ic.py not found — Plan 02 not yet executed")

    import os

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "output",
        "Multitask_output_SP500_2018-2024",
    )

    if not os.path.isdir(output_dir):
        pytest.xfail(f"Inference output dir not found: {output_dir}")

    # If output dir exists, main() should run to completion without raising
    main(output_dir=output_dir)
