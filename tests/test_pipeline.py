"""
Tests for scripts/build_pipeline.py — Phase 9 restructuring.

All tests are xfail(strict=False) until Plan 09-02 and 09-03 land.
Import convention: imports inside test bodies to prevent module-level ImportError
(project convention from Phase 02-01, 04-01, 05-01, 06-01, 08-01).
"""
import os
import pytest


@pytest.mark.xfail(strict=False, reason="PIPELINE_DIR not yet updated to scripts/sp500_pipeline/")
def test_pipeline_dir_resolves():
    """PIPELINE_DIR must point to an existing directory after the move."""
    import importlib
    import scripts.build_pipeline as bp
    importlib.reload(bp)
    assert os.path.isdir(bp.PIPELINE_DIR), (
        f"PIPELINE_DIR does not exist: {bp.PIPELINE_DIR}"
    )


@pytest.mark.xfail(strict=False, reason="_data_dir_from_config not yet added to build_pipeline.py")
def test_config_derives_data_dir():
    """_data_dir_from_config must extract data_dir from the SP500 config."""
    from scripts.build_pipeline import _data_dir_from_config
    data_dir, alpha_360_dir = _data_dir_from_config("config/Multitask_Stock_SP500.conf")
    assert data_dir.endswith("Stock_SP500_2018-01-01_2024-01-01"), (
        f"Expected data_dir ending in Stock_SP500_2018-01-01_2024-01-01, got: {data_dir}"
    )
    assert alpha_360_dir.endswith("features"), (
        f"Expected alpha_360_dir ending in 'features', got: {alpha_360_dir}"
    )


@pytest.mark.xfail(strict=False, reason="_alpha360_done not yet added to build_pipeline.py")
def test_alpha360_done_sentinel_false_when_empty(tmp_path):
    """_alpha360_done returns False when features/ dir is empty."""
    from scripts.build_pipeline import _alpha360_done
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    assert _alpha360_done(str(features_dir)) is False


@pytest.mark.xfail(strict=False, reason="_alpha360_done not yet added to build_pipeline.py")
def test_alpha360_done_sentinel_false_when_partial(tmp_path):
    """_alpha360_done returns False when features/ dir has fewer than 360 CSVs."""
    from scripts.build_pipeline import _alpha360_done
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for i in range(359):
        (features_dir / f"feature_{i:03d}.csv").touch()
    assert _alpha360_done(str(features_dir)) is False


@pytest.mark.xfail(strict=False, reason="_alpha360_done not yet added to build_pipeline.py")
def test_alpha360_done_sentinel_true_when_360(tmp_path):
    """_alpha360_done returns True when features/ dir has exactly 360 CSVs."""
    from scripts.build_pipeline import _alpha360_done
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    for i in range(360):
        (features_dir / f"feature_{i:03d}.csv").touch()
    assert _alpha360_done(str(features_dir)) is True
