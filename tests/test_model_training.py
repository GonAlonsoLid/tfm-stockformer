"""Phase 3 — Model Training test stubs (Wave 0).

All tests are marked xfail(strict=False) so the suite collects and passes
before implementation lands in plans 03-02 and 03-03.

Covers:
  MODEL-01 — config validation and dataset loading
  MODEL-02 — Stockformer forward pass and inference script
"""
import pytest
import os
import sys
import configparser


# ── MODEL-01: Configuration tests ─────────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="config/Multitask_Stock_SP500.conf not yet created")
def test_config_file_exists(project_root):
    """Assert that config/Multitask_Stock_SP500.conf exists on disk."""
    config_path = os.path.join(project_root, "config", "Multitask_Stock_SP500.conf")
    assert os.path.exists(config_path), (
        f"Config file not found: {config_path}"
    )


@pytest.mark.xfail(strict=False, reason="config/Multitask_Stock_SP500.conf not yet created")
def test_config_fields_present(project_root):
    """Load Multitask_Stock_SP500.conf and assert all four INI sections and required keys exist.

    ConfigParser normalises key names to lowercase, so T1/T2 are stored as t1/t2.
    """
    config_path = os.path.join(project_root, "config", "Multitask_Stock_SP500.conf")
    cfg = configparser.ConfigParser()
    cfg.read(config_path)

    required_sections = ["file", "data", "train", "param"]
    for section in required_sections:
        assert section in cfg, f"Missing INI section: [{section}]"

    required_keys = {
        "file": [
            "traffic", "indicator", "adj", "adjgat", "model", "log",
            "alpha_360_dir", "output_dir", "tensorboard_dir",
        ],
        "data": ["t1", "t2", "train_ratio", "val_ratio", "test_ratio"],
        "train": ["cuda", "max_epoch", "batch_size", "learning_rate", "seed"],
        "param": ["layers", "heads", "dims", "samples", "wave", "level"],
    }
    for section, keys in required_keys.items():
        for key in keys:
            assert key in cfg[section], (
                f"Missing key '{key}' in section [{section}]"
            )


# ── MODEL-01/02: Dataset smoke test ───────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="requires Phase 2 data on disk")
def test_dataset_loads(project_root):
    """Smoke test: assert config file exists as a placeholder for full dataset loading.

    Full dataset load requires Phase 2 flow.npz and feature CSVs on disk.
    This stub will be expanded in plan 03-02 once data artefacts are available.
    """
    config_path = os.path.join(project_root, "config", "Multitask_Stock_SP500.conf")
    assert os.path.exists(config_path), (
        f"Config file not found — data pipeline must run before dataset can load: {config_path}"
    )


# ── MODEL-02: Stockformer forward pass ────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="requires full model setup verification")
def test_stockformer_forward_pass():
    """Assert Stockformer forward pass returns a tuple with two tensors of correct shape.

    Imports are inside the test body so the module loads without errors
    even when the model package is not yet on the import path.
    """
    import torch
    from Stockformermodel.Multitask_Stockformer_models import Stockformer  # type: ignore

    device = torch.device("cpu")
    model = Stockformer(
        infea=362,
        hidden=128,
        outfea_class=2,
        outfea_regress=1,
        L=2,
        h=1,
        d=128,
        s=1,
        T1=20,
        T2=2,
        device=device,
    ).to(device)
    model.eval()

    batch, N, T = 2, 5, 22  # T = T1 + T2
    x = torch.randn(batch, N, T)
    with torch.no_grad():
        output = model(x)

    assert isinstance(output, tuple), "Stockformer must return a tuple"
    assert len(output) == 2, "Stockformer output tuple must have two elements"
    for tensor in output:
        assert isinstance(tensor, torch.Tensor), "Each output element must be a torch.Tensor"


# ── MODEL-02: Inference script tests ──────────────────────────────────────────

@pytest.mark.xfail(strict=False, reason="requires run_inference.py to be created")
def test_inference_script_exists(project_root):
    """Assert that scripts/run_inference.py exists on disk."""
    script_path = os.path.join(project_root, "scripts", "run_inference.py")
    assert os.path.exists(script_path), (
        f"Inference script not found: {script_path}"
    )


@pytest.mark.xfail(strict=False, reason="requires run_inference.py to be created")
def test_inference_script_args(project_root):
    """Assert that scripts/run_inference.py accepts --config and --checkpoint arguments."""
    script_path = os.path.join(project_root, "scripts", "run_inference.py")
    assert os.path.exists(script_path), (
        f"Inference script not found — cannot check args: {script_path}"
    )
    with open(script_path, "r", encoding="utf-8") as fh:
        source = fh.read()
    assert "--config" in source, "run_inference.py must accept --config argument"
    assert "--checkpoint" in source, "run_inference.py must accept --checkpoint argument"
