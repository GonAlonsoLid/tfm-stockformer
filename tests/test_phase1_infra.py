"""
Phase 1 Infrastructure tests.
INFRA-01: No hardcoded /root/autodl-tmp/ paths remain in Python source files.
INFRA-02: Core imports succeed after requirements install.
INFRA-03: Smoke test passes (tested via scripts/smoke_test.py).
"""
import os
import glob
import importlib
import pytest


# --- INFRA-01: Path portability ---

def test_no_hardcoded_paths(project_root):
    """All /root/autodl-tmp/ occurrences must be removed from .py files."""
    py_files = glob.glob(os.path.join(project_root, '**', '*.py'), recursive=True)
    # Exclude .planning/, venv/, tests/ (test files contain the string in assertions),
    # and .ipynb_checkpoints/ (auto-generated Jupyter cache files)
    py_files = [f for f in py_files if '.planning' not in f and 'venv' not in f
                and '/tests/' not in f and '.ipynb_checkpoints' not in f]
    offenders = []
    for path in py_files:
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            for lineno, line in enumerate(fh, 1):
                if '/root/autodl-tmp' in line:
                    offenders.append(f"{path}:{lineno}: {line.rstrip()}")
    assert not offenders, "Hardcoded paths found:\n" + "\n".join(offenders)


def test_config_has_new_keys(config):
    """Config must expose the three new path keys added for INFRA-01."""
    assert 'file' in config, "Config missing [file] section"
    for key in ['alpha_360_dir', 'output_dir', 'tensorboard_dir']:
        assert key in config['file'], f"Config [file] missing key: {key}"


def test_config_keys_are_relative(config):
    """All path values in config [file] section must not start with /root/."""
    for key, value in config['file'].items():
        assert not value.startswith('/root/'), \
            f"Config key '{key}' still has absolute /root/ path: {value}"


# --- INFRA-02: Requirements completeness ---

def test_torch_importable():
    """torch must import without error (validates requirements.txt includes torch)."""
    import torch
    assert torch.__version__, "torch imported but has no __version__"


def test_pywavelets_importable():
    """PyWavelets (pywt) must import — required by pytorch-wavelets."""
    import pywt
    assert hasattr(pywt, 'dwt'), "pywt imported but missing expected attribute"


def test_tensorboard_importable():
    """tensorboard must import via torch.utils.tensorboard."""
    from torch.utils.tensorboard import SummaryWriter
    assert SummaryWriter is not None


def test_pytorch_wavelets_importable():
    """pytorch_wavelets DWT must import and be functional."""
    from pytorch_wavelets import DWT1DForward
    assert DWT1DForward is not None


def test_pandas_applymap_removed(project_root):
    """results_data_processing.py must not use deprecated .applymap()."""
    target = os.path.join(
        project_root,
        'data_processing_script',
        'stockformer_input_data_processing',
        'results_data_processing.py'
    )
    with open(target, 'r') as fh:
        content = fh.read()
    assert 'applymap' not in content, \
        "results_data_processing.py still uses deprecated .applymap() — replace with .map()"


# --- INFRA-03: Smoke test exists ---

def test_smoke_test_exists(project_root):
    """scripts/smoke_test.py must exist as the SETUP.md onboarding verification command."""
    path = os.path.join(project_root, 'scripts', 'smoke_test.py')
    assert os.path.isfile(path), "scripts/smoke_test.py not found — INFRA-03 incomplete"
