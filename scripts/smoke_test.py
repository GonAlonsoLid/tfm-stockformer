#!/usr/bin/env python3
"""
Smoke test for Phase 1 Infrastructure.
Verifies: config loads with required keys, core imports work, model instantiates.
Run from project root: python3 scripts/smoke_test.py
No training data required.
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

def test_config():
    import configparser
    config = configparser.ConfigParser()
    config.read('config/Multitask_Stock.conf')
    assert 'file' in config, "Config missing [file] section"
    required_keys = ['traffic', 'indicator', 'adj', 'adjgat', 'model', 'log',
                     'alpha_360_dir', 'output_dir', 'tensorboard_dir']
    for key in required_keys:
        assert key in config['file'], f"Config missing key: {key} — INFRA-01 incomplete"
    print("PASS: config loads and has all required keys")

def test_imports():
    import torch
    from pytorch_wavelets import DWT1DForward, DWT1DInverse
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    import pandas as pd
    print(f"PASS: imports OK — torch {torch.__version__}, pandas {pd.__version__}")

def test_model():
    import torch
    from Stockformermodel.Multitask_Stockformer_models import Stockformer
    # Stockformer(infea, outfea, outfea_class, outfea_regress, L, h, d, s, T1, T2, dev)
    # Using minimal realistic values: infea=362 (360 alpha + 2), outfea=h*d=128, L=2, h=1, d=128, s=1, T1=20, T2=2
    device = torch.device('cpu')
    model = Stockformer(infea=362, outfea=128, outfea_class=2, outfea_regress=1,
                        L=2, h=1, d=128, s=1, T1=20, T2=2, dev=device)
    print("PASS: Stockformer instantiates successfully")

if __name__ == '__main__':
    print("Running Phase 1 smoke tests...")
    test_config()
    test_imports()
    test_model()
    print("\nAll smoke tests passed. Environment is ready.")
