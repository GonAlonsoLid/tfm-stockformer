"""Shared fixtures for Phase 1 infrastructure tests."""
import pytest
import os
import sys
import configparser

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

@pytest.fixture
def project_root():
    return PROJECT_ROOT

@pytest.fixture
def config(project_root):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(project_root, 'config', 'Multitask_Stock.conf'))
    return cfg
