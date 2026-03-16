# tfm-stockformer: Environment Setup Guide

Follow these steps to go from a fresh clone to a passing smoke test. No GPU is required for the smoke test; CPU is sufficient.

---

## Section 1 — Prerequisites

Ensure the following are installed before proceeding:

- **Python 3.9 or later** — verify with `python3 --version`
- **Git** — to clone the repository
- **CUDA-capable GPU** — NOT required for the smoke test; CPU is sufficient. GPU is only needed for full model training (see Section 3 note).

---

## Section 2 — Clone and create virtual environment

```bash
git clone <repo-url>
cd tfm-stockformer
python3 -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows
```

After activation, your prompt will show `(venv)` at the start.

---

## Section 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all required packages including `torch==2.8.0`, `PyWavelets==1.6.0`, `tensorboard>=2.14`, `pandas==2.3.3`, and `pytorch-wavelets==1.3.0`.

> **GPU training note:** For GPU training (CUDA 11.8), replace the torch line in `requirements.txt` with `torch==2.8.0+cu118` before running `pip install -r requirements.txt`.

---

## Section 4 — Verify the environment with the smoke test

```bash
python3 scripts/smoke_test.py
```

Expected output:

```
Running Phase 1 smoke tests...
PASS: config loads and has all required keys
PASS: imports OK — torch 2.8.0, pandas 2.3.3
PASS: Stockformer instantiates successfully

All smoke tests passed. Environment is ready.
```

This smoke test verifies three things:

1. **INFRA-01 — Config file integrity:** The `config/Multitask_Stock.conf` file loads and contains all required path keys (`alpha_360_dir`, `output_dir`, `tensorboard_dir`). These keys replace the original hardcoded `/root/autodl-tmp/` paths so the pipeline runs on any machine.
2. **INFRA-02 — Core package imports:** All core packages (`torch`, `pytorch_wavelets`, `tensorboard`, `numpy`, `pandas`) import correctly — confirming `requirements.txt` is complete and installable.
3. **INFRA-03 — Model instantiation:** The `Stockformer` model instantiates successfully without requiring training data. This confirms that `PyTorch` and `pytorch-wavelets` are correctly installed and compatible.

---

## Section 5 — Running the test suite (optional)

```bash
pytest tests/test_phase1_infra.py -v
```

This runs 9 automated tests covering path portability (INFRA-01), import checks (INFRA-02), and smoke test existence (INFRA-03).

> **Note:** Full training tests require data files from Phase 2. The Phase 1 test suite runs without any data.

---

## Section 6 — Project structure

The key entry points and configuration are:

- `config/Multitask_Stock.conf` — All path and hyperparameter configuration. All file I/O paths are defined here under the `[file]` section and passed through `argparse` to the training script. Edit this file to point paths to your local data directory.
- `MultiTask_Stockformer_train.py` — Main training entry point. Run with `python3 MultiTask_Stockformer_train.py --config config/Multitask_Stock.conf` after acquiring training data.
- `data_processing_script/` — Offline data preprocessing scripts run before training. These are Phase 2 deliverables; they accept `--data_dir` and `--output_dir` CLI arguments matching the config defaults.
- `.planning/` — Project planning documents including the ROADMAP, REQUIREMENTS, and per-phase research and execution summaries.

---

## Section 7 — Troubleshooting

**1. `ModuleNotFoundError: No module named 'pywt'`**

PyWavelets is not installed. Run:

```bash
pip install PyWavelets==1.6.0
```

This is a required transitive dependency of `pytorch-wavelets`. It should be installed automatically via `requirements.txt`; if it was not, run the command above.

**2. `ModuleNotFoundError: No module named 'tensorboard'`**

`tensorboard` is not installed. Run:

```bash
pip install tensorboard
```

`torch.utils.tensorboard` requires the standalone `tensorboard` package, which is not bundled with PyTorch.

**3. `FileNotFoundError: /root/autodl-tmp/...`**

The config path fix from Phase 1 Plan 01 was not applied, or you are using an old version of `config/Multitask_Stock.conf`. Check that `config/Multitask_Stock.conf` contains the `alpha_360_dir` key in the `[file]` section:

```ini
[file]
alpha_360_dir = ./data/Stock_CN_2021-06-04_2024-01-30/Alpha_360_2021-06-04_2024-01-30
output_dir = ./output/Multitask_output_2021-06-04_2024-01-30
tensorboard_dir = ./runs/Multitask_Stockformer/Stock_CN_2021-06-04_2024-01-30
```

If these keys are missing, pull the latest version from the repository.
