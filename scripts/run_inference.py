"""Standalone inference script for the Stockformer model.

Loads a trained checkpoint and produces prediction CSVs for the test period.
Does NOT import from MultiTask_Stockformer_train.py (that module runs side
effects — log file creation, SummaryWriter — at import time).

Usage:
    python scripts/run_inference.py --config config/Multitask_Stock_SP500.conf
    python scripts/run_inference.py --config config/Multitask_Stock_SP500.conf \\
        --checkpoint cpt/STOCK/saved_model_Multitask_SP500_2018-2024
"""

import argparse
import configparser
import math
import os
import sys

import numpy as np
import torch

# Ensure project root is on sys.path so lib/ and Stockformermodel/ are importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Safe lib imports — no module-level side effects in these modules
from lib.Multitask_Stockformer_utils import (  # noqa: E402
    StockDataset,
    metric,
    save_to_csv,
    log_string,
)
from lib.graph_utils import loadGraph  # noqa: E402
from Stockformermodel.Multitask_Stockformer_models import Stockformer  # noqa: E402


def main():
    # ── Step 1: Two-phase argument parsing ───────────────────────────────────
    # Phase 1: capture only --config and (optionally) --checkpoint
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, required=True,
                            help="Path to INI configuration file (required)")
    pre_parser.add_argument("--checkpoint", type=str, default=None,
                            help="Path to model checkpoint to load (overrides config model path)")
    pre_args, _ = pre_parser.parse_known_args()

    # Phase 1b: read config file
    config = configparser.ConfigParser()
    config.read(pre_args.config)

    # Phase 2: full parser with config values as defaults
    parser = argparse.ArgumentParser(
        description="Run Stockformer inference on the test split and save prediction CSVs."
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to INI configuration file (required)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint to load (overrides config model path)")

    # [train] section
    parser.add_argument("--cuda", type=str,
                        default=config["train"]["cuda"])
    parser.add_argument("--batch_size", type=int,
                        default=int(config["train"]["batch_size"]))

    # [data] section
    parser.add_argument("--T1", type=int,
                        default=int(config["data"]["t1"]))
    parser.add_argument("--T2", type=int,
                        default=int(config["data"]["t2"]))
    parser.add_argument("--train_ratio", type=float,
                        default=float(config["data"]["train_ratio"]))
    parser.add_argument("--val_ratio", type=float,
                        default=float(config["data"]["val_ratio"]))
    parser.add_argument("--test_ratio", type=float,
                        default=float(config["data"]["test_ratio"]))

    # [param] section
    parser.add_argument("--L", type=int,
                        default=int(config["param"]["layers"]))
    parser.add_argument("--h", type=int,
                        default=int(config["param"]["heads"]))
    parser.add_argument("--d", type=int,
                        default=int(config["param"]["dims"]))
    parser.add_argument("--j", type=int,
                        default=int(config["param"]["level"]))
    parser.add_argument("--s", type=float,
                        default=float(config["param"]["samples"]))
    parser.add_argument("--w", type=str,
                        default=config["param"]["wave"])

    # [file] section
    parser.add_argument("--traffic_file",
                        default=config["file"]["traffic"])
    parser.add_argument("--indicator_file",
                        default=config["file"]["indicator"])
    parser.add_argument("--adj_file",
                        default=config["file"]["adj"])
    parser.add_argument("--adjgat_file",
                        default=config["file"]["adjgat"])
    parser.add_argument("--model_file",
                        default=config["file"]["model"])
    parser.add_argument("--alpha_360_dir",
                        default=config["file"]["alpha_360_dir"])
    parser.add_argument("--output_dir",
                        default=config["file"]["output_dir"])

    args = parser.parse_args()

    # If --checkpoint was provided, override model_file
    if args.checkpoint is not None:
        args.model_file = args.checkpoint

    # ── Step 2: Device setup ─────────────────────────────────────────────────
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # ── Step 3: Verify Phase 2 outputs exist ─────────────────────────────────
    required_paths = {
        "traffic_file": args.traffic_file,
        "indicator_file": args.indicator_file,
        "adjgat_file": args.adjgat_file,
        "alpha_360_dir": args.alpha_360_dir,
    }
    missing = []
    for name, path in required_paths.items():
        if not os.path.exists(path):
            missing.append(f"  {name}: {path}")
    if missing:
        print(
            "ERROR: The following Phase 2 data artifacts are missing.\n"
            "Run the data pipeline (scripts/build_pipeline.py) first.\n"
            + "\n".join(missing)
        )
        sys.exit(1)

    if not os.path.exists(args.model_file):
        print(
            f"ERROR: Checkpoint not found: {args.model_file}\n"
            "Train the model first with MultiTask_Stockformer_train.py, "
            "or pass --checkpoint <path> to specify an existing checkpoint."
        )
        sys.exit(1)

    # ── Step 4: Load graph adjacency matrix ─────────────────────────────────
    print("Loading graph adjacency matrix...")
    adjgat = loadGraph(args)
    adjgat = torch.from_numpy(adjgat).float().to(device)

    # ── Step 5: Load test dataset ────────────────────────────────────────────
    print("Loading test dataset...")
    test_dataset = StockDataset(args, mode="test")

    # Extract numpy arrays directly (same pattern as MultiTask_Stockformer_train.py)
    testXL = test_dataset.XL
    testXH = test_dataset.XH
    testXC = test_dataset.indicator_X
    testTE = test_dataset.TE
    testY = test_dataset.Y
    testYC = test_dataset.indicator_Y
    bonus_testX = test_dataset.bonus_X

    # ── Step 6: Build model and load checkpoint ───────────────────────────────
    infeature = test_dataset.infea
    print(f"Building Stockformer model (infeature={infeature})...")
    model = Stockformer(
        infeature,
        args.h * args.d,
        2,               # outfea_class
        1,               # outfea_regress
        args.L,
        args.h,
        args.d,
        args.s,
        args.T1,
        args.T2,
        device,
    ).to(device)

    print(f"Loading checkpoint: {args.model_file}")
    model.load_state_dict(torch.load(args.model_file, map_location=device))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params:,}")

    # ── Step 7: Inference loop ────────────────────────────────────────────────
    print("Running inference on test split...")
    num_test = testXL.shape[0]
    num_batch = math.ceil(num_test / args.batch_size)

    pred_class = []
    pred_regress = []
    label_class = []
    label_regress = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)

            xl = torch.from_numpy(testXL[start_idx:end_idx]).float().to(device)
            xh = torch.from_numpy(testXH[start_idx:end_idx]).float().to(device)
            xc = torch.from_numpy(testXC[start_idx:end_idx]).float().to(device)
            te = torch.from_numpy(testTE[start_idx:end_idx]).to(device)
            bonus = torch.from_numpy(bonus_testX[start_idx:end_idx]).float().to(device)
            y = testY[start_idx:end_idx]
            yc = testYC[start_idx:end_idx]

            hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(
                xl, xh, te, bonus, xc, adjgat
            )

            pred_class.append(hat_y_class.cpu().numpy())
            pred_regress.append(hat_y_regress.cpu().numpy())
            label_class.append(yc)
            label_regress.append(y)

    pred_class = np.concatenate(pred_class, axis=0)
    pred_regress = np.concatenate(pred_regress, axis=0)
    label_class = np.concatenate(label_class, axis=0)
    label_regress = np.concatenate(label_regress, axis=0)

    # ── Step 8: Compute metrics ───────────────────────────────────────────────
    print("\nTest set metrics:")
    accs, maes, rmses, mapes = [], [], [], []
    for i in range(pred_regress.shape[1]):
        acc, mae, rmse, mape = metric(
            pred_regress[:, i, :],
            label_regress[:, i, :],
            pred_class[:, i, :],
            label_class[:, i, :],
        )
        accs.append(acc)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        print(
            f"  step {i + 1}: acc={acc:.4f}, mae={mae:.4f}, "
            f"rmse={rmse:.4f}, mape={mape:.4f}"
        )

    avg_acc = np.mean(accs)
    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_mape = np.mean(mapes)
    print(
        f"  average:  acc={avg_acc:.4f}, mae={avg_mae:.4f}, "
        f"rmse={avg_rmse:.4f}, mape={avg_mape:.4f}"
    )

    # ── Step 9: Save prediction CSVs ─────────────────────────────────────────
    os.makedirs(os.path.join(args.output_dir, "classification"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "regression"), exist_ok=True)

    save_to_csv(
        os.path.join(args.output_dir, "classification", "classification_pred_last_step.csv"),
        pred_class[:, -1, :],
    )
    save_to_csv(
        os.path.join(args.output_dir, "classification", "classification_label_last_step.csv"),
        label_class[:, -1],
    )
    save_to_csv(
        os.path.join(args.output_dir, "regression", "regression_pred_last_step.csv"),
        pred_regress[:, -1, :],
    )
    save_to_csv(
        os.path.join(args.output_dir, "regression", "regression_label_last_step.csv"),
        label_regress[:, -1],
    )

    print(f"\nInference complete. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
