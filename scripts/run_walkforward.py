#!/usr/bin/env python3
"""Walk-forward rolling window training for Stockformer.

Instead of a single static split, trains multiple models on rolling windows
and concatenates test predictions for robust IC evaluation.

Usage:
    python scripts/run_walkforward.py --config config/Multitask_Stock_SP500.conf
    python scripts/run_walkforward.py --config config/Multitask_Stock_SP500.conf \
        --train_days 756 --val_days 63 --test_days 21 --step_days 21
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.Multitask_Stockformer_utils import (
    disentangle, combined_ranking_loss, _compute_class_loss,
)
from lib.graph_utils import loadGraph
from lib.config import load_config
from Stockformermodel.Multitask_Stockformer_models import Stockformer


def seq2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    if num_sample <= 0:
        return np.zeros((0, P, dims)), np.zeros((0, Q, dims))
    x = np.zeros((num_sample, P, dims))
    y = np.zeros((num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i:i + P]
        y[i] = data[i + P:i + P + Q]
    return x, y


def bonus_seq2instance(data, P, Q):
    num_step, dims, N = data.shape
    num_sample = num_step - P - Q + 1
    if num_sample <= 0:
        return np.zeros((0, P, dims, N)), np.zeros((0, Q, dims, N))
    x = np.zeros((num_sample, P, dims, N))
    y = np.zeros((num_sample, Q, dims, N))
    for i in range(num_sample):
        x[i] = data[i:i + P]
        y[i] = data[i + P:i + P + Q]
    return x, y


def generate_te(num_step):
    TE = np.zeros([num_step, 2])
    DAYS_PER_MONTH = 21
    MONTHS = 12
    startd = (3 - 1) * DAYS_PER_MONTH
    startt = 0
    for i in range(num_step):
        TE[i, 0] = startd // DAYS_PER_MONTH
        TE[i, 1] = startt
        startd = (startd + 1) % (MONTHS * DAYS_PER_MONTH)
        startt = (startt + 1) % DAYS_PER_MONTH
    return TE


def prepare_window(traffic, indicator, bonus_all, te_all, start, end, T1, T2, w, j):
    """Slice data and prepare model-ready arrays for a window."""
    data = traffic[start:end]
    ind = indicator[start:end]
    bonus = bonus_all[start:end]
    te = te_all[start:end]

    X, Y = seq2instance(data, T1, T2)
    if X.shape[0] == 0:
        return None
    XL, XH = disentangle(X, w, j)
    YL, _ = disentangle(Y, w, j)
    ind_X, ind_Y = seq2instance(ind, T1, T2)
    bonus_X, _ = bonus_seq2instance(bonus, T1, T2)
    te_seq = seq2instance(te, T1, T2)
    TE = np.concatenate(te_seq, axis=1).astype(np.int32)
    infea = bonus.shape[-1] + 2
    return {
        "XL": XL, "XH": XH, "XC": ind_X, "bonus_X": bonus_X,
        "TE": TE, "Y": Y, "YL": YL, "YC": ind_Y, "infea": infea,
    }


def train_window(model, train_d, val_d, adjgat, device, args):
    """Train model on one window, return best model state dict."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_ic = -float("inf")
    best_state = None
    batch_size = args.batch_size
    num_train = train_d["XL"].shape[0]

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        perm = np.random.permutation(num_train)
        num_batch = math.ceil(num_train / batch_size)
        total_loss = 0.0

        for b in range(num_batch):
            s, e = b * batch_size, min(num_train, (b + 1) * batch_size)
            idx = perm[s:e]
            xl = torch.from_numpy(train_d["XL"][idx]).float().to(device)
            xh = torch.from_numpy(train_d["XH"][idx]).float().to(device)
            xc = torch.from_numpy(train_d["XC"][idx]).float().to(device)
            y = torch.from_numpy(train_d["Y"][idx]).float().to(device)
            yc = torch.from_numpy(train_d["YC"][idx]).float().to(device)
            te = torch.from_numpy(train_d["TE"][idx]).to(device)
            bonus = torch.from_numpy(train_d["bonus_X"][idx]).float().to(device)

            optimizer.zero_grad()
            hat_y_class, hat_y_l_class, hat_y_regress, _ = model(xl, xh, te, bonus, xc, adjgat)
            loss = combined_ranking_loss(hat_y_regress, y) + 0.5 * (
                _compute_class_loss(yc, hat_y_class) + _compute_class_loss(yc, hat_y_l_class)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        ic = compute_ic(model, val_d, adjgat, device, batch_size)
        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_ic


def compute_ic(model, data, adjgat, device, batch_size):
    """Compute mean Spearman IC on a dataset."""
    num = data["XL"].shape[0]
    num_batch = math.ceil(num / batch_size)
    preds, labels = [], []
    with torch.no_grad():
        for b in range(num_batch):
            s, e = b * batch_size, min(num, (b + 1) * batch_size)
            xl = torch.from_numpy(data["XL"][s:e]).float().to(device)
            xh = torch.from_numpy(data["XH"][s:e]).float().to(device)
            xc = torch.from_numpy(data["XC"][s:e]).float().to(device)
            te = torch.from_numpy(data["TE"][s:e]).to(device)
            bonus = torch.from_numpy(data["bonus_X"][s:e]).float().to(device)
            _, _, hat_y_regress, _ = model(xl, xh, te, bonus, xc, adjgat)
            preds.append(hat_y_regress.cpu().numpy())
            labels.append(data["Y"][s:e])

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    ics = []
    for i in range(preds.shape[0]):
        p, l = preds[i, -1, :], labels[i, -1, :]
        if np.std(p) > 1e-10 and np.std(l) > 1e-10:
            c, _ = spearmanr(p, l)
            if not np.isnan(c):
                ics.append(c)
    return np.mean(ics) if ics else 0.0


def main():
    parser = argparse.ArgumentParser(description="Walk-forward rolling window training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train_days", type=int, default=756)
    parser.add_argument("--val_days", type=int, default=63)
    parser.add_argument("--test_days", type=int, default=21)
    parser.add_argument("--step_days", type=int, default=21)
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--output_dir", default="output/walkforward")
    parser.add_argument("--max_features", type=int, default=-1,
                        help="-1=load all, 0=auto-fit RAM, N=limit to N features")
    cli_args = parser.parse_args()

    args = load_config(cli_args.config)
    args.max_epoch = cli_args.max_epoch
    if cli_args.max_features != -1:
        args.max_features = cli_args.max_features

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    traffic = np.load(args.traffic_file)["result"]
    indicator = np.load(args.indicator_file)["result"]
    path = args.alpha_360_dir
    files = sorted(os.listdir(path))
    data_list = []
    for f in files:
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, f), index_col=0)
            data_list.append(np.expand_dims(df.values, axis=2))
    bonus_all = np.concatenate(data_list, axis=2)
    if np.isnan(bonus_all).any():
        bonus_all = np.nan_to_num(bonus_all, nan=0.0)

    adjgat_np = np.load(args.adjgat_file)
    adjgat = torch.from_numpy(adjgat_np).float().to(device)

    T1, T2 = args.T1, args.T2
    total_days = traffic.shape[0]
    te_all = generate_te(total_days)
    infea = bonus_all.shape[-1] + 2

    train_d = cli_args.train_days
    val_d = cli_args.val_days
    test_d = cli_args.test_days
    step_d = cli_args.step_days
    window_size = train_d + val_d + test_d + T1 + T2

    all_test_ics = []
    window_results = []

    window_idx = 0
    for ws in range(0, total_days - window_size + 1, step_d):
        window_idx += 1
        train_end = ws + train_d
        val_end = train_end + val_d
        test_end = val_end + test_d

        print(f"\n{'='*60}")
        print(f"Window {window_idx}: train[{ws}:{train_end}] val[{train_end}:{val_end}] test[{val_end}:{test_end}]")

        train_data = prepare_window(traffic, indicator, bonus_all, te_all, ws, train_end, T1, T2, args.w, args.j)
        val_data = prepare_window(traffic, indicator, bonus_all, te_all, train_end, val_end, T1, T2, args.w, args.j)
        test_data = prepare_window(traffic, indicator, bonus_all, te_all, val_end, test_end, T1, T2, args.w, args.j)

        if train_data is None or val_data is None or test_data is None:
            print(f"  [SKIP] Window too small")
            continue

        # Build fresh model
        model = Stockformer(
            infea, args.h * args.d, 2, 1, args.L, args.h, args.d, args.s, T1, T2, device
        ).to(device)

        best_state, val_ic = train_window(model, train_data, val_data, adjgat, device, args)
        print(f"  Val IC: {val_ic:.6f}")

        if best_state:
            model.load_state_dict(best_state)
        model.to(device)

        test_ic = compute_ic(model, test_data, adjgat, device, args.batch_size)
        print(f"  Test IC: {test_ic:.6f}")
        all_test_ics.append(test_ic)
        window_results.append({"window": window_idx, "val_ic": val_ic, "test_ic": test_ic})

    # Save results
    os.makedirs(cli_args.output_dir, exist_ok=True)

    if window_results:
        df_results = pd.DataFrame(window_results)
        df_results.to_csv(os.path.join(cli_args.output_dir, "window_results.csv"), index=False)

        mean_ic = np.mean(all_test_ics)
        std_ic = np.std(all_test_ics) if len(all_test_ics) > 1 else 0
        icir = mean_ic / std_ic if std_ic > 0 else 0
        pct_pos = np.mean([ic > 0 for ic in all_test_ics]) * 100

        summary = {
            "mean_ic": mean_ic, "std_ic": std_ic, "icir": icir,
            "pct_positive": pct_pos, "n_windows": len(all_test_ics),
        }
        pd.DataFrame([summary]).to_csv(os.path.join(cli_args.output_dir, "walkforward_summary.csv"), index=False)

        print(f"\n{'='*60}")
        print(f"Walk-Forward Summary ({len(all_test_ics)} windows):")
        print(f"  Mean IC:       {mean_ic:.6f}")
        print(f"  IC IR:         {icir:.4f}")
        print(f"  % Positive IC: {pct_pos:.1f}%")
    else:
        print("\nNo windows completed.")


if __name__ == "__main__":
    main()
