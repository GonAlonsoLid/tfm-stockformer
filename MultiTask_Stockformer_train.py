from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import configparser
import math
import csv
import random
from pytorch_wavelets import DWT1DForward, DWT1DInverse
from scipy.stats import spearmanr
from lib.Multitask_Stockformer_utils import log_string, _compute_regression_loss, _compute_class_loss, combined_ranking_loss, metric, save_to_csv, StockDataset
from lib.graph_utils import loadGraph
from Stockformermodel.Multitask_Stockformer_models import Stockformer

import os
from torch.utils.tensorboard import SummaryWriter

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')

# First parse: only get config file
args, unknown = parser.parse_known_args()  # Use known_args to avoid conflict with later-added args

# Read config file
config = configparser.ConfigParser()
config.read(args.config)

# Add other config parameters
parser.add_argument('--cuda', type=str, default=config['train']['cuda'])
parser.add_argument('--seed', type=int, default=config['train']['seed'])
parser.add_argument('--batch_size', type=int, default=config['train']['batch_size'])
parser.add_argument('--max_epoch', type=int, default=config['train']['max_epoch'])
parser.add_argument('--learning_rate', type=float, default=config['train']['learning_rate'])

parser.add_argument('--Dataset', default=config['data']['dataset'])
parser.add_argument('--T1', type=int, default=config['data']['T1'])
parser.add_argument('--T2', type=int, default=config['data']['T2'])
parser.add_argument('--train_ratio', type=float, default=config['data']['train_ratio'])
parser.add_argument('--val_ratio', type=float, default=config['data']['val_ratio'])
parser.add_argument('--test_ratio', type=float, default=config['data']['test_ratio'])

parser.add_argument('--L', type=int, default=config['param']['layers'])
parser.add_argument('--h', type=int, default=config['param']['heads'])
parser.add_argument('--d', type=int, default=config['param']['dims'])
parser.add_argument('--j', type=int, default=config['param']['level'])
parser.add_argument('--s', type=float, default=config['param']['samples'])
parser.add_argument('--w', default=config['param']['wave'])
parser.add_argument('--max_features', type=int, default=config['param'].get('max_features', '0'))
parser.add_argument('--decomposition', default=config['param'].get('decomposition', 'dwt'))

parser.add_argument('--traffic_file', default=config['file']['traffic'])
parser.add_argument('--indicator_file', default=config['file']['indicator'])
parser.add_argument('--adj_file', default=config['file']['adj'])
parser.add_argument('--adjgat_file', default=config['file']['adjgat'])
parser.add_argument('--model_file', default=config['file']['model'])
parser.add_argument('--log_file', default=config['file']['log'])
parser.add_argument('--alpha_360_dir', default=config['file']['alpha_360_dir'])
parser.add_argument('--output_dir', default=config['file']['output_dir'])
parser.add_argument('--tensorboard_dir', default=config['file']['tensorboard_dir'])

# Final argument parse
args = parser.parse_args()

# Check and create log file directory
log_directory = os.path.dirname(args.log_file)
if not os.path.exists(log_directory):
    os.makedirs(log_directory)
    print(f"Directory created for log file: {log_directory}")
    
log = open(args.log_file, 'w')

# Check and create model file directory
model_directory = os.path.dirname(args.model_file)
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    print(f"Directory created for model file: {model_directory}")

print(f"Model file path is ready at {args.model_file}")


device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

tensorboard_folder = args.tensorboard_dir

# Check and create the main TensorBoard folder
if not os.path.exists(tensorboard_folder):
    os.makedirs(tensorboard_folder)
    log_string(log, f"Folder created: {tensorboard_folder}")
else:
    log_string(log, f"Folder already exists: {tensorboard_folder}")

# Determine the name for the new subfolder
subfolders = [f.name for f in os.scandir(tensorboard_folder) if f.is_dir()]
versions = [int(folder.replace('version', '')) for folder in subfolders if folder.startswith('version')]
next_version = 0 if not versions else max(versions) + 1
new_folder = os.path.join(tensorboard_folder, f'version{next_version}')

# Create the new subfolder
if not os.path.exists(new_folder):
    os.makedirs(new_folder)
    log_string(log, f"Subfolder created: {new_folder}")

# Create a SummaryWriter instance pointing to the new subfolder
tensor_writer = SummaryWriter(new_folder)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

def _evaluate(model, dataXL, dataXH, dataXC, bonus_dataX, dataTE, dataY, dataYC, adjgat):
    """Run model inference on a dataset split and return predictions + metrics.

    Returns (pred_class, pred_regress, label_class, label_regress, avg_acc, avg_mae, avg_rmse, avg_mape).
    """
    model.eval()
    num_samples = dataXL.shape[0]
    num_batch = math.ceil(num_samples / args.batch_size)

    pred_class = []
    pred_regress = []
    label_class = []
    label_regress = []

    with torch.no_grad():
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_samples, (batch_idx + 1) * args.batch_size)

            xl = torch.from_numpy(dataXL[start_idx:end_idx]).float().to(device)
            xh = torch.from_numpy(dataXH[start_idx:end_idx]).float().to(device)
            xc = torch.from_numpy(dataXC[start_idx:end_idx]).float().to(device)
            te = torch.from_numpy(dataTE[start_idx:end_idx]).to(device)
            bonus = torch.from_numpy(bonus_dataX[start_idx:end_idx]).float().to(device)
            y = dataY[start_idx:end_idx]
            yc = dataYC[start_idx:end_idx]

            hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

            pred_class.append(hat_y_class.cpu().numpy())
            pred_regress.append(hat_y_regress.cpu().numpy())
            label_class.append(yc)
            label_regress.append(y)

    pred_class = np.concatenate(pred_class, axis=0)
    pred_regress = np.concatenate(pred_regress, axis=0)
    label_class = np.concatenate(label_class, axis=0)
    label_regress = np.concatenate(label_regress, axis=0)

    accs, maes, rmses, mapes = [], [], [], []
    for i in range(pred_class.shape[1]):
        acc, mae, rmse, mape = metric(pred_regress[:, i, :], label_regress[:, i, :],
                                       pred_class[:, i, :], label_class[:, i, :])
        accs.append(acc)
        maes.append(mae)
        rmses.append(rmse)
        mapes.append(mape)
        log_string(log, f'step {i+1}, acc: {acc:.4f}, mae: {mae:.4f}, rmse: {rmse:.4f}, mape: {mape:.4f}')

    avg_acc = np.mean(accs)
    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_mape = np.mean(mapes)

    # Compute cross-sectional IC (Spearman) on last time step
    ic_values = []
    for b in range(pred_regress.shape[0]):
        pred_slice = pred_regress[b, -1, :]
        label_slice = label_regress[b, -1, :]
        if np.std(pred_slice) > 1e-10 and np.std(label_slice) > 1e-10:
            corr, _ = spearmanr(pred_slice, label_slice)
            if not np.isnan(corr):
                ic_values.append(corr)
    avg_ic = np.mean(ic_values) if ic_values else 0.0

    log_string(log, f'average, acc: {avg_acc:.4f}, mae: {avg_mae:.4f}, rmse: {avg_rmse:.4f}, mape: {avg_mape:.4f}, IC: {avg_ic:.6f}')

    return pred_class, pred_regress, label_class, label_regress, avg_acc, avg_mae, avg_rmse, avg_mape, avg_ic


def validate_epoch(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat, epoch, log, tensor_writer):
    """Validate model on the validation split and log metrics to TensorBoard."""
    _, _, _, _, avg_acc, avg_mae, avg_rmse, avg_mape, avg_ic = _evaluate(
        model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat
    )
    tensor_writer.add_scalar('Val/Average_Accuracy', avg_acc, epoch)
    tensor_writer.add_scalar('Val/Average_MAE', avg_mae, epoch)
    tensor_writer.add_scalar('Val/Average_RMSE', avg_rmse, epoch)
    tensor_writer.add_scalar('Val/Average_MAPE', avg_mape, epoch)
    tensor_writer.add_scalar('Val/IC_Spearman', avg_ic, epoch)
    return avg_acc, avg_mae, avg_rmse, avg_mape, avg_ic


def evaluate_test(model, testXL, testXH, testXC, bonus_testX, testTE, testY, testYC, adjgat):
    """Evaluate model on the test split and save prediction CSVs."""
    pred_class, pred_regress, label_class, label_regress, avg_acc, avg_mae, avg_rmse, avg_mape, avg_ic = _evaluate(
        model, testXL, testXH, testXC, bonus_testX, testTE, testY, testYC, adjgat
    )

    os.makedirs(os.path.join(args.output_dir, 'classification'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'regression'), exist_ok=True)

    save_to_csv(os.path.join(args.output_dir, 'classification', 'classification_pred_last_step.csv'), pred_class[:, -1, :])
    save_to_csv(os.path.join(args.output_dir, 'classification', 'classification_label_last_step.csv'), label_class[:, -1])
    save_to_csv(os.path.join(args.output_dir, 'regression', 'regression_pred_last_step.csv'), pred_regress[:, -1, :])
    save_to_csv(os.path.join(args.output_dir, 'regression', 'regression_label_last_step.csv'), label_regress[:, -1])

    return avg_acc, avg_mae, avg_rmse, avg_mape, avg_ic

def train(model, trainXL, trainXH, trainXC, bonus_trainX, trainTE, trainY, trainYL, trainYC, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat):
    num_train = trainXL.shape[0]
    best_ic = -float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 15
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=20,
        threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08
    )

    for epoch in tqdm(range(1, args.max_epoch + 1)):
        model.train()
        train_l_sum, batch_count, start = 0.0, 0, time.time()
        permutation = np.random.permutation(num_train)
        num_batch = math.ceil(num_train / args.batch_size)

        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
                batch_perm = permutation[start_idx:end_idx]

                xl = torch.from_numpy(trainXL[batch_perm]).float().to(device)
                xh = torch.from_numpy(trainXH[batch_perm]).float().to(device)
                xc = torch.from_numpy(trainXC[batch_perm]).float().to(device)
                y = torch.from_numpy(trainY[batch_perm]).float().to(device)
                yl = torch.from_numpy(trainYL[batch_perm]).float().to(device)
                yc = torch.from_numpy(trainYC[batch_perm]).float().to(device)
                te = torch.from_numpy(trainTE[batch_perm]).to(device)
                bonus = torch.from_numpy(bonus_trainX[batch_perm]).float().to(device)

                optimizer.zero_grad()

                hat_y_class, hat_y_l_class, hat_y_regress, hat_y_l_regress = model(xl, xh, te, bonus, xc, adjgat)

                # Ranking loss (ListNet + IC + MAE) replaces pure MAE for regression
                loss_ranking = combined_ranking_loss(hat_y_regress, y)
                loss_class = _compute_class_loss(yc, hat_y_class) + _compute_class_loss(yc, hat_y_l_class)
                loss = loss_ranking + 0.5 * loss_class

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.3)
                optimizer.step()

                train_l_sum += loss.cpu().item()
                batch_count += 1
                pbar.update(1)

        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))

        tensor_writer.add_scalar('training loss', train_l_sum / batch_count, epoch)

        acc, mae, rmse, mape, ic = validate_epoch(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat, epoch, log, tensor_writer)

        lr_scheduler.step(ic)  # Maximize IC instead of minimizing MAE

        if ic > best_ic:
            best_ic = ic
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.model_file)
            log_string(log, f'Epoch {epoch}: New best IC: {best_ic:.6f}, Model saved.')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                log_string(log, f'Early stopping at epoch {epoch} (no IC improvement for {early_stopping_patience} epochs)')
                break


def test(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat):
    try:
        model.load_state_dict(torch.load(args.model_file))
        total_params = sum(p.numel() for p in model.parameters())
        log_string(log, 'Total parameters: {}'.format(total_params))
    except EOFError:
        print(f"Error: Unable to load model state dictionary from file {args.model_file}. File may be empty or corrupted.")
        return

    acc, mae, rmse, mape, ic = evaluate_test(model, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat)
    log_string(log, f"Test IC: {ic:.6f}")
    return acc, mae, rmse, mape, ic


if __name__ == '__main__':
    log_string(log, "loading data....")
    outfea_class = 2
    outfea_regress = 1
    # trainXL, trainXH, trainTE, trainY, trainYL, valXL, valXH, valTE, valY, testXL, testXH, testTE, testY, bonus_all_trainX, bonus_all_valX, bonus_all_testX, infeature = loadData(args)
    train_dataset = StockDataset(args, mode='train')
    val_dataset = StockDataset(args, mode='val')
    test_dataset = StockDataset(args, mode='test')
    # get data
    # train data
    trainXL = train_dataset.XL
    trainXH = train_dataset.XH
    trainXC = train_dataset.indicator_X
    trainTE = train_dataset.TE
    trainY = train_dataset.Y
    trainYL = train_dataset.YL
    trainYC = train_dataset.indicator_Y
    bonus_trainX = train_dataset.bonus_X
    # val data
    valXL = val_dataset.XL
    valXH = val_dataset.XH
    valXC = val_dataset.indicator_X
    valTE = val_dataset.TE
    valY = val_dataset.Y
    valYL = val_dataset.YL
    valYC = val_dataset.indicator_Y
    bonus_valX = val_dataset.bonus_X
    # test data
    testXL = test_dataset.XL
    testXH = test_dataset.XH
    testXC = test_dataset.indicator_X
    testTE = test_dataset.TE
    testY = test_dataset.Y
    testYL = test_dataset.YL
    testYC = test_dataset.indicator_Y
    bonus_testX = test_dataset.bonus_X
    # infeature number
    infeature = train_dataset.infea
    graph_type = getattr(args, 'graph_type', 'static')
    if graph_type == 'static':
        adjgat = loadGraph(args)
        adjgat = torch.from_numpy(adjgat).float().to(device)
    else:
        from lib.dynamic_graph import load_dynamic_graph
        n_stocks = train_dataset.bonus_X.shape[2]  # N dimension
        d_model = args.h * args.d
        adjgat = load_dynamic_graph(args, n_stocks, d_model, graph_type)
        adjgat = adjgat.to(device)  # nn.Module, moves parameters to device
    log_string(log, "loading end....")

    log_string(log, "constructing model begin....")
    model = Stockformer(infeature, args.h*args.d, outfea_class, outfea_regress, args.L, args.h, args.d, args.s, args.T1, args.T2, device).to(device)
    log_string(log, "constructing model end....")

    log_string(log, "training begin....")
    train(model, trainXL, trainXH, trainXC, bonus_trainX, trainTE, trainY, trainYL, trainYC, valXL, valXH, valXC, bonus_valX, valTE, valY, valYC, adjgat)
    log_string(log, "training end....")

    log_string(log, "testing begin....")
    test(model, testXL, testXH, testXC, bonus_testX, testTE, testY, testYC, adjgat)
    log_string(log, "testing end....")
