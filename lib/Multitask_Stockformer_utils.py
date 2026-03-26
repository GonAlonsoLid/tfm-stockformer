import numpy as np
import pandas as pd
import os
import torch
import math
from pytorch_wavelets import DWT1DForward, DWT1DInverse
import csv
from torch.utils.data import Dataset

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(reg_pred, reg_label, class_pred, class_label):
    with np.errstate(divide='ignore', invalid='ignore'):
        # Regression task metric computation
        mask = np.not_equal(reg_label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(reg_pred, reg_label)).astype(np.float32)
        wape = np.divide(np.sum(mae), np.sum(reg_label))
        wape = np.nan_to_num(wape * mask)
        rmse = np.square(mae)
        mape = np.divide(mae, np.abs(reg_label))
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
        
        # Classification task accuracy computation
        pred_classes = np.argmax(class_pred, axis=-1)
        correct = (pred_classes == class_label).astype(np.float32)
        acc = np.mean(correct)

    return acc, mae, rmse, mape


# Initialize cross-entropy loss function
criterion = torch.nn.CrossEntropyLoss()

def _compute_class_loss(y_true, y_predicted):
    # Flatten y_predicted and y_true
    y_predicted_flat = y_predicted.view(-1, y_predicted.size(-1))  # [batch_size * seq_len * num_nodes, num_classes]
    y_true_flat = y_true.view(-1).long()  # Convert to long type

    # Compute loss
    loss = criterion(y_predicted_flat, y_true_flat)
    return loss


def _compute_regression_loss(y_true, y_predicted):
    return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


# ── Ranking Loss Functions ────────────────────────────────────────────────────
# These losses optimize cross-sectional ranking (IC) rather than point-wise error.
# Research shows ranking losses dramatically improve stock prediction on S&P 500
# (Kwiatkowski & Chudziak, arXiv:2510.14156, 2025).

def listnet_loss(y_pred, y_true, temperature=1.0):
    """ListNet listwise ranking loss.

    Computes KL divergence between softmax distributions of predictions and labels
    across the stock dimension. Optimizes the full ranking distribution.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted returns, shape [B, T, N] or [B, N].
    y_true : torch.Tensor
        Actual returns, shape [B, T, N] or [B, N].
    temperature : float
        Softmax temperature. Lower = sharper distribution.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    # Use last time step if 3D
    if y_pred.dim() == 3:
        y_pred = y_pred[:, -1, :]  # [B, N]
        y_true = y_true[:, -1, :]
    # Softmax over stock dimension (cross-sectional ranking)
    p_true = torch.softmax(y_true / temperature, dim=-1)
    p_pred = torch.log_softmax(y_pred / temperature, dim=-1)
    # KL divergence: sum over stocks, mean over batch
    loss = -torch.sum(p_true * p_pred, dim=-1)
    return torch.mean(loss)


def ic_loss(y_pred, y_true, eps=1e-8):
    """Differentiable Information Coefficient (Pearson correlation) loss.

    Computes 1 - Pearson correlation between predictions and labels
    across the stock dimension. Directly optimizes cross-sectional IC.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted returns, shape [B, T, N] or [B, N].
    y_true : torch.Tensor
        Actual returns, shape [B, T, N] or [B, N].

    Returns
    -------
    torch.Tensor
        Scalar loss = 1 - mean(IC), so minimizing this maximizes IC.
    """
    if y_pred.dim() == 3:
        y_pred = y_pred[:, -1, :]
        y_true = y_true[:, -1, :]
    # Center
    pred_mean = y_pred.mean(dim=-1, keepdim=True)
    true_mean = y_true.mean(dim=-1, keepdim=True)
    pred_centered = y_pred - pred_mean
    true_centered = y_true - true_mean
    # Pearson correlation per batch element
    cov = (pred_centered * true_centered).sum(dim=-1)
    pred_std = torch.sqrt((pred_centered ** 2).sum(dim=-1) + eps)
    true_std = torch.sqrt((true_centered ** 2).sum(dim=-1) + eps)
    corr = cov / (pred_std * true_std)
    return 1.0 - corr.mean()


def combined_ranking_loss(y_pred, y_true, alpha=0.5, beta=0.3, gamma=0.2):
    """Combined ranking loss: ListNet + IC + MAE.

    Parameters
    ----------
    y_pred, y_true : torch.Tensor
        Shape [B, T, N] or [B, N].
    alpha : float
        Weight for ListNet loss.
    beta : float
        Weight for IC loss.
    gamma : float
        Weight for MAE loss.

    Returns
    -------
    torch.Tensor
        Weighted combination of ranking and regression losses.
    """
    l_listnet = listnet_loss(y_pred, y_true)
    l_ic = ic_loss(y_pred, y_true)
    l_mae = _compute_regression_loss(y_true, y_pred)
    return alpha * l_listnet + beta * l_ic + gamma * l_mae


def disentangle(data, w, j):
    # Disentangle
    dwt = DWT1DForward(wave=w, J=j)
    idwt = DWT1DInverse(wave=w)
    torch_traffic = torch.from_numpy(data).transpose(1,-1).reshape(data.shape[0]*data.shape[2], -1).unsqueeze(1)
    torch_trafficl, torch_traffich = dwt(torch_traffic.float())
    placeholderh = torch.zeros(torch_trafficl.shape)
    placeholderl = []
    for i in range(j):
        placeholderl.append(torch.zeros(torch_traffich[i].shape))
    torch_trafficl = idwt((torch_trafficl, placeholderl)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    torch_traffich = idwt((placeholderh, torch_traffich)).reshape(data.shape[0],data.shape[2],1,-1).squeeze(2).transpose(1,2)
    trafficl = torch_trafficl.numpy()
    traffich = torch_traffich.numpy()
    return trafficl, traffich

def generate_temporal_embeddings(num_step, args):
    """Generate day-of-week and time-of-day temporal embeddings.

    Encodes each trading day with two indices:
    - TE[:, 0]: month index (0..11), cycling every 12*21=252 trading days (~1 year)
    - TE[:, 1]: day-within-month index (0..20), cycling every 21 trading days (~1 month)
    """
    TRADING_DAYS_PER_MONTH = 21
    MONTHS_PER_YEAR = 12
    TE = np.zeros([num_step, 2])
    startd = (3 - 1) * TRADING_DAYS_PER_MONTH  # Start at month index 2 (March)
    startt = 0
    for i in range(num_step):
        TE[i, 0] = startd // TRADING_DAYS_PER_MONTH
        TE[i, 1] = startt
        startd = (startd + 1) % (MONTHS_PER_YEAR * TRADING_DAYS_PER_MONTH)
        startt = (startt + 1) % TRADING_DAYS_PER_MONTH
    return TE

def _estimate_max_features(n_files, path, T1, T2, ram_fraction=0.6):
    """Estimate how many features fit in available RAM.

    Calculates memory needed for 3 splits of bonus_seq2instance arrays
    plus overhead for XL, XH, indicators, labels, etc.
    """
    import psutil
    available = psutil.virtual_memory().available

    # Read one CSV to get T and N dimensions
    sample_file = sorted(f for f in os.listdir(path) if f.endswith('.csv'))[0]
    sample = pd.read_csv(os.path.join(path, sample_file), index_col=0)
    T, N = sample.shape

    # Sliding window samples
    num_samples = T - T1 - T2 + 1

    # Memory per feature per split: num_samples × T1 × N × 4 bytes (float32)
    # × 2 (bonus_X + bonus_Y) × 3 splits (train/val/test)
    bytes_per_feature = num_samples * T1 * N * 4 * 2
    total_per_feature = bytes_per_feature * 3  # 3 splits loaded simultaneously

    # Overhead: XL, XH, X, Y, indicator, TE arrays ≈ 2× base data
    overhead_bytes = num_samples * T1 * N * 4 * 6 * 3

    usable = available * ram_fraction - overhead_bytes
    max_f = max(10, int(usable / total_per_feature))
    max_f = min(max_f, n_files)  # don't exceed available files

    print(f"[StockDataset] RAM available: {available / 1e9:.1f} GB, "
          f"usable ({ram_fraction:.0%}): {usable / 1e9:.1f} GB, "
          f"estimated max features: {max_f}")
    return max_f


class StockDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        # Load data
        Traffic = np.load(args.traffic_file)['result']
        indicator = np.load(args.indicator_file)['result']
        path = args.alpha_360_dir
        files = sorted(f for f in os.listdir(path) if f.endswith('.csv'))
        max_features = getattr(args, 'max_features', -1)
        if max_features == 0:
            # Auto-detect: estimate max features that fit in available RAM
            max_features = _estimate_max_features(
                n_files=len(files), path=path, T1=args.T1, T2=args.T2,
                ram_fraction=0.6,  # use at most 60% of available RAM
            )
        # max_features < 0 means "load all" (no limit)
        if max_features > 0 and len(files) > max_features:
            core = [f for f in files if not f.startswith('MACRO_') and not f.startswith(('RSI_', 'BBANDS_', 'MACD_', 'ATR_', 'ROC_', 'RVOL_'))]
            extra = [f for f in files if f not in core]
            files = (core + extra)[:max_features]
            files.sort()
            print(f"[StockDataset] Limited to {len(files)}/{max_features} features (auto-fit to available RAM)")
        data_list = []
        for file in files:
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path, index_col=0)
            arr = np.expand_dims(df.values, axis=2)
            data_list.append(arr)
        concatenated_arr = np.concatenate(data_list, axis=2)
        # Safety net: replace any residual NaN with 0.0 (the cross-sectional mean
        # for z-score normalized features) so NaN never reaches the model.
        if np.isnan(concatenated_arr).any():
            concatenated_arr = np.nan_to_num(concatenated_arr, nan=0.0)
        bonus_all = concatenated_arr

        # CRITICAL: Align Traffic/indicator (T_lab rows from d_0) with
        # bonus_all (T_feat rows starting from d_60 due to Alpha360 lag buffer).
        #
        # The correct offset is the Alpha360 LAG_BUFFER (60), NOT T_lab - T_feat (59).
        # With offset=59: features[0]=d_60 pairs with label[59]=d_59, and
        # CLOSE_d1[d_60] = Close[d_60]/Close[d_59] ≡ label[d_59]+1 → leakage!
        # With offset=60: features[0]=d_60 pairs with label[60]=d_60, and
        # label[d_60] = Close[d_61]/Close[d_60] ≠ CLOSE_d1[d_60] → correct.
        ALPHA360_LAG = 60  # Alpha360 drops first 60 rows for the 60-day ratio window
        Traffic = Traffic[ALPHA360_LAG:]
        indicator = indicator[ALPHA360_LAG:]
        min_T = min(Traffic.shape[0], bonus_all.shape[0])
        Traffic = Traffic[:min_T]
        indicator = indicator[:min_T]
        bonus_all = bonus_all[:min_T]
        print(f"[StockDataset] Aligned labels to features (offset={ALPHA360_LAG}, T={min_T})")

        num_step = Traffic.shape[0]  # now matches bonus_all
        train_steps = round(args.train_ratio * num_step)
        test_steps = round(args.test_ratio * num_step)
        val_steps = num_step - train_steps - test_steps
        TE = generate_temporal_embeddings(num_step, args)
        if mode == 'train':
            data_slice = slice(None, train_steps)
        elif mode == 'val':
            data_slice = slice(train_steps, train_steps + val_steps)
        else:  # mode == 'test'
            data_slice = slice(-test_steps, None)
        self.data = Traffic[data_slice]
        self.indicator = indicator[data_slice]
        self.bonus_all = bonus_all[data_slice]
        self.TE = TE[data_slice]
        self.X, self.Y = self.seq2instance(self.data, args.T1, args.T2)
        decomp = getattr(args, 'decomposition', 'dwt')
        if decomp == 'stl':
            from lib.decomposition import stl_decompose_batch
            stl_period = getattr(args, 'stl_period', 5)
            self.XL, self.XH = stl_decompose_batch(self.X, period=stl_period)
            self.YL, self.YH = stl_decompose_batch(self.Y, period=stl_period)
        elif decomp == 'vmd':
            from lib.decomposition import sliding_vmd_batch
            self.XL, self.XH = sliding_vmd_batch(self.X)
            self.YL, self.YH = sliding_vmd_batch(self.Y)
        else:  # default: dwt
            self.XL, self.XH = disentangle(self.X, args.w, args.j)
            self.YL, self.YH = disentangle(self.Y, args.w, args.j)
        self.indicator_X, self.indicator_Y = self.seq2instance(self.indicator, args.T1, args.T2)
        self.bonus_X, self.bonus_Y = self.bonus_seq2instance(self.bonus_all, args.T1, args.T2)
        self.TE = self.seq2instance(self.TE, args.T1, args.T2)
        self.TE = np.concatenate(self.TE, axis=1).astype(np.int32)
        # Adding the infea attribute based on bonus_all
        # +2 accounts for the XL and indicator channels concatenated with bonus features
        self.infea = bonus_all.shape[-1] + 2

    def __getitem__(self, index):
        return {
            'X': self.X[index],
            'X_low': self.XL[index],
            'X_high': self.XH[index],
            'indicator_X': self.indicator_X[index],
            'bonus_X': self.bonus_X[index],
            'TE': self.TE[index]
        }, {
            'Y': self.Y[index],
            'Y_low': self.YL[index],
            'Y_high': self.YH[index],
            'indicator_Y': self.indicator_Y[index],
            'bonus_Y': self.bonus_Y[index]
        }

    def __len__(self):
        return len(self.X)

    def seq2instance(self, data, P, Q):
        num_step, dims = data.shape
        num_sample = num_step - P - Q + 1
        x = np.zeros((num_sample, P, dims))
        y = np.zeros((num_sample, Q, dims))
        for i in range(num_sample):
            x[i] = data[i:i+P]
            y[i] = data[i+P:i+P+Q]
        return x, y

    def bonus_seq2instance(self, data, P, Q):
        num_step, dims, N = data.shape
        num_sample = num_step - P - Q + 1
        x = np.zeros((num_sample, P, dims, N))
        y = np.zeros((num_sample, Q, dims, N))
        for i in range(num_sample):
            x[i] = data[i:i+P]
            y[i] = data[i+P:i+P+Q]
        return x, y


def save_to_csv(file_path, data):
    # Check if the directory exists, if not, create it
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Write data to CSV
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)
    print(f"Data saved to {file_path}")  # You might want to replace this with your logging method