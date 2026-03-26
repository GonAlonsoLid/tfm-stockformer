import pandas as pd
import numpy as np
import argparse
import os


def save_model_arrays(label_df: pd.DataFrame, output_dir: str) -> None:
    """
    Save flow.npz (raw forward returns) and trend_indicator.npz (binary) from label_df.
    label_df: DataFrame [T, N] with 1-day forward returns (raw, not normalized).
    These are the regression and classification targets for StockDataset.
    """
    data = label_df.values.astype(np.float32)  # [T, N]
    assert not np.isnan(data).any(), "label_df contains NaN — run clean_and_align() first"
    np.savez(os.path.join(output_dir, 'flow.npz'), result=data)
    trend = (data > 0).astype(np.int32)
    np.savez(os.path.join(output_dir, 'trend_indicator.npz'), result=trend)
    print(f"Saved flow.npz and trend_indicator.npz with shape {data.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/Stock_SP500_2018-01-01_2024-01-01')
    parser.add_argument('--label_file', default='label.csv')
    args = parser.parse_args()

    label_path = os.path.join(args.data_dir, args.label_file)
    df = pd.read_csv(label_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    # Drop tickers (columns) that are entirely NaN
    df = df.dropna(axis=1, how='all')
    # Forward-fill sparse gaps (standard in financial time series), then drop remaining
    df = df.ffill().dropna()
    save_model_arrays(df, args.data_dir)


if __name__ == '__main__':
    main()
