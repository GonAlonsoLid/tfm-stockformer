import pandas as pd
import numpy as np
import argparse
import os
import json


def cross_sectional_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-date cross-sectional z-score normalization.
    For each row (date), z-score across the N stocks.
    No time leakage: only uses stocks at the same time point.
    Exported for reuse by other pipeline modules.
    """
    row_mean = df.mean(axis=1)
    row_std = df.std(axis=1)
    row_std = row_std.where(row_std >= 1e-8, other=1.0)
    normalized = df.sub(row_mean, axis=0).div(row_std, axis=0)
    return normalized


def split_by_date(df: pd.DataFrame, train_ratio: float = 0.75,
                  val_ratio: float = 0.125) -> tuple:
    """
    Date-ordered split. Returns (train_df, val_df, test_df).
    Split boundaries are integer positions (reproducible, no randomness).
    """
    T = len(df)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data/Stock_SP500_2018-01-01_2024-01-01')
    parser.add_argument('--label_file', default='label.csv')
    args = parser.parse_args()

    label_path = os.path.join(args.data_dir, args.label_file)
    df = pd.read_csv(label_path, index_col=0)
    df.index = pd.to_datetime(df.index)

    # Split by date position only — feature CSVs are already normalized (Plan 02-03).
    # This step records split boundaries for reproducibility; no normalization applied here.
    train_df, val_df, test_df = split_by_date(df)

    splits = {'train_end': len(train_df), 'val_end': len(train_df) + len(val_df)}
    with open(os.path.join(args.data_dir, 'split_indices.json'), 'w') as f:
        json.dump(splits, f)
    print(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} trading days")
    print(f"split_indices.json written to {args.data_dir}")


if __name__ == '__main__':
    main()
