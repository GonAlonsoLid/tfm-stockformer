import os
import pandas as pd

def filter_date_range(file_path, start_date, end_date):
    # Read data
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # Slice data for the given date range
    filtered_data = data[start_date:end_date]
    return filtered_data

def save_filtered_data(source_dir, target_dir, start_date, end_date):
    # Create target folder if it does not exist
    os.makedirs(target_dir, exist_ok=True)

    # Iterate over all CSV files in the source folder
    for filename in os.listdir(source_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(source_dir, filename)
            # Read and slice data for the given date range
            selected_data = filter_date_range(file_path, start_date, end_date)
            # Save to the new target folder
            selected_data.to_csv(os.path.join(target_dir, filename), index=True)

def main(start_date, end_date, label_source_path=None, alpha_source_dir=None, target_base_dir=None):
    if label_source_path is None:
        label_source_path = './data/Stock_CN_2018-03-01_2024-03-01/label_processed.csv'
    if alpha_source_dir is None:
        alpha_source_dir = './data/Alpha_360_2018-03-01_2024-03-01_data'
    if target_base_dir is None:
        target_base_dir = './data/'

    # Target folder path
    target_folder_name = f'Stock_CN_{start_date}_{end_date}'
    target_dir = os.path.join(target_base_dir, target_folder_name)

    # Create top-level target folder
    os.makedirs(target_dir, exist_ok=True)

    # Process label file
    label_data = filter_date_range(label_source_path, start_date, end_date)
    label_data.to_csv(os.path.join(target_dir, 'label.csv'), index=True)

    # Create and save Alpha 360 data
    alpha_target_dir = os.path.join(target_dir, f'Alpha_360_{start_date}_{end_date}')
    save_filtered_data(alpha_source_dir, alpha_target_dir, start_date, end_date)

    print(f"All files have been processed and saved to: {target_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date', default='2021-06-04')
    parser.add_argument('--end_date', default='2024-01-30')
    parser.add_argument('--label_source', default='./data/Stock_CN_2018-03-01_2024-03-01/label_processed.csv')
    parser.add_argument('--alpha_source_dir', default='./data/Alpha_360_2018-03-01_2024-03-01_data')
    parser.add_argument('--target_base_dir', default='./data/')
    _args = parser.parse_args()
    main(_args.start_date, _args.end_date, _args.label_source, _args.alpha_source_dir, _args.target_base_dir)
