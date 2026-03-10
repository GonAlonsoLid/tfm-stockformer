import pandas as pd
import numpy as np
import re
import os
import argparse

parser = argparse.ArgumentParser(description='Post-process Stockformer prediction outputs')
parser.add_argument('--data_dir', default='./data/Stock_CN_2021-06-04_2024-01-30',
                    help='Directory containing label.csv')
parser.add_argument('--output_dir', default='./output/Multitask_output_2021-06-04_2024-01-30',
                    help='Directory containing model output CSVs')
_args = parser.parse_args()

def load_and_index_data(file_path, index, columns):
    # Load data, set index and column names
    data = pd.read_csv(file_path, header=None)
    data.columns = columns
    data.index = index
    return data

def apply_extraction_and_softmax(data):
    # Extract numbers and convert strings to floats
    pattern = r'-?\d+\.\d+(?:[eE][+-]?\d+)?|-?\d+'
    data = data.astype(str).map(lambda x: [float(num) for num in re.findall(pattern, x)])

    # Compute softmax and extract max index and class '1' probability
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    max_indices = {}
    class_1_probabilities = {}

    for column in data.columns:
        probabilities = data[column].apply(softmax)
        max_indices[column] = probabilities.apply(np.argmax)
        class_1_probabilities[column] = probabilities.apply(lambda x: x[1])

    pred_index = pd.DataFrame(max_indices, index=data.index)
    pred_prob = pd.DataFrame(class_1_probabilities, index=data.index)
    return pred_index, pred_prob

# Use detail_data as base to get index and column names
detail_data = pd.read_csv(os.path.join(_args.data_dir, 'label.csv'), index_col=0)
detail_data.index = pd.to_datetime(detail_data.index)
index = detail_data.index
columns = detail_data.columns

start_date = '2023-11-07'
filtered_index = detail_data.loc[start_date:].index

# Folder paths
folder_paths = {
    'regression': os.path.join(_args.output_dir, 'regression') + os.sep,
    'classification': os.path.join(_args.output_dir, 'classification') + os.sep,
}

# Load data and apply index
regression_label_data = load_and_index_data(folder_paths['regression'] + 'regression_label_last_step.csv', filtered_index, columns)
regression_pred_data = load_and_index_data(folder_paths['regression'] + 'regression_pred_last_step.csv', filtered_index, columns)
classification_label_data = load_and_index_data(folder_paths['classification'] + 'classification_label_last_step.csv', filtered_index, columns)
classification_pred_data = load_and_index_data(folder_paths['classification'] + 'classification_pred_last_step.csv', filtered_index, columns)

# Save the first three files
regression_label_data.to_csv(folder_paths['regression'] + 'regression_label_with_index.csv')
regression_pred_data.to_csv(folder_paths['regression'] + 'regression_pred_with_index.csv')
classification_label_data.to_csv(folder_paths['classification'] + 'classification_label_with_index.csv')

# Apply extraction and softmax to classification prediction data
pred_index, pred_prob = apply_extraction_and_softmax(classification_pred_data)
pred_index.to_csv(folder_paths['classification'] + 'classification_pred_with_index.csv')
pred_prob.to_csv(folder_paths['classification'] + 'classification_pred_prob.csv')

# Create mask from 0 values in labels
mask = (regression_label_data == 0)
# Use mask to replace corresponding elements in regression predictions with -1e9
regression_pred_data[mask] = -1e9
# Use mask to replace corresponding elements in classification prediction probabilities with 0
pred_prob[mask] = 0

regression_pred_data.to_csv(folder_paths['regression'] + 'regression_pred_with_index_fill_-1e9.csv')
pred_prob.to_csv(folder_paths['classification'] + 'classification_pred_prob_fill_0.csv')


print("All files have been processed and saved.")
