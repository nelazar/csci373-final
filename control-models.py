"""
control-models.py
Conducts grid search for each dataset and then evaluates the fairness of the best models
"""


import csv
from pathlib import Path

import neuralnet


# Read locations of datasets
def get_datasets():
    dataset_locations = []
    with open(Path("./data/datasets.csv"), 'r') as dataset_locations_file:
        dataset_locations_reader = csv.DictReader(dataset_locations_file)
        for dataset in dataset_locations_reader:
            dataset_locations.append(dataset)

    return dataset_locations

# Changes the dataset label column to "label"
def relabel(dataset_path, label_col):
    with open(dataset_path, 'r') as dataset:
        lines = dataset.readlines()

    column_headers = lines[0].split(',')
    column_headers = list(map(lambda x: x.replace(label_col, "label"), column_headers))
    lines[0] = ','.join(column_headers)
    
    with open(dataset_path, 'w') as dataset:
        dataset.writelines(lines)

# Preprocess datasets
def preprocess():
    dataset_locations = get_datasets()
    for dataset_info in dataset_locations:
        relabel(dataset_info['path'], dataset_info['label-col'])

# Grid search for each dataset
def gridsearch():
    dataset_locations = get_datasets()
    for dataset_info in dataset_locations:
        print(f"Grid search for {dataset_info['name']}")
        neuralnet.gridsearch_mode(dataset_info['path'], dataset_info['name'], 0.75, 12345, True)

if __name__ == "__main__":
    gridsearch()