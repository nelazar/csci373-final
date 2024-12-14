"""
control-models.py
Conducts grid search for each dataset and then evaluates the fairness of the best models
"""


import csv
from pathlib import Path
import pickle

import pandas as pd

import neuralnet as nn


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

# Removes the columns in the given list
def drop_cols(dataset_path, cols):
    with open(dataset_path, 'r') as dataset:
        lines = dataset.readlines()

    col_indices = [lines[0].split(',').index(col) for col in cols]
    for i, line in enumerate(lines):
        line = line.split(',')
        for col_index in col_indices:
            line.pop(col_index)
        lines[i] = ','.join(line)

    with open(dataset_path, 'w') as dataset:
        dataset.writelines(lines)

# Preprocess datasets
def preprocess():
    dataset_locations = get_datasets()
    for dataset_info in dataset_locations:
        relabel(dataset_info['path'], dataset_info['label-col'])
        drop_cols(dataset_info['path'], dataset_info['drop'].split(';'))

# Grid search for each dataset
def gridsearch():
    dataset_locations = get_datasets()
    for dataset_info in dataset_locations:
        print(f"Grid search for {dataset_info['name']}")
        nn.gridsearch_mode(dataset_info['path'], dataset_info['name'], 0.75, 12345, True)

# Creates, trains, and saves a model and its data
def create_model(n_hidden, learning_rate):

    dataset_locations = get_datasets()
    dataset_info = dataset_locations[0]

    dataset = pd.read_csv(dataset_info['path'])
    training_X, training_y, testing_X, testing_y = nn.preprocess(dataset, 0.75, 12345, True)
    with open("data/training_X.pickle", 'wb') as f:
        pickle.dump(training_X, f, pickle.HIGHEST_PROTOCOL)
    with open("data/training_y.pickle", 'wb') as f:
        pickle.dump(training_y, f, pickle.HIGHEST_PROTOCOL)
    with open("data/testing_X.pickle", 'wb') as f:
        pickle.dump(testing_X, f, pickle.HIGHEST_PROTOCOL)
    with open("data/testing_y.pickle", 'wb') as f:
        pickle.dump(testing_y, f, pickle.HIGHEST_PROTOCOL)

    # n_input = training_X.shape[1]
    # n_output = len(training_y.unique())
    # network = nn.create_network(n_input, n_hidden, n_output, True)
    # nn.train_network(network, training_X, training_y, learning_rate, True)
    # with open("data/control-model.pickle", 'wb') as f:
    #     pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    create_model(256, 0.01)