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
        reader = csv.DictReader(dataset)
        lines = [line for line in reader]
        
    for line in lines:
        for col in cols:
            line.pop(col, None)

    with open(dataset_path, 'w') as dataset:
        writer = csv.DictWriter(dataset, fieldnames = lines[0].keys())
        writer.writeheader()
        writer.writerows(lines)

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

# Creates, trains, and saves the control model and its data
def control_model(n_hidden, learning_rate):

    dataset_locations = get_datasets()
    dataset_info = dataset_locations[0]

    dataset = pd.read_csv(dataset_info['path'])
    training, testing = nn.split_data_fairness(dataset, 0.75, 12345)
    training.to_csv('data/training.csv', index=False)
    testing.to_csv('data/testing.csv', index=False)

    training, testing = nn.preprocess_fairness(training, testing)
    training_X, training_y, testing_X, testing_y = nn.split_labels(training, testing)

    n_input = training_X.shape[1]
    n_output = len(training_y.unique())
    network = nn.create_network(n_input, n_hidden, n_output, True)
    nn.train_network(network, training_X, training_y, learning_rate, True)
    with open("data/control-model.pickle", 'wb') as f:
        pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)

# Creates, trains, and saves the fair model
def fair_model(n_hidden, learning_rate):

    training = pd.read_csv('data/transformed.csv')
    testing = pd.read_csv('data/testing.csv')

    training, testing = nn.preprocess_fairness(training, testing)
    training_X, training_y, testing_X, testing_y = nn.split_labels(training, testing)

    n_input = training_X.shape[1]
    n_output = len(training_y.unique())
    network = nn.create_network(n_input, n_hidden, n_output, True)
    nn.train_network(network, training_X, training_y, learning_rate, True)
    with open("data/fair-model.pickle", 'wb') as f:
        pickle.dump(network, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # preprocess()
    # gridsearch()
    control_model(256, 0.001)
    # fair_model(256, 0.001)