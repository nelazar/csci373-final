"""
control-models.py
Conducts grid search for each dataset and then evaluates the fairness of the best models
"""


import csv
from pathlib import Path

import neuralnet


# Read locations of datasets
dataset_locations = []
with open(Path("./data/datasets.csv"), 'r') as dataset_locations_file:
    dataset_locations_reader = csv.DictReader(dataset_locations_file)
    for dataset in dataset_locations_reader:
        dataset_locations.append(dataset)

# Grid search for each dataset
def gridsearch():
    for dataset_info in dataset_locations:
        neuralnet.gridsearch_mode(dataset_info['path'], dataset_info['name'], 0.75, 12345, True)

if __name__ == "__main__":
    gridsearch()