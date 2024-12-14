import csv
from pathlib import Path
import pickle

import pandas as pd
import aif360.metrics, aif360.algorithms.preprocessing, aif360.datasets
import torch

import aif360.data
import neuralnet as nn

privileged_groups = [{'Gender': 'Man'}]
unprivileged_groups = [{'Gender': 'Woman'}, {'Gender': 'NonBinary'}]

with open('data/training_X.pickle', 'rb') as f:
    training_X = pickle.load(f)
with open('data/training_y.pickle', 'rb') as f:
    training_y = pickle.load(f)
with open('data/testing_X.pickle', 'rb') as f:
    testing_X = pickle.load(f)
with open('data/testing_y.pickle', 'rb') as f:
    testing_y = pickle.load(f)

training = pd.concat([training_X, training_y])
testing = pd.concat([testing_X, testing_y])

training_bld = aif360.datasets.BinaryLabelDataset(df=training, label_names=['label'], protected_attribute_names=['Gender'])
testing_bld = aif360.datasets.BinaryLabelDataset(df=testing, label_names=['label'], protected_attribute_names=['Gender'])
metric_before = aif360.metrics.BinaryLabelDatasetMetric(training_bld, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
mean_diff_before = metric_before.mean_difference()
print(f"Training mean difference before: {mean_diff_before}")

RW = aif360.algorithms.preprocessing.Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
training_transformed = RW.fit_transform(training_bld)

metric_after = aif360.metrics.BinaryLabelDatasetMetric(training_transformed, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
mean_diff_after = metric_after.mean_difference()
print(f"Training mean difference after: {mean_diff_after}")

with open('data/control-model.pickle', 'rb') as f:
    control_network = pickle.load(f)
with open('data/fair-model.pickle', 'rb') as f:
    fair_network = pickle.load(f)
   
predictions_before = torch.argmax(control_network(testing_bld.features), dim=1).numpy()
predictions_after = torch.argmax(fair_network(testing_bld.features), dim=1).numpy()

before_pred = testing_bld.copy()
before_pred.labels = predictions_before
after_pred = testing_bld.copy()
after_pred.labels = predictions_after

class_metrics_before = aif360.metrics.ClassificationMetric(testing_bld, before_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
class_metrics_after = aif360.metrics.ClassificationMetric(testing_bld, after_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)