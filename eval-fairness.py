import csv
from pathlib import Path
import pickle

import pandas as pd
import numpy as np
import aif360.metrics, aif360.algorithms.preprocessing, aif360.datasets, aif360.data
import torch

import neuralnet as nn

with open('data/control-model.pickle', 'rb') as f:
    control_network = pickle.load(f)
with open('data/fair-model.pickle', 'rb') as f:
    fair_network = pickle.load(f)

training = pd.read_csv("data/training.csv").dropna()
testing = pd.read_csv("data/testing.csv").dropna()
training, testing = nn.preprocess_fairness(training, testing)
training_X, training_y, testing_X, testing_y = nn.split_labels(training, testing)
testing_bld = aif360.datasets.BinaryLabelDataset(df=testing, label_names=['label'], protected_attribute_names=['Gender_Woman', 'Gender_NonBinary'])

testing_tensor = torch.from_numpy(testing_X.values).float()
if torch.cuda.is_available():
    device = torch.device('cuda')
    testing_tensor = testing_tensor.to(device)
    control_network = control_network.to(device)
    fair_network = fair_network.to(device)

predictions_before = torch.argmax(control_network(testing_tensor), dim=1).cpu().numpy()
predictions_after = torch.argmax(fair_network(testing_tensor), dim=1).cpu().numpy()

before_pred = testing_bld.copy()
before_pred.labels = np.expand_dims(predictions_before, axis=1)
after_pred = testing_bld.copy()
after_pred.labels = np.expand_dims(predictions_after, axis=1)

privileged_groups = [{'Gender_Woman': 0, 'Gender_NonBinary': 0}]
unprivileged_groups = [{'Gender_Woman': 1}, {'Gender_NonBinary': 1}]
class_metrics_before = aif360.metrics.ClassificationMetric(testing_bld, before_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
class_metrics_after = aif360.metrics.ClassificationMetric(testing_bld, after_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

before_acc = class_metrics_before.accuracy()
before_erd = class_metrics_before.error_rate_difference()
before_di = class_metrics_before.disparate_impact()
after_acc = class_metrics_after.accuracy()
after_erd = class_metrics_after.error_rate_difference()
after_di = class_metrics_after.disparate_impact()
print("Transformed Dataset?\t\tAccuracy\tError Rate Diff\t\tDisparate Impact")
print(f"No\t\t\t\t{before_acc:.4f}\t\t{before_erd:.4f}\t\t\t{before_di:.4f}")
print(f"Yes\t\t\t\t{after_acc:.4f}\t\t{after_erd:.4f}\t\t\t{after_di:.4f}")