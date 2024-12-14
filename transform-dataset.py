import pandas as pd
import aif360.metrics, aif360.algorithms.preprocessing, aif360.datasets, aif360.data

import neuralnet as nn

privileged_groups = [{'Gender_Woman': 0, 'Gender_NonBinary': 0}]
unprivileged_groups = [{'Gender_Woman': 1}, {'Gender_NonBinary': 1}]

training = pd.read_csv("data/training.csv").dropna()
testing = pd.read_csv("data/testing.csv").dropna()
training, testing = nn.preprocess_fairness(training, testing)

training_bld = aif360.datasets.BinaryLabelDataset(df=training, label_names=['label'], protected_attribute_names=['Gender_Woman', 'Gender_NonBinary'])
testing_bld = aif360.datasets.BinaryLabelDataset(df=testing, label_names=['label'], protected_attribute_names=['Gender_Woman', 'Gender_NonBinary'])
metric_before = aif360.metrics.BinaryLabelDatasetMetric(training_bld, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
mean_diff_before = metric_before.mean_difference()
print(f"Training mean difference before: {mean_diff_before}")

RW = aif360.algorithms.preprocessing.Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
training_transformed = RW.fit_transform(training_bld)

metric_after = aif360.metrics.BinaryLabelDatasetMetric(training_transformed, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
mean_diff_after = metric_after.mean_difference()
print(f"Training mean difference after: {mean_diff_after}")

transformed_df = training_transformed.convert_to_dataframe()[0]
transformed_df.to_csv("data/transformed.csv", index=False)