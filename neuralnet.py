"""
neuralnet.py
Trains a neural network and evaluates it, or performs a grid search for hyperparameters.
"""


import sys
from pathlib import Path
import pandas as pd
import pandas.api.types
import torch


# Converts the the labels of the dataset into numbers
def convert_labels(dataset):
    labels = dataset["label"].unique()
    
    # Convert each of the string labels into the corresponding number
    for label, number in zip(labels, range(len(labels))):
        dataset.loc[dataset["label"] == label, "label"] = number

    # Make sure the column type is a number
    dataset["label"] = pandas.to_numeric(dataset["label"])

# Takes in a dataset and scales all numeric attributes
def scale_dataset(dataset):
    dataset_scaled = dataset.copy()
    for col_name in dataset_scaled.columns:
        if col_name != "label" and pandas.api.types.is_numeric_dtype(dataset_scaled[col_name]):
            col = dataset_scaled[col_name]
            if col.max() != col.min():
                dataset_scaled[col_name] = (col - col.min()) / (col.max() - col.min())

    return dataset_scaled

# Takes in a dataset and converts the categorical attributes into one-hot encodings
def one_hot(dataset):
    dataset_onehot = dataset.copy()
    for column in dataset_onehot.columns:
        if column == "label" or pandas.api.types.is_numeric_dtype(dataset_onehot[column]):
            continue
        onehots = pandas.get_dummies(dataset_onehot[column], column, drop_first=True, dtype=int)
        dataset_onehot = pandas.concat([dataset_onehot.drop(column, axis=1), onehots], axis=1)

    return dataset_onehot

# a method for splitting our data into training and testing
def split_data(data_set, train_percentage, seed):

    # create the training and testing sets
    shuffled = data_set.sample(frac=1, random_state=seed)
    total_rows = shuffled.shape[0]
    training_rows = int(train_percentage * total_rows)
    training = shuffled.iloc[:training_rows, :]
    testing = shuffled.iloc[training_rows:, :]

    # split the training attributes and labels
    training_X = training.drop("label", axis=1)
    training_y = training["label"]
   
    # split the testing attributes and labels
    testing_X = testing.drop("label", axis=1)
    testing_y = testing["label"]

    return training_X, training_y, testing_X, testing_y

# Complete all preprocessing steps
def preprocess(dataset, train_percentage, seed, classification=True, maxmin=True):

    if classification:
        convert_labels(dataset)

    if maxmin:
        scaled = scale_dataset(dataset)
        final = one_hot(scaled)
    else:
        final = one_hot(dataset)

    return split_data(final, train_percentage, seed)

# Split data for fairness transformation
def split_data_fairness(data_set, train_percentage, seed):
    shuffled = data_set.sample(frac=1, random_state=seed)
    total_rows = shuffled.shape[0]
    training_rows = int(train_percentage * total_rows)
    training = shuffled.iloc[:training_rows, :]
    testing = shuffled.iloc[training_rows:, :]

    return training, testing

# Preprocess data after fairness transformation
def preprocess_fairness(training, testing, classification=True, maxmin=True):

    if classification:
        convert_labels(training)
        convert_labels(testing)

    if maxmin:
        scaled_train = scale_dataset(training)
        scaled_test = scale_dataset(testing)
        final_train = one_hot(scaled_train)
        final_test = one_hot(scaled_test)
    else:
        final_train = one_hot(training)
        final_test = one_hot(testing)

    # CODE ONLY WORKS FOR SPECIFIC DATASET
    final_train.drop(["Gender_Woman", "Gender_NonBinary"], axis=1)
    final_test.drop(["Gender_Woman", "Gender_NonBinary"], axis=1)

    return final_train, final_test

# Split labels from dataset
def split_labels(training, testing):
    training_X = training.drop("label", axis=1)
    training_y = training["label"]
    testing_X = testing.drop("label", axis=1)
    testing_y = testing["label"]

    return training_X, training_y, testing_X, testing_y

# Creates a neural network with one hidden layer with a given number of attributes and labels
def create_network(n_input, n_hidden, n_output, classification=True):
    hidden_layer = [
        torch.nn.Linear(n_input, n_hidden),
        torch.nn.Sigmoid(),
    ]

    if classification:
        output_layer = [
            torch.nn.Linear(n_hidden, n_output),
            torch.nn.Softmax(dim=1),
        ]
    else:
        output_layer = [
            torch.nn.Linear(n_hidden, n_output),
        ]
            
    all_layers = hidden_layer + output_layer

    network = torch.nn.Sequential(*all_layers)

    return network

# Converts a training set into smaller train and validation sets
def create_validation(training_X, training_y, valid_percentage):
    training_n = training_X.shape[0]
    valid_rows = int(valid_percentage * training_n)

    valid_X = training_X.iloc[:valid_rows]
    valid_y = training_y.iloc[:valid_rows]

    train_X = training_X.iloc[valid_rows:]
    train_y = training_y.iloc[valid_rows:]

    return train_X, train_y, valid_X, valid_y

# Calculates the accuracy of the predictions of a neural network
def calculate_accuracy(network, X, y):
    if type(X) is not torch.Tensor:
        X = torch.from_numpy(X.values).float()
        y = torch.from_numpy(y.values).long()

    softmax_probs = network(X)    
    predictions = torch.argmax(softmax_probs, dim=1)

    accuracy = sum(predictions == y) / len(predictions)
    return float(accuracy)

def calculate_MAE(network, X, y):
    if type(X) is not torch.Tensor:
        X = torch.from_numpy(X.values).float()
        y = torch.from_numpy(y.values).float()

    predictions = network(X)    
    
    sum = 0
    for pair in zip(predictions, y):
        sum += abs(pair[0] - pair[1])
    mae = sum / len(y)

    return float(mae)

# Trains a neural network with given training data
def train_network(network, training_X, training_y, learning_rate, classification=True):
    train_X, train_y, valid_X, valid_y = create_validation(training_X, training_y, 0.2)
    
    train_X = torch.from_numpy(train_X.values).float()
    train_y = torch.from_numpy(train_y.values).long() if classification else torch.from_numpy(train_y.values).float()
    valid_X = torch.from_numpy(valid_X.values).float()
    valid_y = torch.from_numpy(valid_y.values).long() if classification else torch.from_numpy(valid_y.values).float()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        
        train_X = train_X.to(device)
        train_y = train_y.to(device)
        valid_X = valid_X.to(device)
        valid_y = valid_y.to(device)      
        network = network.to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    if classification:
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        loss_function = torch.nn.MSELoss()

    num_epochs = 1000
    progress = []
    for epoch in range(num_epochs):
        train_predictions = network(train_X).squeeze(-1)
        valid_predictions = network(valid_X).squeeze(-1)

        train_loss = loss_function(train_predictions, train_y)
        training_loss = train_loss.item()

        valid_loss = loss_function(valid_predictions, valid_y)
        validation_loss = valid_loss.item()

        if classification:
            training_acc = calculate_accuracy(network, train_X, train_y)
            validation_acc = calculate_accuracy(network, valid_X, valid_y)
        else:
            training_mae = calculate_MAE(network, train_X, train_y)
            validation_mae = calculate_MAE(network, valid_X, valid_y)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if classification:
            progress.append({
                'epoch': epoch,
                'train_loss': training_loss,
                'valid_loss': validation_loss,
                'train_acc': training_acc,
                'valid_acc': validation_acc
            })

            if validation_acc == 1:
                break

        else:
            progress.append({
                'epoch': epoch,
                'train_loss': training_loss,
                'valid_loss': validation_loss,
                'train_mae': training_mae,
                'valid_mae': validation_mae
            })

            if validation_mae == 0:
                break
    
    return pandas.DataFrame(progress)

def eval_mode(dataset_path, dataset_name, train_percentage, seed, learning_rate, n_hidden):

    # Read in and preprocess data
    dataset = pd.read_csv(dataset_path)
    classification = not pandas.api.types.is_numeric_dtype(dataset["label"])
    training_X, training_y, testing_X, testing_y = preprocess(dataset, train_percentage, seed, classification)

    # Train a neural network
    n_input = training_X.shape[1]
    n_output = len(training_y.unique()) if classification else 1
    network = create_network(n_input, n_hidden, n_output, classification)
    train_network(network, training_X, training_y, learning_rate, classification)

    # Calculate performance
    if classification:
        performance = calculate_accuracy(network, testing_X, testing_y)
    else:
        performance = calculate_MAE(network, testing_X, testing_y)

    # Output results
    if not Path("results.csv").is_file():
        with open(Path("results.csv"), 'w') as results_file:
            results_file.write("Dataset,Rate,Neurons,Performance\n")
    with open(Path("results.csv"), 'a') as results_file:
        results_file.write(f"{dataset_name},{learning_rate},{n_hidden},{performance}\n")

def gridsearch_mode(dataset_path, dataset_name, train_percentage, seed, VERBOSE=False):

    # Preprocess dataset
    dataset = pd.read_csv(dataset_path)
    classification = True # not pandas.api.types.is_numeric_dtype(dataset["label"])
    training_X, training_y, testing_X, testing_y = preprocess(dataset, train_percentage, seed, classification)

    # Perform grid search
    learning_rates = [0.001, 0.01, 0.1]
    neurons = [32, 64, 128, 256]
    for rate in learning_rates:
        if VERBOSE:
            print(rate)
        for n_hidden in neurons:
            if VERBOSE:
                print(n_hidden)

            # Train a neural network
            n_input = training_X.shape[1]
            n_output = len(training_y.unique()) if classification else 1
            network = create_network(n_input, n_hidden, n_output, classification)
            progress = train_network(network, training_X, training_y, rate, classification)

            # Calculate performance
            if classification:
                performance = progress['valid_acc'].iloc[-1]
            else:
                performance = progress['valid_mae'].iloc[-1]

            # Output results
            if not Path("gridsearch.csv").is_file():
                with open(Path("gridsearch.csv"), 'w') as results_file:
                    results_file.write("Dataset,Rate,Neurons,Performance\n")
            with open(Path("gridsearch.csv"), 'a') as results_file:
                results_file.write(f"{dataset_name},{rate},{n_hidden},{performance}\n")


def main():
    
    args = sys.argv
    if len(args) == 6: # Evaluation mode

        # Check program arguments
        if not Path(args[1]).is_file():
            raise FileNotFoundError("Data set file not found")
        else:
            dataset_path = Path(args[1])
            dataset_name = args[1].split(".")[0]
        try:
            train_percentage = float(args[2])
        except:
            raise TypeError("Invalid training percentage, should be a decimal")
        if not args[3].isnumeric():
            raise TypeError("Invalid seed, should be an integer")
        else:
            seed = int(args[3])
        try:
            learning_rate = float(args[4])
        except:
            raise TypeError("Invalid learning rate, should be a decimal")
        if not args[5].isnumeric():
            raise TypeError("Invalid number of hidden neurons, should be an integer")
        else:
            n_hidden = int(args[5])

        eval_mode(dataset_path, dataset_name, train_percentage, seed, learning_rate, n_hidden)

    elif len(args) == 4: # Hyperparameter search mode
        
        # Check program arguments
        if not Path(args[1]).is_file():
            raise FileNotFoundError("Data set file not found")
        else:
            dataset_path = Path(args[1])
            dataset_name = args[1].split(".")[0]
        try:
            train_percentage = float(args[2])
        except:
            raise TypeError("Invalid training percentage, should be a decimal")
        if not args[3].isnumeric():
            raise TypeError("Invalid seed, should be an integer")
        else:
            seed = int(args[3])

        gridsearch_mode(dataset_path, dataset_name, train_percentage, seed)

    else:
        raise Exception("Invalid number of arguments")


if __name__ == "__main__":
    main()