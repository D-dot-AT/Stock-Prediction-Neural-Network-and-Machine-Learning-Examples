import csv
import time
from functools import reduce
from itertools import product
from operator import mul
from statistics import median

import pytorch_lightning as L
import torch
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from common import calculate_precision_p_value
from hyperparameter_tuning.load_data import load_data

########################################################
#
# Parallelization
# Extensibility for hyperparameter types
#
########################################################

# loading data
train_dataset, X_test_scaled, Y_test, input_feature_size = load_data()

# Level to binarize our output.
PREDICTION_THRESHOLD = 0.5

# For stochastic methods such as training neural networks, results will vary.
# if one hyperparameter configuration outperforms another, how do we know
# this is not due to random variation?  One way to reduce this effect is to
# re-run the process multiple times and pick the median performance.
# This variable allows us to set how many times the process is re-run.
# The larger this number, the more random performance variation is reduced.
# However, the larger the number, the longer the execution time.
# Set this to 1 to run each configuration only once.
RERUN_COUNT = 1

# Add to this! If there are more hyperparameter types you'd like to explore.
LEARNING_RATE = 'Learning Rate'
MAX_EPOCHS = 'Max Epochs'
BATCH_SIZE = 'Batch Size'
HIDDEN_LAYERS = 'HIDDEN_LAYERS'

# Modify this!  Add all the possible values you want to explore.
# A word of caution: due to the multiplicative nature of iterations,
# each added value can significantly increase execution time.
hyperparameter_values = {
    LEARNING_RATE: [0.001],
    MAX_EPOCHS: [8],
    BATCH_SIZE: [32],
    HIDDEN_LAYERS: [
        [2],
        [2, 3, 2, 1, 0.5],
        [2, 3, 2, 1, 0.5, .25],
        [2, 1.5, 1],
        [2, 1, 2, 1],
    ]
}

# calculating the total number of combinations of values
total_combinations = reduce(mul, (len(values) for values in hyperparameter_values.values()))

# Returns a NN class based on a set of hyperparameters
def neural_network(params):
    class SimpleNN(L.LightningModule):
        def __init__(self):
            super().__init__()

            # Calculate hidden layer sizes
            sizes = [int(round(input_feature_size * h)) for h in params[HIDDEN_LAYERS]]

            # Dynamically build layers
            all_sizes = [input_feature_size] + sizes + [1]
            self.layers = nn.ModuleList([nn.Linear(all_sizes[i], all_sizes[i + 1])
                                         for i in range(len(all_sizes) - 1)])
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = x.type(self.layers[0].weight.dtype)
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:  # Only apply ReLU to non-last layers
                    x = nn.ReLU()(x)
            x = self.sigmoid(x)
            return x

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = nn.BCELoss()(y_hat, y.view(-1, 1).float())
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=params[LEARNING_RATE])

    return SimpleNN


def run_model(model_class, params):
    model = model_class()

    train_loader = DataLoader(train_dataset, batch_size=params[BATCH_SIZE])

    trainer = L.Trainer(max_epochs=params[MAX_EPOCHS])
    trainer.fit(model, train_loader)

    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test_scaled)).numpy()

    predictions_bin = (predictions > PREDICTION_THRESHOLD).astype(int)

    # calculating p-value
    tn, fp, fn, tp = confusion_matrix(Y_test, predictions_bin).ravel()
    p_value = calculate_precision_p_value(tp=tp, fp=fp, fn=fn, tn=tn)

    return p_value


def evaluate_hyperparameters(params):
    model_class = neural_network(params)

    # running multiple times to get the median value
    # this helps account for random performance variation
    p_values = [run_model(model_class, params) for _ in range(RERUN_COUNT)]
    return median(p_values)


def iterate_hyperparameters():
    # Initialize an empty list to store the results
    results = []

    # Tracking where we are in the exploration
    iteration_count = 0

    # Loop through each combination of hyperparameters and evaluate them
    for values in product(*hyperparameter_values.values()):
        # Create a dictionary with the current combination of values
        params = {key: value for key, value in zip(hyperparameter_values.keys(), values)}

        start_time = time.time()
        p_value = evaluate_hyperparameters(params)
        end_time = time.time()
        execution_time = end_time - start_time
        iteration_count += 1

        # Storing the results in a dictionary
        result_dict = params.copy()
        result_dict.update({'p_value': p_value, 'execution_time': execution_time})
        results.append(result_dict)

        print(f"Iteration: {iteration_count} of {total_combinations}")
        print(f"Parameters: {params}")
        print(f"P-value: {p_value}")

    # Finding the parameters that produced the lowest p-value
    best_result = min(results, key=lambda x: x['p_value'])
    print(f"The best parameters are: {best_result} with a p-value of {best_result['p_value']}")

    # Sorting the results by p-value in ascending order
    sorted_results = sorted(results, key=lambda x: x['p_value'])

    # Saving the sorted results to a CSV file
    with open('hyperparameter_results.csv', 'w', newline='') as csvfile:
        fieldnames = list(hyperparameter_values.keys()) + ['p_value', 'execution_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in sorted_results:
            writer.writerow(result)


# running the iterations
iterate_hyperparameters()
