import csv
import logging
import random
import time
from itertools import product
from multiprocessing import Pool
from statistics import median

import pytorch_lightning as L
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from common import calculate_precision_p_value, PREDICTION_THRESHOLD
from hyperparameter_tuning.config import (
    Hyper, hyperparameter_values, EXPLORE_ALL_COMBINATIONS,
    NUMBER_OF_COMBINATIONS_TO_TRY, CPU_COUNT, RERUN_COUNT
)
from hyperparameter_tuning.get_ffnn import get_ffnn
from hyperparameter_tuning.load_data import load_data

# Configure logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# Load data
train_dataset, X_test_scaled, Y_test, input_feature_size = load_data()


def get_hyperparameter_combinations():
    """Generate hyperparameter combinations, either all or a subset."""
    all_combinations = list(product(*hyperparameter_values.values()))

    print(f"Number of combinations: {len(all_combinations)}")

    if EXPLORE_ALL_COMBINATIONS:
        return all_combinations

    # Get a random subset of combinations
    return random.sample(all_combinations, min(NUMBER_OF_COMBINATIONS_TO_TRY, len(all_combinations)))


def run_model_for_hyperparameters(params, model_class):
    """Train the model with given hyperparameters and return the p-value."""
    train_loader = DataLoader(train_dataset, batch_size=params[Hyper.BATCH_SIZE])
    model = model_class()
    trainer = L.Trainer(max_epochs=params[Hyper.MAX_EPOCHS], logger=False)
    trainer.fit(model, train_loader)

    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test_scaled)).numpy()
    predictions_bin = (predictions > PREDICTION_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(Y_test, predictions_bin).ravel()
    return calculate_precision_p_value(tp=tp, fp=fp, fn=fn, tn=tn)


def evaluate_hyperparameters(args):
    """Evaluate the model with given hyperparameters."""
    iteration, values = args
    params = dict(zip(hyperparameter_values.keys(), values))
    start_time = time.time()

    try:
        model_class = get_ffnn(params, input_feature_size)
        p_values = [run_model_for_hyperparameters(params, model_class) for _ in range(RERUN_COUNT)]
        p_value_median = median(p_values)
        return params, p_value_median, time.time() - start_time, None  # Calculate elapsed time here
    except Exception as err:
        return params, None, None, str(err)


def store_results(results, errors):
    """Store results and errors to CSV files."""

    # Sort the results by p-value in ascending order and save to CSV
    sorted_results = sorted(results, key=lambda x: x['p_value'])

    # Converting values to strings for easier readability
    sorted_results = [{str(key): str(value) for key, value in data.items()} for data in sorted_results]
    errors = [{str(key): str(value) for key, value in data.items()} for data in errors]

    with open('results/hyperparameter_results.csv', 'w', newline='') as csvfile:
        fieldnames = [str(key) for key in hyperparameter_values.keys()] + ['p_value', 'execution_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_results)

    # If there were errors, save them too
    if errors:
        with open('results/errors.csv', 'w', newline='') as errorFile:
            fieldnames = [str(key) for key in hyperparameter_values.keys()] + ['error']
            writer = csv.DictWriter(errorFile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(errors)


def iterate_hyperparameters():
    """Main function to iterate over hyperparameters."""
    results = []
    errors = []

    with Pool(CPU_COUNT) as pool:
        combinations = get_hyperparameter_combinations()
        for params, p_value, exec_time, error in pool.map(evaluate_hyperparameters, enumerate(combinations, 1)):
            result = {**params, 'p_value': p_value, 'execution_time': exec_time}
            if error is None:
                results.append(result)
            else:
                errors.append({**params, 'error': error})

    best_result = min(results, key=lambda x: x['p_value'])
    print(f"The best parameters are: {best_result} with a p-value of {best_result['p_value']}")

    store_results(results, errors)


if __name__ == '__main__':
    iterate_hyperparameters()
