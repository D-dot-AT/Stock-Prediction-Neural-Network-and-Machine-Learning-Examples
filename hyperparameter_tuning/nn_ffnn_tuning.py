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
from hyperparameter_tuning.config import Hyper, hyperparameter_values, EXPLORE_ALL_COMBINATIONS, \
    NUMBER_OF_COMBINATIONS_TO_TRY, CPU_COUNT, RERUN_COUNT
from hyperparameter_tuning.get_ffnn import get_ffnn
from hyperparameter_tuning.load_data import load_data

# Reducing verbosity of output
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# loading data
train_dataset, X_test_scaled, Y_test, input_feature_size = load_data()

# Generate all the combinations of hyperparameter values
all_combinations = list(product(*hyperparameter_values.values()))

# Grid search or Random search
if EXPLORE_ALL_COMBINATIONS:
    combinations_to_try = all_combinations
else:
    # Get a random subset of combinations
    combinations_to_try = random.sample(all_combinations, min(NUMBER_OF_COMBINATIONS_TO_TRY, len(all_combinations)))


def run_model(model_class, params):
    model = model_class()

    train_loader = DataLoader(train_dataset, batch_size=params[Hyper.BATCH_SIZE])

    trainer = L.Trainer(max_epochs=params[Hyper.MAX_EPOCHS], logger=False)
    trainer.fit(model, train_loader)

    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X_test_scaled)).numpy()

    predictions_bin = (predictions > PREDICTION_THRESHOLD).astype(int)

    # calculating p-value
    tn, fp, fn, tp = confusion_matrix(Y_test, predictions_bin).ravel()
    p_value = calculate_precision_p_value(tp=tp, fp=fp, fn=fn, tn=tn)

    return p_value


def evaluate_wrapper(args):
    iteration, values = args

    # converting tuples to dict
    params = {key: value for key, value in zip(hyperparameter_values.keys(), values)}

    try:
        start_time = time.time()
        model_class = get_ffnn(params, input_feature_size)

        # running multiple times to get the median value
        # this helps account for random performance variation
        p_values = [run_model(model_class, params) for _ in range(RERUN_COUNT)]
        p_value = median(p_values)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Iteration: {iteration} of {len(combinations_to_try)}")
        print(f"Parameters: {params}")
        print(f"P-value: {p_value}")
        return params, p_value, execution_time, None
    except Exception as err:
        return params, None, None, str(err)


def iterate_hyperparameters():
    # Initialize lists to store the results and errors
    results = []
    errors = []

    # Setting pools across our CPUs
    pool = Pool(CPU_COUNT)

    # Enumerate the combinations to track the iteration number
    params_with_iter = list(enumerate(combinations_to_try, 1))

    # Using Pool to parallelize
    for params, p_value, execution_time, error in list(pool.map(evaluate_wrapper, params_with_iter)):

        if error is None:
            params.update({'p_value': p_value, 'execution_time': execution_time})
            results.append(params)
        else:
            params.update({'error': error})
            errors.append(params)

    # Finding the parameters that produced the lowest p-value
    best_result = min(results, key=lambda x: x['p_value'])
    print(f"The best parameters are: {best_result} with a p-value of {best_result['p_value']}")

    # Sorting the results by p-value in ascending order
    sorted_results = sorted(results, key=lambda x: x['p_value'])

    # Saving the sorted results to a CSV file
    with open('results/hyperparameter_results.csv', 'w', newline='') as csvfile:
        fieldnames = list(hyperparameter_values.keys()) + ['p_value', 'execution_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in sorted_results:
            writer.writerow(result)

    # If any errors occurred, record those
    if len(errors) > 0:
        with open('results/errors.csv', 'w', newline='') as errorFile:
            fieldnames = list(hyperparameter_values.keys()) + ['error']
            writer = csv.DictWriter(errorFile, fieldnames=fieldnames)
            writer.writeheader()
            for error in errors:
                writer.writerow(error)

    # Close the pool
    pool.close()
    pool.join()


# running the iterations
if __name__ == '__main__':
    iterate_hyperparameters()
