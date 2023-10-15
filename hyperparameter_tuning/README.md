# Hyperparameter Optimization for Neural Networks

This script provides a mechanism to search for optimial hyperparameters for training neural networks.
It supports both grid search (exploring all combinations) and random search.

## Hyperparameters Explored

Here are the hyperparameters we currently search across:

- **Learning Rate**: The step size at each iteration while moving towards a minimum of the loss function.
- **Max Epochs**: The maximum number of times the learning algorithm will work through the entire training dataset.
- **Batch Size**: The number of training examples utilized in one iteration.
- **Hidden Layers**: The architecture of the neural network in terms of layers and nodes.
- **Loss Function**: Determines the difference between the network's predictions and the actual data.
- **Activation Function**: The function used to introduce non-linearity to the network.
- **Optimizer**: Algorithms or methods used to change the attributes of the neural network such as weights to reduce the
  losses.
- **Dropout**: A regularization method where input and recurrent connections to a layer are probabilistically excluded
  from during training.
- **L1 Regularization**: L1 regularization adds a penalty for non-zero coefficients.
- **L2 Regularization**: L2 regularization adds a penalty for larger coefficient values.
- **Weight Initialization**: Methods to set the initial random weights of neural network layers.

## Hyperparameter Values

In `convig.py` the `hyperparameter_values` dictionary in the code outlines the specific values and ranges we're
exploring for each hyperparameter. Adjust the values in this dictionary to search across different hyperparameters.

## Search Strategy

- **Grid Search**: If `EXPLORE_ALL_COMBINATIONS` is set to `True`, the script will exhaustively explore all combinations
  of hyperparameter values.

- **Random Search**: If `EXPLORE_ALL_COMBINATIONS` is set to `False`, the script will randomly sample from the
  hyperparameter space. The number of random combinations sampled is set by `NUMBER_OF_COMBINATIONS_TO_TRY`.

## Performance Ranking
All models were trained and tested on the `example_data` from [D.AT](https://d.at/ref/github-python-examples)
to rank which performed best.

Precision p-value is the method for comparing performance.  Why precision?
When investing, you care much more about the performance of the stocks you have purchased
than those that you decided not to buy. Put in real terms, the 10,000 no-buy decisions (negatives)
are not nearly as important as the 50 buy decisions (positives) if all 50 have profitable exits (true positives).
P-values are chosen as the measure because a low p-value rewards for both high precision and a large
amount of true positives, connoting a more robust model.

## Robustness

Due to the stochastic nature of neural network training, we rerun each hyperparameter combination several times (as
defined by `RERUN_COUNT`) and pick the median performance. This helps reduce the impact of random chance in evaluating
performance.

## Parallel Execution

The script is capable of parallel execution, utilizing multiple CPUs for hyperparameter search. The number of CPUs
dedicated to the search is set by `CPU_COUNT`.

## Modifying the Code

To customize the hyperparameter search:

1. Update the `hyperparameter_values` dictionary with desired values/ranges.
2. Adjust search strategy settings (`EXPLORE_ALL_COMBINATIONS` and `NUMBER_OF_COMBINATIONS_TO_TRY`).
3. Set the `RERUN_COUNT` to define how many times each hyperparameter combination should be rerun.
4. Adjust the `CPU_COUNT` to allocate more or fewer CPUs to the search.

---

Run the script and results will be saved it the `results` dir. Evaluate the performance across different hyperparameter
combinations to find the optimal network configuration for your task.