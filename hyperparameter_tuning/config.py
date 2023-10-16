from multiprocessing import cpu_count

# Neural Network Constants
from hyperparameter_tuning.nn_constants import (
    Hyper,
    WeightInit as Wi,
    Optimizer as Opti,
    LossFunction as Lf,
    ActivationFunction as Af
)

# Hyperparameters Configuration

# Caution: Due to the multiplicative nature of iterations, adding more values can
# significantly increase execution time.
hyperparameter_values = {
    Hyper.LEARNING_RATE: [0.001, 0.0005],
    Hyper.MAX_EPOCHS: [8],
    Hyper.BATCH_SIZE: [32, 64],
    # Define hidden layers based on input layer nodes.
    # E.g., for 30 input nodes, [2, 1, 0.5] gives layers of size 60, 30, 15.
    Hyper.HIDDEN_LAYERS: [
        [2, 3, 2, 1, 0.5],
        [2, 3, 2, 1, 0.5, 0.25]
    ],
    Hyper.LOSS_FUNCTION: [Lf.BCE, Lf.MSE, Lf.HUBER, Lf.SMOOTH_L1],
    Hyper.ACTIVATION_FUNCTION: [Af.LeakyReLU, Af.PReLU, Af.ReLU, Af.Tanh],
    Hyper.OPTIMIZER: [Opti.ADAM, Opti.RMSPROP],
    Hyper.DROPOUT: [0, 0.2],
    Hyper.L1_REGULARIZATION: [0],  # Common values: [0, 0.01, 0.1]
    Hyper.L2_REGULARIZATION: [0],  # Common values: [0, 0.01, 0.1]
    Hyper.WEIGHT_INITIALIZATION: [Wi.XAVIER_UNIFORM, Wi.XAVIER_NORMAL]
}

# Search Strategy Configuration

# Set True for "grid search" or False for "random search"
EXPLORE_ALL_COMBINATIONS = True

# For random search, define the number of combinations to explore
NUMBER_OF_COMBINATIONS_TO_TRY = 100

# Training Variability Reduction
# To account for random variation in training, we can rerun
# the training process multiple times and take the median performance.
# Odd values recommended; 1 means each configuration runs once.
RERUN_COUNT = 3

# System Configuration

# Set number of CPUs for parallel execution; Max is system's cpu_count
CPU_COUNT = cpu_count()
