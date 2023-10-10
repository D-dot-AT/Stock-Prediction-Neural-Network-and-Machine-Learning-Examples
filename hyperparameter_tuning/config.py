from multiprocessing import cpu_count

from torch import nn

from hyperparameter_tuning.nn_constants import Hyper, WeightInit, Optimizer, LossFunction

# Modify this! Add all the possible values you want to explore.
# A word of caution: due to the multiplicative nature of iterations,
# each added value can significantly increase execution time.
hyperparameter_values = {
    Hyper.LEARNING_RATE: [0.001, 0.0005],
    Hyper.MAX_EPOCHS: [8],
    Hyper.BATCH_SIZE: [32, 64],
    # Hidden layers are defined as lists of numbers, where each number is a multiple of the number of
    # the number of nodes in the input layer.  For example, if we have 30 input nodes and one of
    # our layer lists is [2, 1, 0.5], that will produce hidden layers of size 60, 30, 15.
    Hyper.HIDDEN_LAYERS: [
        [2, 3, 2, 1, 0.5],
        [2, 3, 2, 1, 0.5, 0.25],
    ],
    Hyper.LOSS_FUNCTION: [LossFunction.MSE, LossFunction.SMOOTH_L1, LossFunction.HUBER],
    Hyper.ACTIVATION_FUNCTION: [nn.LeakyReLU, nn.PReLU, nn.ReLU, nn.Tanh],
    Hyper.OPTIMIZER: [Optimizer.ADAM, Optimizer.RMSPROP],
    Hyper.DROPOUT: [0, 0.2, 0.5],
    Hyper.L1_REGULARIZATION: [0],  # [0, 0.01, 0.1],
    Hyper.L2_REGULARIZATION: [0],  # [0, 0.01, 0.1],
    Hyper.WEIGHT_INITIALIZATION: [WeightInit.XAVIER_UNIFORM, WeightInit.XAVIER_NORMAL]
}

# Do you want to explore all combinations or a randomly selected subset?
# Searching all combinations is known as "grid search", a randomly selected subset is known as "random search"
EXPLORE_ALL_COMBINATIONS = True

# If we are doing a random search (EXPLORE_ALL_COMBINATIONS is `False`)
# then how many combinations do you want to try?
NUMBER_OF_COMBINATIONS_TO_TRY = 100

# Neural network training involves a stochastic process, so results will vary.
# If one hyperparameter configuration outperforms another, how do we know
# this is not due to random variation?  One way to reduce random effect is to
# re-run the process multiple times and pick the median performance.
# This variable allows us to set how many times the process is re-run.
# The larger this number, the more random performance variation is reduced.
# However, the larger the number, the longer the execution time.
# Set to some odd number; 1 to run each configuration only once.
RERUN_COUNT = 5

# The number of CPUs to dedicate to this.  You can set this.  Minimum 1, Max is cpu_count
CPU_COUNT = cpu_count()
