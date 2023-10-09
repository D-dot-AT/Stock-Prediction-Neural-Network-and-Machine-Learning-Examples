from enum import Enum

import torch.optim as optim
from torch import nn


# Define an enum for hyperparameter keys to improve readability
class Hyper(Enum):
    LEARNING_RATE = 'Learning Rate'
    MAX_EPOCHS = 'Max Epochs'
    BATCH_SIZE = 'Batch Size'
    HIDDEN_LAYERS = 'Hidden Layers'
    LOSS_FUNCTION = 'Loss Function'
    ACTIVATION_FUNCTION = 'Activation Function'
    OPTIMIZER = 'Optimizer'
    DROPOUT = 'Dropout'
    L1_REGULARIZATION = 'L1 Regularization'
    L2_REGULARIZATION = 'L2 Regularization'
    WEIGHT_INITIALIZATION = 'Weight Initialization'


# Define an enum for optimizer keys
class Optimizer(Enum):
    ADAM = 'Adam'
    SGD = 'SGD'
    RMSPROP = 'RMSprop'
    ADAMW = 'AdamW'
    ADAGRAD = 'Adagrad'
    ADADELTA = 'Adadelta'
    ADAMAX = 'Adamax'
    LBFGS = 'LBFGS'
    RPROP = 'Rprop'
    ASGD = 'ASGD'
    FTRL = 'FTRL'


# Optimizer function lookup
OPTIMIZER_CLASSES = {
    Optimizer.ADAM: optim.Adam,
    Optimizer.SGD: optim.SGD,
    Optimizer.RMSPROP: optim.RMSprop,
    Optimizer.ADAMW: optim.AdamW,
    Optimizer.ADAGRAD: optim.Adagrad,
    Optimizer.ADADELTA: optim.Adadelta,
    Optimizer.ADAMAX: optim.Adamax,
    Optimizer.LBFGS: optim.LBFGS,
    Optimizer.RPROP: optim.Rprop,
    Optimizer.ASGD: optim.ASGD,
}


ACTIVATION_FUNCTIONS_ALL = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU,
                            nn.LogSigmoid, nn.MultiheadAttention, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU,
                            nn.CELU, nn.GELU, nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign,
                            nn.Tanh, nn.Tanhshrink, nn.Threshold, nn.GLU, nn.Softmin, nn.Softmax, nn.Softmax2d,
                            nn.LogSoftmax, nn.AdaptiveLogSoftmaxWithLoss]

# The single-node classifications NN we are building works with these activation functions
ACTIVATION_FUNCTIONS_WORKING = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU,
                                nn.LogSigmoid, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU,
                                nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh,
                                nn.Tanhshrink, nn.Softmin, nn.Softmax, nn.LogSoftmax]


# Define an enum for weight initialization keys
class WeightInit(Enum):
    XAVIER_UNIFORM = 'xavier_uniform'
    XAVIER_NORMAL = 'xavier_normal'
    KAIMING_UNIFORM = 'kaiming_uniform'
    KAIMING_NORMAL = 'kaiming_normal'
    ORTHOGONAL = 'orthogonal'
    ZEROS = 'zeros'
    ONES = 'ones'
    CONSTANT = 'constant'
    EYE = 'eye'
    SPARSE = 'sparse'
    NORMAL = 'normal'
    UNIFORM = 'uniform'
    DIRAC = 'dirac'


# Weight initialization function lookup
WEIGHT_INITIALIZATIONS = {
    WeightInit.XAVIER_UNIFORM: nn.init.xavier_uniform_,
    WeightInit.XAVIER_NORMAL: nn.init.xavier_normal_,
    WeightInit.KAIMING_UNIFORM: nn.init.kaiming_uniform_,
    WeightInit.KAIMING_NORMAL: nn.init.kaiming_normal_,
    WeightInit.ORTHOGONAL: nn.init.orthogonal_,
    WeightInit.ZEROS: nn.init.zeros_,
    WeightInit.ONES: nn.init.ones_,
    WeightInit.CONSTANT: nn.init.constant_,
    WeightInit.EYE: nn.init.eye_,
    WeightInit.SPARSE: nn.init.sparse_,
    WeightInit.NORMAL: nn.init.normal_,
    WeightInit.UNIFORM: nn.init.uniform_,
    WeightInit.DIRAC: nn.init.dirac_,
}


class LossFunction(Enum):
    MSE = 'MSE'
    SMOOTH_L1 = 'Smooth L1'
    HUBER = 'Huber'


LOSS_FUNCTIONS = {
    LossFunction.MSE: nn.MSELoss,
    LossFunction.SMOOTH_L1: nn.SmoothL1Loss,
    LossFunction.HUBER: nn.HuberLoss
}