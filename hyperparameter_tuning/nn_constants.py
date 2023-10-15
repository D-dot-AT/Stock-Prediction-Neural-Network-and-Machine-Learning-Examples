from enum import Enum

import torch.optim as optim
from torch import nn


class StringEnum(Enum):

    def __str__(self):
        return self.value


# Define an enum for hyperparameter keys to improve readability
class Hyper(StringEnum):
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
    LSTM_NUMBER_OF_LAYERS = 'Number of Layers'
    LSTM_HIDDEN_LAYER_SIZE = 'Hidden Layer Size'


# Define an enum for optimizer keys
class Optimizer(StringEnum):
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


class ActivationFunction(StringEnum):
    ELU = "ELU"
    Hardshrink = "Hardshrink"
    Hardsigmoid = "Hardsigmoid"
    Hardtanh = "Hardtanh"
    Hardswish = "Hardswish"
    LeakyReLU = "LeakyReLU"
    LogSigmoid = "LogSigmoid"
    MultiheadAttention = "MultiheadAttention"
    PReLU = "PReLU"
    ReLU = "ReLU"
    ReLU6 = "ReLU6"
    RReLU = "RReLU"
    SELU = "SELU"
    CELU = "CELU"
    GELU = "GELU"
    Sigmoid = "Sigmoid"
    SiLU = "SiLU"
    Mish = "Mish"
    Softplus = "Softplus"
    Softshrink = "Softshrink"
    Softsign = "Softsign"
    Tanh = "Tanh"
    Tanhshrink = "Tanhshrink"
    Threshold = "Threshold"
    GLU = "GLU"
    Softmin = "Softmin"
    Softmax = "Softmax"
    Softmax2d = "Softmax2d"
    LogSoftmax = "LogSoftmax"
    AdaptiveLogSoftmaxWithLoss = "AdaptiveLogSoftmaxWithLoss"


ACTIVATION_FUNCTIONS = {
    ActivationFunction.ELU: nn.ELU,
    ActivationFunction.Hardshrink: nn.Hardshrink,
    ActivationFunction.Hardsigmoid: nn.Hardsigmoid,
    ActivationFunction.Hardtanh: nn.Hardtanh,
    ActivationFunction.Hardswish: nn.Hardswish,
    ActivationFunction.LeakyReLU: nn.LeakyReLU,
    ActivationFunction.LogSigmoid: nn.LogSigmoid,
    ActivationFunction.MultiheadAttention: nn.MultiheadAttention,
    ActivationFunction.PReLU: nn.PReLU,
    ActivationFunction.ReLU: nn.ReLU,
    ActivationFunction.ReLU6: nn.ReLU6,
    ActivationFunction.RReLU: nn.RReLU,
    ActivationFunction.SELU: nn.SELU,
    ActivationFunction.CELU: nn.CELU,
    ActivationFunction.GELU: nn.GELU,
    ActivationFunction.Sigmoid: nn.Sigmoid,
    ActivationFunction.SiLU: nn.SiLU,
    ActivationFunction.Mish: nn.Mish,
    ActivationFunction.Softplus: nn.Softplus,
    ActivationFunction.Softshrink: nn.Softshrink,
    ActivationFunction.Softsign: nn.Softsign,
    ActivationFunction.Tanh: nn.Tanh,
    ActivationFunction.Tanhshrink: nn.Tanhshrink,
    ActivationFunction.Threshold: nn.Threshold,
    ActivationFunction.GLU: nn.GLU,
    ActivationFunction.Softmin: nn.Softmin,
    ActivationFunction.Softmax: nn.Softmax,
    ActivationFunction.Softmax2d: nn.Softmax2d,
    ActivationFunction.LogSoftmax: nn.LogSoftmax,
    ActivationFunction.AdaptiveLogSoftmaxWithLoss: nn.AdaptiveLogSoftmaxWithLoss
}


# The single-node classifications NN we are building works with these activation functions
ACTIVATION_FUNCTIONS_WORKING_ENUMS = [
    ActivationFunction.ELU,
    ActivationFunction.Hardshrink,
    ActivationFunction.Hardsigmoid,
    ActivationFunction.Hardtanh,
    ActivationFunction.Hardswish,
    ActivationFunction.LeakyReLU,
    ActivationFunction.LogSigmoid,
    ActivationFunction.PReLU,
    ActivationFunction.ReLU,
    ActivationFunction.ReLU6,
    ActivationFunction.RReLU,
    ActivationFunction.SELU,
    ActivationFunction.CELU,
    ActivationFunction.GELU,
    ActivationFunction.Sigmoid,
    ActivationFunction.SiLU,
    ActivationFunction.Mish,
    ActivationFunction.Softplus,
    ActivationFunction.Softshrink,
    ActivationFunction.Softsign,
    ActivationFunction.Tanh,
    ActivationFunction.Tanhshrink,
    ActivationFunction.Softmin,
    ActivationFunction.Softmax,
    ActivationFunction.LogSoftmax
]


# Define an enum for weight initialization keys
class WeightInit(StringEnum):
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


class LossFunction(StringEnum):
    MSE = 'MSE'
    SMOOTH_L1 = 'Smooth L1'
    HUBER = 'Huber'


LOSS_FUNCTIONS = {
    LossFunction.MSE: nn.MSELoss,
    LossFunction.SMOOTH_L1: nn.SmoothL1Loss,
    LossFunction.HUBER: nn.HuberLoss
}