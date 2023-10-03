from torch import nn, optim


OPTIMIZER_CLASSES = {
    'Adam': optim.Adam,
    'SGD': optim.SGD,
    'RMSprop': optim.RMSprop,
    'AdamW': optim.AdamW,
    # ... add any other optimizers you want to consider
}

ACTIVATION_FUNCTIONS_ALL = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU,
                            nn.LogSigmoid, nn.MultiheadAttention, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU,
                            nn.CELU, nn.GELU, nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign,
                            nn.Tanh, nn.Tanhshrink, nn.Threshold, nn.GLU, nn.Softmin, nn.Softmax, nn.Softmax2d,
                            nn.LogSoftmax, nn.AdaptiveLogSoftmaxWithLoss]

ACTIVATION_FUNCTIONS_WORKING = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU,
                                nn.LogSigmoid, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU,
                                nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh,
                                nn.Tanhshrink, nn.Softmin, nn.Softmax, nn.LogSoftmax]


WEIGHT_INITIALIZATIONS = {
    'xavier_uniform': nn.init.xavier_uniform_,
    'xavier_normal': nn.init.xavier_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'orthogonal': nn.init.orthogonal_,
    'zeros': nn.init.zeros_,
    'ones': nn.init.ones_,
    'constant': nn.init.constant_,
    'eye': nn.init.eye_,
    'sparse': nn.init.sparse_,
    'normal': nn.init.normal_,
    'uniform': nn.init.uniform_,
    'dirac': nn.init.dirac_,
    # ... add other initializations as needed
}
