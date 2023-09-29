
from torch import nn


ACTIVATION_FUNCTIONS_ALL = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU, nn.LogSigmoid,
                            nn.MultiheadAttention, nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU,
                            nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink,
                            nn.Threshold, nn.GLU, nn.Softmin, nn.Softmax, nn.Softmax2d, nn.LogSoftmax,
                            nn.AdaptiveLogSoftmaxWithLoss]

ACTIVATION_FUNCTIONS_WORKING = [nn.ELU, nn.Hardshrink, nn.Hardsigmoid, nn.Hardtanh, nn.Hardswish, nn.LeakyReLU,
                                nn.LogSigmoid,
                                nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU, nn.SELU, nn.CELU, nn.GELU,
                                nn.Sigmoid, nn.SiLU, nn.Mish, nn.Softplus, nn.Softshrink, nn.Softsign, nn.Tanh, nn.Tanhshrink,
                                nn.Softmin, nn.Softmax, nn.LogSoftmax]