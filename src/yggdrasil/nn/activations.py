from enum import Enum

from torch import nn


class ActivationType(Enum):
    relu = nn.ReLU
    sigmoid = nn.Sigmoid
    tanh = nn.Tanh
    prelu = nn.PReLU
    leaky_relu = nn.LeakyReLU
    gelu = nn.GELU
    elu = nn.ELU
    selu = nn.SELU
    softmax = nn.Softmax
    log_softmax = nn.LogSoftmax
    identity = nn.Identity
