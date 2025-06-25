from enum import Enum
from functools import partial

from torch import nn


class InitializationType(Enum):
    kaiming_normal = partial(nn.init.kaiming_normal_)
    kaiming_uniform = partial(nn.init.kaiming_uniform_)
    xavier_normal = partial(nn.init.xavier_normal_)
    xavier_uniform = partial(nn.init.xavier_uniform_)
    normal = partial(nn.init.normal_)
    uniform = partial(nn.init.uniform_)
    constant = partial(nn.init.constant_)
    ones = partial(nn.init.ones_)
    zeros = partial(nn.init.zeros_)
    eye = partial(nn.init.eye_)
    dirac = partial(nn.init.dirac_)
