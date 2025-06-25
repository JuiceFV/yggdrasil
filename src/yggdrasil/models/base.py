from copy import deepcopy

import torch
from torch import nn

from yggdrasil.core.dtypes.base import TensorDataClass


class BaseModel(nn.Module):
    def input_prototype(
        self,
    ) -> tuple[torch.Tensor, ...] | dict[str, torch.Tensor] | TensorDataClass:
        raise NotImplementedError

    def get_distributed_data_parallel_model(self) -> "BaseModel":
        raise NotImplementedError

    def cpu_model(self) -> "BaseModel":
        return deepcopy(self).cpu()

    def requires_model_parallel(self) -> bool:
        return False
