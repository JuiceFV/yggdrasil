from copy import deepcopy

import torch
from torch import nn

from yggdrasil.core.dtypes.base import TensorDataClass


class BaseModel(nn.Module):
    """
    BaseModel is an abstract base class for all models in Yggdrasil.
    It inherits from :class:`~torch.nn.Module` and provides a common
    interface for model operations.
    """

    def input_prototype(
        self,
    ) -> tuple[torch.Tensor, ...] | dict[str, torch.Tensor] | TensorDataClass:
        r"""
        Returns a prototype input for the model. This method should be
        overridden by subclasses to provide a specific input prototype.

        .. note::

            Usually, this method is used to define the expected input shape
            and type for the model. It is useful for logging the model's
            artifact and for ensuring that the model can handle the expected input.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Returns:
            tuple[torch.Tensor, ...] | dict[str, torch.Tensor] | TensorDataClass: Input prototype for the model.
        """
        raise NotImplementedError

    def get_distributed_data_parallel_model(self) -> "BaseModel":
        r"""
        Returns a distributed data parallel model. This method should be
        overridden by subclasses to provide a specific implementation.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Returns:
            BaseModel: Distributed data parallel model.
        """
        raise NotImplementedError

    def cpu_model(self) -> "BaseModel":
        r"""
        Returns a copy of the model on CPU. This method is useful for
        ensuring that the model can be used on CPU devices.

        Returns:
            BaseModel: Copy of the model on CPU.
        """
        return deepcopy(self).cpu()

    def requires_model_parallel(self) -> bool:
        r"""
        Checks if the model requires model parallelism.

        Returns:
            bool: True if the model requires model parallelism, False otherwise.
        """
        return False
