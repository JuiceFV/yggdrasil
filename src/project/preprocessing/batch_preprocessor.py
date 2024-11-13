import abc

import torch
from torch import nn

from project.core.dtypes.base import TensorDataClass


class BatchPreprocessor(nn.Module, abc.ABC):
    r"""
    Abstract class for batch preprocessors. It's applied on collate stage.
    The :meth:`forward` method should return an instance of :class:`TensorDataClass`.
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(self, batch: dict[str, torch.Tensor]) -> TensorDataClass:
        r"""
        Defines batch preprocessing logic.

        Args:
            batch (dict[str, torch.Tensor]): Collated batch of data.

        Returns:
            TensorDataClass: Preprocessed batch in suitable format.
        """


def batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    r"""
    Move all tensors in a batch to a specified device.

    Args:
        batch (dict[str, torch.Tensor]): Collated batch of data.
        device (torch.device): The device to move tensors to.

    Returns:
        dict[str, torch.Tensor]: Batch with tensors moved to the specified device.
    """
    return {key: value.to(device) for key, value in batch.items()}
