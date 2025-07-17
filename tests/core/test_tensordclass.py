import unittest
from dataclasses import dataclass

import pytest
import torch

from yggdrasil.core.dtypes.base import TensorDataClass


@dataclass
class MyTensorType(TensorDataClass):
    dense_features: torch.Tensor
    mask: torch.Tensor | None = None


class TestConfigParser:
    def test_tensor_data_class_getattr(self) -> None:
        t = MyTensorType(torch.Tensor(1, 3), torch.ones(1, 3, dtype=torch.bool))
        assert t.size().dense_features == torch.Size([1, 3])
        assert t.size().mask == torch.Size([1, 3])

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_tensor_data_class_cuda(self) -> None:
        t = MyTensorType(torch.Tensor(1, 3), torch.ones(1, 3, dtype=torch.bool))
        t_cuda = t.cuda()
        assert t_cuda.dense_features.is_cuda
        assert t_cuda.mask.is_cuda  # type: ignore

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_tensor_data_class_cpu(self) -> None:
        t = MyTensorType(torch.Tensor(1, 3).cuda(), torch.ones(1, 3, dtype=torch.bool).cuda())
        t_cpu = t.cpu()
        assert not t_cpu.dense_features.is_cuda
        assert not t_cpu.mask.is_cuda  # type: ignore

    def test_tensor_data_class_non_tensor_attribute(self) -> None:
        t = MyTensorType(torch.Tensor(1, 3), torch.ones(1, 3, dtype=torch.bool))
        with pytest.raises(AttributeError):
            t.non_existent_method()

    def test_tensor_data_class_non_callable_attribute(self) -> None:
        t = MyTensorType(torch.Tensor(1, 3), torch.ones(1, 3, dtype=torch.bool))
        with pytest.raises(TypeError):
            _ = t.dtype
