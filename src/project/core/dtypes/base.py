from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import torch

from project.core.base_dclass import BaseDataClass
from project.utils import init_logger

log = init_logger(__name__)


@dataclass
class TensorDataClass(BaseDataClass):
    r"""
    The base data structure represents n-dimensional tensor-based data.
    Generally, we don't need the internal :class:`torch.Tensor` implementation
    to represent tensor-based data, i.e. the explicit interface is enough.
    If a structure has multiple :class:`torch.Tensor` fields then an attribute
    call will be applied to each one.

    Example::

        @dataclass
        class MyTensorType(TensorDataClass):
            dense_features: torch.Tensor
            mask: Optional[torch.Tensor] = None

        t = MyTensorType(torch.Tensor(1, 3), torch.ones(1,3, dtype=torch.bool))
        t.is_shared() # MyTensorType(dense_features=False, mask=False)
    """

    def __getattr__(self, __name: str) -> Callable[..., Self]:
        if __name.startswith("__") and __name.endswith("__"):
            msg = "We don't wanna call superprivate method of torch.Tensor"
            raise AttributeError(msg)

        tensor_attr = getattr(torch.Tensor, __name, None)

        if tensor_attr is None:
            msg = f"{self.__class__.__name__} doesn't have {__name} attribute."
            raise AttributeError(msg)

        if not callable(tensor_attr):
            msg = f"{self.__class__.__name__}.{__name} is not callable."
            raise TypeError(msg)

        def tensor_attrs_call(*args: Self | Any, **kwargs: Any) -> Self:
            """The TensorDataClass is the base one, thus we wanna get
            attribute (when we call `__getattr__`) at every single
            child's `Callable` attribute where it possible (if
            child's attribute has torch.Tensor instance).
            """

            def recursive_call(obj: Any) -> Any:
                if isinstance(obj, torch.Tensor | TensorDataClass):
                    return getattr(obj, __name)(*args, **kwargs)
                if isinstance(obj, dict):
                    return {key: recursive_call(value) for key, value in obj.items()}
                if isinstance(obj, tuple):
                    return tuple(recursive_call(value) for value in obj)
                return obj

            return self.__class__(**recursive_call(self.__dict__))

        return tensor_attrs_call

    def cuda(self, *args: Any, **kwargs: Any) -> Self:
        r"""
        Returns a copy of this object in CUDA memory.

        Args:
            *args (Any): Arguments required by :meth:`torch.Tensor.cuda`
            **kwargs (Any): Keyword arguments required by :meth:`torch.Tensor.cuda`

        Returns:
            typing.Union[TensorDataClass, torch.Tensor]: Copy of the object.
        """
        cuda_tensor: dict[str, torch.Tensor | TensorDataClass] = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                kwargs["non_blocking"] = kwargs.get("non_blocking", True)
                cuda_tensor[k] = v.cuda(*args, **kwargs)
            elif isinstance(v, TensorDataClass):
                cuda_tensor[k] = v.cuda(*args, **kwargs)
            else:
                cuda_tensor[k] = v
        return self.__class__(**cuda_tensor)

    def cpu(self) -> Self:
        r"""
        Returns a copy of this object in CPU memory.

        Returns:
            typing.Union[TensorDataClass, torch.Tensor]: Copy of the object.
        """
        cpu_tensor = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor | TensorDataClass):
                cpu_tensor[k] = v.cpu()
            else:
                cpu_tensor[k] = v
        return self.__class__(**cpu_tensor)
