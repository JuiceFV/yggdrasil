from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import StrEnum
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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        r"""
        Prodices TensorDataClass instance from python dictionary.

        .. note::

            This method accpets unstructured data, meaning any data type.
            However, the data should be processed s.t. it returns types
            instantiated from class:`torch.Tensor`.

        Args:
            data (dict[str, Any]): Dictionary of the unstructured data.

        Raises:
            NotImplementedError: Not overridden method.

        Returns:
            Self: TensorDataClass instance.
        """
        raise NotImplementedError


class Ftype(StrEnum):
    """
    Feature type which is detected while Data Analysis process.
    """

    #: Feature value can be either binary (0 or 1) or unique (``min == max``).
    BINARY = "binary"

    #: Feature value is a real number and a distribution adheres normal one.
    CONTINUOUS = "continuous"

    #: Feature value lies within range :math:`[0; 1]`.
    PROBABILITY = "probability"

    #: Feature value is a real number whose distribution differs enough from
    #: normal to apply the box-cox transformation.
    BOXCOX = "boxcox"

    #: Feature takes any discrete value which will be processed distinctly.
    ENUM = "enum"

    #: Feature value is a real number whose distribution differs enough from
    #: normal to apply the quantile normalization.
    QUANTILE = "quantile"

    #: Feature will not be processed. Commonly used for fake features.
    DO_NOT_PREPROCESS = "do_not_preprocess"


@dataclass
class Feature(TensorDataClass):
    r"""
    Feature wrapper which helps to handle different types of features.

    .. warning::

        Currently, only dense features are in use. Thus, there is no implementation
        of the sparse features.
    """

    #: Dense float features. (E.g. time spent)
    dense_features: torch.Tensor


@dataclass
class ExtraData(TensorDataClass):
    r"""
    Extra fields that are not directly included in the main data flow, but
    those fields could be helpful in process' formalization.
    """

    #: Hashed unique identifier of a table.
    sample_id: torch.Tensor | None = None

    @classmethod
    def from_dict(cls, data: dict[str, torch.Tensor]) -> Self:
        return cls(**{f.name: data.get(f.name, None) for f in fields(cls)})
