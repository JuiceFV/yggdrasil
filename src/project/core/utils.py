import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any


class lazy_property:  # noqa: N801
    def __init__(self, retreiver: Callable[[Any], Any]) -> None:
        self._retreiver = retreiver
        self.__doc__ = retreiver.__doc__
        self.__name__ = retreiver.__name__

    def __get__(self, __o: Any | None, __t: type) -> Any | None:
        if __o is None:
            return None
        value = self._retreiver(__o)
        setattr(__o, self.__name__, value)
        return value


def deprecated_attrs(*attrs: str) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        original_getattribute = cls.__getattribute__

        @wraps(original_getattribute)
        def new_getattribute(self: object, name: str) -> Any:
            if name in attrs:
                warnings.warn(
                    f"{cls.__name__}.{name} is deprecated and will be removed in "
                    "future versions.",
                    DeprecationWarning,
                    stacklevel=2,
                )

            # TODO: figure out how to annotate decorator over builtins
            return original_getattribute(self, name)  # type: ignore

        cls.__getattribute__ = new_getattribute  # type: ignore
        return cls

    return decorator
