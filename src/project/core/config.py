from dataclasses import fields, is_dataclass
from typing import Any


def param_hash(self: object) -> int:
    """
    Use this to make parameters hashable. This is required because :func:`__hash__`
    is not inherited when subclass redefines :func:`__eq__`. We only need this when
    the parameter dataclass has a list or dict field.

    Args:
        self (object): Dataclass object.

    Raises:
        TypeError: If got non-dataclass object.

    Returns:
        int: Hash of given dataclass ``self``.
    """
    if not is_dataclass(self):
        msg = f"Expected dataclass, got {type(self)}"
        raise TypeError(msg)
    return hash(tuple(_hash_field(getattr(self, f.name)) for f in fields(self)))


def _hash_field(val: Any) -> Any:
    """
    Returns hashable value of the argument. A list is converted to a tuple,
    and each element is processed recursively. A dict is converted to a
    tuple of sorted pairs of key and value, with values processed recursively.
    """
    if isinstance(val, list):
        return tuple(_hash_field(v) for v in val)
    if isinstance(val, dict):
        return tuple(sorted((k, _hash_field(v)) for k, v in val.items()))
    return val
