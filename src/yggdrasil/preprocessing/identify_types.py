import numpy as np
import numpy.typing as npt

from yggdrasil.core.dtypes.base import Ftype
from yggdrasil.preprocessing.constants import DEFAULT_MAX_UNIQUE_ENUM


def _is_probability(fvalues: npt.ArrayLike) -> np.bool_:
    return np.all(np.array(0) <= fvalues) and np.all(fvalues <= np.array(1))


def _is_binary(fvalues: npt.ArrayLike) -> np.bool_:
    return np.all(np.logical_or(fvalues == np.array(0), fvalues == np.array(1))) or np.min(fvalues) == np.max(fvalues)


def _is_continuous(fvalues: npt.ArrayLike) -> np.bool_:
    return np.bool_(True)


def _is_enum(fvalues: npt.ArrayLike, enum_threshold: int) -> np.bool_:
    are_all_ints = np.vectorize(lambda value: float(value).is_integer())
    return np.bool_(
        np.min(fvalues) >= np.array(0.0) and len(np.unique(fvalues)) <= enum_threshold and np.all(are_all_ints(fvalues))
    )


def identify_type(fvalues: npt.ArrayLike, enum_threshold: int = DEFAULT_MAX_UNIQUE_ENUM) -> Ftype:
    r"""
    Identify the type of a feature based on its values.

    There are four types of features that can be identified:

    - Binary: Feature values are either 0 or 1, or all values are the same (``min == max``).
    - Probability: Feature values are in the range [0, 1].
    - Enum: Feature values are discrete and the number of unique values is less than or equal to ``enum_threshold``.
    - Continuous: Feature values are real numbers and their distribution adheres to a normal distribution.

    Args:
        fvalues (npt.ArrayLike): Array-like structure containing feature values.
        enum_threshold (int, optional): Threshold for identifying enum types.
            Defaults to DEFAULT_MAX_UNIQUE_ENUM.

    Raises:
        TypeError: If the feature type cannot be identified.

    Returns:
        Ftype: Identified feature type as an instance of ``Ftype`` enum.
    """
    if _is_binary(fvalues):
        return Ftype.BINARY
    if _is_probability(fvalues):
        return Ftype.PROBABILITY
    if _is_enum(fvalues, enum_threshold):
        return Ftype.ENUM
    if _is_continuous(fvalues):
        return Ftype.CONTINUOUS
    msg = "Unidentified feature type."
    raise TypeError(msg)
