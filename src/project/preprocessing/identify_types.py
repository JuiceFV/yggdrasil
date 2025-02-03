import numpy as np
import numpy.typing as npt

from project.core.dtypes.base import Ftype
from project.preprocessing import DEFAULT_MAX_UNIQUE_ENUM


def _is_probability(fvalues: npt.ArrayLike) -> np.bool_:
    return np.all(np.array(0) <= fvalues) and np.all(fvalues <= np.array(1))


def _is_binary(fvalues: npt.ArrayLike) -> np.bool_:
    return np.all(
        np.logical_or(fvalues == np.array(0), fvalues == np.array(1))
    ) or np.min(fvalues) == np.max(fvalues)


def _is_continuous(fvalues: npt.ArrayLike) -> np.bool_:
    return np.bool_(True)


def _is_enum(fvalues: npt.ArrayLike, enum_threshold: int) -> np.bool_:
    are_all_ints = np.vectorize(lambda value: float(value).is_integer())
    return np.bool_(
        np.min(fvalues) >= np.array(0.0)
        and len(np.unique(fvalues)) <= enum_threshold
        and np.all(are_all_ints(fvalues))
    )


def identify_type(
    fvalues: npt.ArrayLike, enum_threshold: int = DEFAULT_MAX_UNIQUE_ENUM
) -> Ftype:
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
