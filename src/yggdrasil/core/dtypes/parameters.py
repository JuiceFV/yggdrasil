from yggdrasil.core.base_dclass import BaseDataClass
from yggdrasil.core.config import param_hash
from yggdrasil.core.dataclasses import dataclass
from yggdrasil.core.dtypes.base import Ftype

SortedList = list[float]
EnumValues = list[int]


@dataclass(frozen=True)
class NormalizationParams(BaseDataClass):
    __hash__ = param_hash

    ftype: Ftype
    boxcox_lambda: float | None = None
    boxcox_shift: float | None = None
    mean: float | None = None
    stdev: float | None = None
    possible_values: EnumValues | None = None
    quantiles: SortedList | None = None
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class NormalizationData(BaseDataClass):
    __hash__ = param_hash
    dense_normalization_params: dict[int, NormalizationParams]


class NormalizationKey:
    FEATURES = "features"
