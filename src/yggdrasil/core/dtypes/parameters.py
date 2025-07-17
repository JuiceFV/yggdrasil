from yggdrasil.core.base_dclass import BaseDataClass
from yggdrasil.core.config import param_hash
from yggdrasil.core.dataclasses import dataclass
from yggdrasil.core.dtypes.base import Ftype

SortedList = list[float]
EnumValues = list[int]


@dataclass(frozen=True)
class NormalizationParams(BaseDataClass):
    r"""
    Normalization parameters for a feature.

    Attributes:
        ftype (Ftype): Feature values type.
        boxcox_lambda (float | None): Lambda of the Box-Cox transformation if such transformation is applied.
        boxcox_shift (float | None): Shift of the Box-Cox transformation if such transformation is applied.
        mean (float | None): Mean of the feature values.
        stdev (float | None): Standard deviation of the feature values.
        possible_values (EnumValues | None): List of possible values for the feature if it is an enum type.
        quantiles (SortedList | None): Quantiles of the feature values if the quantile normalization is applied.
        min_value (float | None): Minimum value of the feature values.
        max_value (float | None): Maximum value of the feature values.
    """
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
    r"""
    Normalization data for a set of features.

    Attributes:
        dense_normalization_params (dict[int, NormalizationParams]): Normalization parameters for dense features
    """
    __hash__ = param_hash
    dense_normalization_params: dict[int, NormalizationParams]


class NormalizationKey:
    r"""
    Keys by which normalization data can be accessed.
    Usually, these keys replicate the names of the features
    (:class:`~yggdrasil.core.dtypes.preprocessing.base.InputColumn`)

    Attributes:
        FEATURES (str): Key for accessing dense features normalization parameters.
    """
    FEATURES = "features"
