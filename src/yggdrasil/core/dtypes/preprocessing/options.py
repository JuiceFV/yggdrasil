r"""
Defines the DPP options which will be set during configuration.
"""

from yggdrasil.core.dataclasses import dataclass
from yggdrasil.core.dtypes.base import Ftype
from yggdrasil.preprocessing.constants import (
    DEFAULT_MAX_QUANTILE_SIZE,
    DEFAULT_MAX_UNIQUE_ENUM,
    DEFAULT_NSAMPLES,
    DEFAULT_QUANTILE_K2_THRESHOLD,
)


@dataclass
class PreprocessingOptions:
    r"""
    Options are used to preprocess and transform data. For the details,
    follow :func:`~yggdrasil.preprocessing.normalization.identify_param`.

    .. warning::

        Currently ``set_missing_value_to_zero``, ``allowed_features``
        ``assert_allowlist_feature_coverage`` not in use. They will
        be implemented in the further versions.

        1. ``set_missing_value_to_zero`` - intends to replace missing
        values (``x_presence == False``) with 0. During sparse data
        processing.

        2. ``allowed_features`` - implements "check-box" for features.
        To consider only given features ModelManager must implement
        the logic of processing these features.

        3. ``assert_allowlist_feature_coverage`` - asserts if processing
        features are ``allowed_features``.

    Attributes:
        nsamples (int): Number of samples used to infer statistics from raw data.
        max_unique_enum_values (int): Maximum unique values of a categorical feature.
            Each value will be transformed to a one-hot vector.
        quantile_size (int): Number of quantiles that splits the raw data during transformation.
        quantile_k2_threshold (float): Skewness (:math:`s^2`) and Kurtosis (:math:`k^2`) threshold of
            `normal test <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html>`_.
        skip_boxcox (bool): Forcibly skip box cox transformation.
        skip_quantiles (bool): Forcibly skip quantile transformation.
        feature_overrides (dict[int, Ftype] | None): Optionally pre-defined feature types.
        table_sample (float | None): Part of dataset intended to be used in the training process.
        set_missing_value_to_zero (bool | None): Optionally replace missing values to zero in sparse data.
        allowed_features (list[int] | None): Optionally selected features which will be used during process.
        assert_allowlist_feature_coverage (bool): Optionally check if processing features are the same as
            ``allowed_features``.
    """

    nsamples: int = DEFAULT_NSAMPLES
    max_unique_enum_values: int = DEFAULT_MAX_UNIQUE_ENUM
    quantile_size: int = DEFAULT_MAX_QUANTILE_SIZE
    quantile_k2_threshold: float = DEFAULT_QUANTILE_K2_THRESHOLD
    skip_boxcox: bool = False
    skip_quantiles: bool = True
    feature_overrides: dict[int, Ftype] | None = None
    table_sample: float | None = None
    set_missing_value_to_zero: bool | None = False
    allowed_features: list[int] | None = None
    assert_allowlist_feature_coverage: bool = True
