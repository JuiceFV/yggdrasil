import numpy as np
from scipy import special, stats

from project.core.dtypes.base import Ftype
from project.core.dtypes.parameters import NormalizationParams
from project.preprocessing import BOXCOX_MARGIN, MAX_FVALUE, MIN_FVALUE, MISSING_VALUE

BINARY_FEATURE_ID = 1
BINARY_FEATURE_ID_2 = 2
BOXCOX_FEATURE_ID = 3
CONTINUOUS_FEATURE_ID = 4
CONTINUOUS_FEATURE_ID_2 = 5
ENUM_FEATURE_ID = 6
PROBABILITY_FEATURE_ID = 7
QUANTILE_FEATURE_ID = 8

ARRAY_SHAPE = 2
MISSING_VALUE_MARGIN = 1e-2

NDFeatureT = np.ndarray[np.int16, np.dtype[np.float32]]


def fid2type(fid: int) -> str:
    if fid in (BINARY_FEATURE_ID, BINARY_FEATURE_ID_2):
        return "binary"
    if fid == BOXCOX_FEATURE_ID:
        return "boxcox"
    if fid in (CONTINUOUS_FEATURE_ID, CONTINUOUS_FEATURE_ID_2):
        return "continuous"
    if fid == ENUM_FEATURE_ID:
        return "enum"
    if fid == PROBABILITY_FEATURE_ID:
        return "probability"
    if fid == QUANTILE_FEATURE_ID:
        return "quantile"
    msg = f"Invalid feature id: {fid}"
    raise AssertionError(msg)


def variate_data() -> dict[int, NDFeatureT]:
    rng = np.random.default_rng(1)
    feature_value_map = {}
    feature_value_map[BINARY_FEATURE_ID] = rng.binomial(1, 0.5, size=10000).astype(
        np.float32
    )
    feature_value_map[BINARY_FEATURE_ID_2] = rng.binomial(1, 0.5, size=10000).astype(
        np.float32
    )
    feature_value_map[CONTINUOUS_FEATURE_ID] = rng.normal(size=10000).astype(np.float32)
    feature_value_map[CONTINUOUS_FEATURE_ID_2] = rng.normal(size=10000).astype(
        np.float32
    )
    feature_value_map[BOXCOX_FEATURE_ID] = rng.exponential(size=10000).astype(
        np.float32
    )
    feature_value_map[ENUM_FEATURE_ID] = (
        rng.integers(0, 10, size=10000) * 1000
    ).astype(np.float32)
    feature_value_map[QUANTILE_FEATURE_ID] = np.concatenate(
        (rng.normal(size=5000), rng.exponential(size=5000))
    ).astype(np.float32)
    feature_value_map[PROBABILITY_FEATURE_ID] = np.clip(
        rng.beta(a=2.0, b=2.0, size=10000), 0.01, 0.99
    )
    return feature_value_map


class NumpyFeaturePreprocessor:
    @staticmethod
    def value_to_quantile(
        original_value: np.float32, quantiles: np.ndarray
    ) -> np.float32:
        if original_value <= quantiles[0]:
            return np.float32(0.0)
        if original_value >= quantiles[-1]:
            return np.float32(1.0)
        nquantiles = float(len(quantiles) - 1)
        right = np.searchsorted(quantiles, original_value)
        left = right - 1
        interpolated = (
            left
            + (
                (original_value - quantiles[left])
                / (quantiles[right] + 1e-6 - quantiles[left])
            )
        ) / nquantiles
        return interpolated

    @classmethod
    def preprocess_feature(
        cls, feature: NDFeatureT, params: NormalizationParams
    ) -> NDFeatureT:
        is_not_missing = 1 - np.isclose(feature, MISSING_VALUE)
        if params.ftype == Ftype.BINARY:
            return ((feature != 0) * is_not_missing).astype(np.float32)
        if params.ftype == Ftype.BOXCOX:
            assert params.boxcox_lambda is not None
            assert params.boxcox_shift is not None
            x = np.maximum(feature + params.boxcox_shift, BOXCOX_MARGIN)
            # NOTE: In the following box-cox configuration (x and lmbda)
            # it has to return a 1D array and it does, but pylance highlights
            # it as incompatible with the expected type.
            feature = stats.boxcox(x, params.boxcox_lambda)  # type: ignore
        elif params.ftype == Ftype.PROBABILITY:
            feature = np.clip(feature, 0.01, 0.99)
            feature = special.logit(feature)
        elif params.ftype == Ftype.QUANTILE:
            assert params.quantiles is not None
            transformed_feature = np.zeros_like(feature)
            for i in range(feature.shape[0]):
                transformed_feature[i] = cls.value_to_quantile(
                    feature[i], np.array(params.quantiles)
                )
            feature = transformed_feature
        elif params.ftype == Ftype.ENUM:
            assert params.possible_values is not None
            possible_values = params.possible_values
            value_feature_mapping: dict[int, int] = {}
            for i, possible_value in enumerate(possible_values):
                value_feature_mapping[possible_value] = i
            output_feature = np.zeros(
                (len(feature), len(possible_values)), dtype=np.float32
            )
            for i, value in enumerate(feature):
                if abs(value - MISSING_VALUE) < MISSING_VALUE_MARGIN:
                    continue
                output_feature[i][value_feature_mapping[value]] = np.float32(1.0)
            return output_feature
        else:
            assert params.mean is not None
            assert params.stdev is not None
            feature = feature - params.mean
            feature /= params.stdev
            feature = np.clip(feature, MIN_FVALUE, MAX_FVALUE)
        feature *= is_not_missing
        return feature

    @classmethod
    def preprocess(
        cls, features: dict[int, NDFeatureT], params: dict[int, NormalizationParams]
    ) -> dict[int, NDFeatureT]:
        res: dict[int, NDFeatureT] = {}
        for fid in features:
            res[fid] = cls.preprocess_feature(features[fid], params[fid])
        return res

    @classmethod
    def preprocess_array(
        cls,
        arr: np.ndarray,
        features: dict[int, np.ndarray[np.int32, np.dtype[np.float32]]],
        params: dict[int, NormalizationParams],
    ) -> np.ndarray:
        assert len(arr.shape) == ARRAY_SHAPE
        assert arr.shape[1] == len(features)
        preprocessed_values = [
            cls.preprocess(dict(zip(features, row, strict=False)), params)
            for row in arr
        ]
        return np.array(
            [[pv[fid] for fid in features] for pv in preprocessed_values],
            dtype=np.float32,
        )
