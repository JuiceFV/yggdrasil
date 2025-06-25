import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

import numpy as np
from scipy import stats

from yggdrasil.core.dtypes.base import Ftype
from yggdrasil.core.dtypes.parameters import NormalizationParams
from yggdrasil.preprocessing import (
    BOXCOX_MARGIN,
    BOXCOX_MAX_STDEV,
    DEFAULT_MAX_QUANTILE_SIZE,
    DEFAULT_MAX_UNIQUE_ENUM,
    DEFAULT_QUANTILE_K2_THRESHOLD,
    MIN_SAMPLES_TO_IDENTIFY,
)
from yggdrasil.preprocessing.identify_types import identify_type

logger = logging.getLogger(__name__)


def identify_param(  # noqa: C901, PLR0912, PLR0915
    fid: int,
    fvalues: np.ndarray,
    max_unique_enum_values: int = DEFAULT_MAX_UNIQUE_ENUM,
    quantile_size: int = DEFAULT_MAX_QUANTILE_SIZE,
    quantile_k2_threshold: float = DEFAULT_QUANTILE_K2_THRESHOLD,
    skip_boxcox: bool = False,
    skip_quantiles: bool = False,
    ftype: Ftype | None = None,
) -> NormalizationParams:
    r"""
    Infers statistics (metadata) from a given sample of feature values.

    Recently, Ioffe & Szegedy demonstrated in their work that normalization mitigates
    issues arising from varying feature scales and distributions, improving model
    performance and convergence.

    Steps:
        1. Identify the feature type (e.g., ``float``, ``int``, ``enum``).
        2. Check if sampled feature values are normally distributed.
        3. If not, apply one or more of the following transformations:

            * BoxCox:

              .. math::
                  y = \mathbf{1}_{\lambda \neq 0}\left[\frac{x^{\lambda} - 1}{\lambda},
                  \log{x}\right]

            * Quantile Transformation (see `scipy.stats.mquantiles`)
            * Gauss Normalization:

              .. math::
                  y = \frac{x - \mu{(x)}}{\sigma{(x)}}

    Pre-computed statistics are applied during training and serving through the
    :class:`athena.preprocessing.preprocessor.Preprocessor`.

    Args:
        fid (int): Feature name.
        fvalues (np.ndarray): Sample of feature values.
        max_unique_enum_values (int, optional): Maximum unique values for a
            categorical feature. Defaults to ``DEFAULT_MAX_UNIQUE_ENUM``.
        quantile_size (int, optional): Number of quantile splits during
            transformation. Defaults to ``DEFAULT_MAX_QUANTILE_SIZE``.
        quantile_k2_threshold (float, optional): Threshold for skewness
            (:math:`s^2`) and kurtosis (:math:`k^2`) in the normality test
            (see `scipy.stats.normaltest`). Defaults to
            ``DEFAULT_QUANTILE_K2_THRESHOLD``.
        skip_boxcox (bool, optional): Skip the Box-Cox transformation.
            Defaults to ``False``.
        skip_quantiles (bool, optional): Skip the quantile transformation.
            Defaults to ``False``.
        ftype (Optional[Ftype], optional): Manually specified feature type.
            Defaults to ``None``.

    Raises:
        TypeError: If the feature type is undefined.
        RuntimeError: If the number of samples is insufficient to infer statistics.

    Returns:
        NormalizationParams: Accumulated metadata for normalization.
    """
    boxcox_required = ftype == Ftype.BOXCOX
    continuous_required = ftype == Ftype.CONTINUOUS
    quantile_required = ftype == Ftype.QUANTILE

    # If manual type is not given, identify it automatically
    if ftype is None:
        ftype = identify_type(fvalues, max_unique_enum_values)

    boxcox_lambda: float | None = None
    boxcox_shift: float | None = 0.0
    mean = 0.0
    stdev = 1.0
    possible_values: list[int] | None = None
    quantiles: list[float] | None = None

    if ftype not in Ftype:
        msg = f"Unknown type {ftype.value}"
        raise TypeError(msg)
    if len(fvalues) < MIN_SAMPLES_TO_IDENTIFY:
        msg = "Insufficient information to identify parameter."
        raise RuntimeError(msg)

    min_fvalue = float(np.min(fvalues))
    max_fvalue = float(np.max(fvalues))

    # If it's not required to normalize feature
    # compute its gauss statistics.
    if ftype == Ftype.DO_NOT_PREPROCESS:
        mean = float(np.mean(fvalues))
        fvalues = fvalues - mean
        # NOTE: Due to computations occures over sample
        # use degree of freedom equal to 1, means that
        # we calculate statistic (normalization factor N-1)
        # not parameter (normalization factor N)
        stdev = max(float(np.std(fvalues, ddof=1)), 1.0)

    # If feature is continuous make sure it's normalized
    if ftype == Ftype.CONTINUOUS or boxcox_required or quantile_required:
        # If fake vectors (padding for example) is given then no normalization applied
        if min_fvalue == max_fvalue and not (boxcox_required or quantile_required):
            return NormalizationParams(
                Ftype.CONTINUOUS, None, 0, 0, 1, None, None, None, None
            )

        # identify the difference between current feature distribution
        # and Normal distribution using normal tes. It returns k2 = s^2 + k^2
        # where s^2 - skewness (left/right) and k^2 - kurtosis (peak up/down).
        k2_original, p_original = stats.normaltest(fvalues)

        # apply box cox transformation and perform normal test to the transformed values
        boxcox_shift = float(min_fvalue * -1)
        boxcox_result = stats.boxcox(np.maximum(fvalues + boxcox_shift, BOXCOX_MARGIN))
        transformed_fvalues, lambda_ = boxcox_result[:2]
        if not (
            isinstance(transformed_fvalues, np.ndarray) and isinstance(lambda_, float)
        ):
            msg = (
                "BoxCox transformation failed. "
                "(Unexpected error, this only for the pylance)"
            )
            raise RuntimeError(msg)
        k2_boxcox, p_boxcox = stats.normaltest(transformed_fvalues)
        logger.info(
            f"Feature stats; Original K2: {k2_original} P: {p_original} "
            f"BoxCox K2: {k2_boxcox} P: {p_boxcox}"
        )

        # ===Box Cox Normalizaton===
        # In case transformation is tangible (lambda alter is higher than 0.1) and no
        # other restrictions defined mark that it's been successful and should be
        # applied during training and inference.
        lambda_lower_bound = 0.9
        lambda_upper_bound = 1.1
        if (
            (
                lambda_ < lambda_lower_bound
                or lambda_ > lambda_upper_bound
                or boxcox_required
            )
            and not (continuous_required or quantile_required)
            and (
                (k2_original > k2_boxcox * 10 and k2_boxcox <= quantile_k2_threshold)
                or boxcox_required
            )
        ):
            # We must be sure that box cox differs enough and it's much more closer
            # to the normal distribution than original values distribution. Besides,
            # it also possible that data variates too much, s.t. quantile normalization
            # is more suitable.
            stdev = float(np.std(transformed_fvalues, ddof=1))

            # For the few sample the data may be too noise or oposite too dense,
            # so make sure that it's distributed smoothly.
            if (
                np.isfinite(stdev)
                and stdev < BOXCOX_MAX_STDEV
                and not np.isclose(stdev, 0)
            ) or boxcox_required:
                fvalues = transformed_fvalues
                boxcox_lambda = float(lambda_)
        if boxcox_lambda is None or skip_boxcox:
            boxcox_shift = None
            boxcox_lambda = None
        if boxcox_lambda is not None:
            ftype = Ftype.BOXCOX

        # ===Quantile Normalization===
        # In case box cox hasn't been applied but original feature
        # distribution is still too far from gauss distribution apply
        # quantile normalization
        if (
            boxcox_lambda is None
            and k2_original > quantile_k2_threshold
            and not skip_quantiles
            and not continuous_required
        ) or quantile_required:
            ftype = Ftype.QUANTILE
            # Get quantiles emperically extracted from the feature values
            # (alphap, betap) are coefficient of beta distribution which
            # are used to interpolate cdf of given distribution. In this
            # case where alphap = 0, betap = 1 the interpolation is linear.
            quantiles = (
                np.unique(
                    stats.mstats.mquantiles(
                        fvalues,
                        np.arange(quantile_size + 1, dtype=np.float64)
                        / float(quantile_size),
                        alphap=0.0,
                        betap=1.0,
                    )
                )
                .astype(float)
                .tolist()
            )

    # ===Apply Gauss normalization===
    if ftype in (Ftype.CONTINUOUS, Ftype.BOXCOX):
        mean = float(np.mean(fvalues))
        fvalues = fvalues - mean
        stdev = max(float(np.std(fvalues, ddof=1)), 1.0)
        if not np.isfinite(stdev):
            msg = f"Standard deviation is infinite for feature {fid}"
            raise ValueError(msg)
        fvalues /= stdev

    # Infer all unique values of catigorical feature
    if ftype == Ftype.ENUM:
        possible_values = np.unique(fvalues.astype(int)).astype(int).tolist()

    return NormalizationParams(
        ftype=ftype,
        boxcox_lambda=boxcox_lambda,
        boxcox_shift=boxcox_shift,
        mean=mean,
        stdev=stdev,
        possible_values=possible_values,
        quantiles=quantiles,
        min_value=min_fvalue,
        max_value=max_fvalue,
    )


def sort_features_by_normalization(
    normalization_params: dict[int, NormalizationParams],
) -> tuple[list[int], list[int], list[int]]:
    sorted_features: list[int] = []
    fheaders: list[int] = []
    sorted_indices: list[int] = []
    if not isinstance(next(iter(normalization_params.keys())), int):
        msg = "Feature id must be integer type."
        raise TypeError(msg)
    sorted_fids = sorted(normalization_params.keys())
    for ftype in Ftype:
        fheaders.append(len(sorted_features))
        for idx, fid in enumerate(sorted_fids):
            if normalization_params[fid].ftype == ftype:
                sorted_indices.append(idx)
                sorted_features.append(fid)
    return sorted_features, fheaders, sorted_indices


def serialize(params: dict[int, NormalizationParams]) -> dict[int, str]:
    return {fid: json.dumps(asdict(fparams)) for fid, fparams in params.items()}


def deserialize(params_json: dict[int, str]) -> dict[int, NormalizationParams]:
    params: dict[int, NormalizationParams] = {}
    for fid, fparmas in params_json.items():
        norm_params = NormalizationParams(**json.loads(fparmas))
        if norm_params.ftype == Ftype.ENUM and norm_params.possible_values is None:
            msg = f"Expected values for {Ftype.ENUM} feature type"
            raise RuntimeError(msg)
        params[int(fid)] = norm_params
    return params


def get_normalization_data_dim(
    normalization_params: dict[int, NormalizationParams],
) -> int:
    return sum(
        len(np.possible_values)
        if np.ftype == Ftype.ENUM and np.possible_values is not None
        else 1
        for np in normalization_params.values()
    )


def get_feature_norm_metadata(
    fid: int, fvalue_list: list[float], norm_params: dict[str, Any]
) -> NormalizationParams:
    logger.info(f"Extracting normalization for feature: {fid}")
    nfeatures = len(fvalue_list)
    if nfeatures < MIN_SAMPLES_TO_IDENTIFY:
        msg = "Number of samples isn't enough to extract metadata."
        raise RuntimeError(msg)
    feature_override = None
    if norm_params["feature_overrides"] is not None:
        feature_override = norm_params["feature_overrides"].get(fid, None)
    feature_override = feature_override or norm_params.get("default_feature_override")

    fvalues = np.array(fvalue_list, dtype=np.float32)
    if np.any(np.isinf(fvalues)):
        msg = f"Feature {fid} contains infinity."
        raise ValueError(msg)
    if np.any(np.isnan(fvalues)):
        msg = f"Feature {fid} contains NaN."
        raise ValueError(msg)

    normalization_params = identify_param(
        fid,
        fvalues,
        norm_params["max_unique_enum_values"],
        norm_params["quantile_size"],
        norm_params["quantile_k2_threshold"],
        norm_params["skip_boxcox"],
        norm_params["skip_quantiles"],
        feature_override,
    )
    logger.info(f"Feature {fid} normalization {normalization_params}")
    return normalization_params


def infer_normalization(
    max_unique_enum_values: int,
    qunatile_size: int,
    quantile_k2_threshold: float,
    skip_box_cox: bool = False,
    skip_quantiles: bool = False,
    feature_overrides: dict[int, Ftype] | None = None,
    allowed_features: list[int] | None = None,
    assert_allowlist_feature_coverage: bool = True,
) -> Callable[[list], dict[int, NormalizationParams]]:
    norm_params = {
        "max_unique_enum_values": max_unique_enum_values,
        "quantile_size": qunatile_size,
        "quantile_k2_threshold": quantile_k2_threshold,
        "skip_boxcox": skip_box_cox,
        "skip_quantiles": skip_quantiles,
        "feature_overrides": feature_overrides,
    }

    allowed_features = allowed_features or []

    def assert_allowed_features(params: dict[int, NormalizationParams]) -> None:
        if not allowed_features:
            return
        allowed_features_set = {int(fid) for fid in allowed_features}
        available_features = set(params.keys())
        if allowed_features_set != available_features:
            msg = (
                f"Could not identify preprocessing type for the following features: "
                f"{allowed_features_set - available_features}; Extra features: "
                f"{available_features - allowed_features_set};"
            )
            raise RuntimeError(msg)

    def process(rows: list) -> dict[int, NormalizationParams]:
        params: dict[int, NormalizationParams] = {}
        for row in rows:
            if "fid" not in row or "fvalues" not in row:
                msg = f"Feature name or/and values are missing; {row}"
                raise AttributeError(msg)
            norm_metadata = get_feature_norm_metadata(
                row["fid"], row["fvalues"], norm_params
            )
            if (
                norm_metadata is not None
                and not allowed_features
                or row["fid"] in allowed_features
            ):
                params[row["fid"]] = norm_metadata

        if assert_allowlist_feature_coverage:
            assert_allowed_features(params)
        return params

    return process
