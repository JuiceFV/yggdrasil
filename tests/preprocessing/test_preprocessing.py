import numpy as np
import numpy.testing as npt
import torch
from scipy import special

from project.core.dtypes.base import Ftype
from project.core.dtypes.parameters import NormalizationParams
from project.preprocessing import MISSING_VALUE, normalization
from project.preprocessing.normalization import (
    identify_param,
    sort_features_by_normalization,
)
from project.preprocessing.preprocessor import Preprocessor
from tests.preprocessing.utils import (
    BOXCOX_FEATURE_ID,
    CONTINUOUS_FEATURE_ID,
    ENUM_FEATURE_ID,
    MISSING_VALUE_MARGIN,
    PROBABILITY_FEATURE_ID,
    NumpyFeaturePreprocessor,
    fid2type,
    variate_data,
)


class TestPreprocessing:
    def test_prepare_normalization_and_normalize(self) -> None:  # noqa: C901, PLR0912, PLR0915
        fvalue_map = variate_data()
        normalization_params: dict[int, NormalizationParams] = {}
        for fid, fvalues in fvalue_map.items():
            normalization_params[fid] = identify_param(fid, fvalues, 10)
        for fid, norm_params in normalization_params.items():
            if fid2type(fid) == Ftype.CONTINUOUS:
                assert norm_params.ftype == Ftype.CONTINUOUS
                assert norm_params.boxcox_lambda is None
                assert norm_params.boxcox_shift is None
            elif fid2type(fid) == Ftype.BOXCOX:
                assert norm_params.ftype == Ftype.BOXCOX
                assert norm_params.boxcox_lambda is not None
                assert norm_params.boxcox_shift is not None
            else:
                assert norm_params.ftype == fid2type(fid)

        preprocessor = Preprocessor(normalization_params)
        sorted_features, _, indcs = sort_features_by_normalization(normalization_params)
        input_matrix = torch.zeros([10000, len(sorted_features)])
        # sorted(normalization_params.keys()) replicates original order in tmp dataset
        for i, fid in enumerate(sorted(normalization_params.keys())):
            input_matrix[:, i] = torch.from_numpy(fvalue_map[fid])
        input_matrix = input_matrix[:, indcs]
        normalized_feature_matrix = preprocessor(
            input_matrix, (input_matrix != MISSING_VALUE)
        )
        normalized_features: dict[int, torch.Tensor] = {}
        on_column = 0
        for fid in sorted_features:
            norm = normalization_params[fid]
            if norm.ftype == Ftype.ENUM:
                assert norm.possible_values is not None
                column_size = len(norm.possible_values)
            else:
                column_size = 1
            normalized_features[fid] = normalized_feature_matrix[
                :, on_column : (on_column + column_size)
            ]
            on_column += column_size

        assert all(
            np.isfinite(param.stdev) and np.isfinite(param.mean)
            for param in normalization_params.values()
            if param.stdev is not None and param.mean is not None
        )
        for fid, nfvalues in normalized_features.items():
            np_nfvalues = nfvalues.numpy()
            assert np.all(np.isfinite(np_nfvalues))
            ftype = normalization_params[fid].ftype
            if ftype == Ftype.PROBABILITY:
                sigmoidv = special.expit(np_nfvalues)
                assert np.all(
                    np.logical_and(np.greater(sigmoidv, 0), np.less(sigmoidv, 1))
                )
            elif ftype == Ftype.ENUM:
                possible_values = normalization_params[fid].possible_values
                assert possible_values is not None
                assert np_nfvalues.shape[0] == len(fvalue_map[fid])
                assert np_nfvalues.shape[1] == len(possible_values)
                possible_value_map = {}
                for i, possible_value in enumerate(possible_values):
                    possible_value_map[possible_value] = i
                for i, row in enumerate(np_nfvalues):
                    original_feature = fvalue_map[fid][i]
                    if abs(original_feature - MISSING_VALUE) < MISSING_VALUE_MARGIN:
                        assert np.sum(row) == 0.0
                    else:
                        assert (
                            possible_value_map[original_feature]
                            == np.where(row == 1)[0][0]
                        )
            elif ftype == Ftype.QUANTILE:
                for i, feature in enumerate(np_nfvalues[0]):
                    original_feature = fvalue_map[fid][i]
                    expected = NumpyFeaturePreprocessor.value_to_quantile(
                        original_feature, np.array(normalization_params[fid].quantiles)
                    )
                    assert np.isclose(feature, expected, atol=1e-2)
            elif ftype in (Ftype.CONTINUOUS, Ftype.BOXCOX):
                one_stdev = np.isclose(np.std(np_nfvalues, ddof=1), 1, atol=0.05)
                zero_stdev = np.isclose(np.std(np_nfvalues, ddof=1), 0, atol=0.05)
                zero_mean = np.isclose(np.mean(np_nfvalues), 0, atol=0.01)
                assert np.all(zero_mean)
                assert np.logical_or(one_stdev, zero_stdev)
            elif ftype == Ftype.BINARY:
                pass
            else:
                raise NotImplementedError

    def test_normalize_dense_matrix_enum(self) -> None:
        normalization_params = {
            1: NormalizationParams(
                Ftype.ENUM, None, None, None, None, [12, 4, 2], None, None, None
            ),
            2: NormalizationParams(
                Ftype.CONTINUOUS, None, 0, 0, 1, None, None, None, None
            ),
            3: NormalizationParams(
                Ftype.ENUM, None, None, None, None, [15, 3], None, None, None
            ),
        }

        preprocessor = Preprocessor(normalization_params)

        inputs = np.zeros([4, 3], dtype=np.float32)
        fids = [2, 1, 3]
        inputs[:, fids.index(1)] = [12, 4, 2, 2]
        inputs[:, fids.index(2)] = [1.0, 2.0, 3.0, 3.0]
        inputs[:, fids.index(3)] = [15, 3, 15, MISSING_VALUE]
        inputs = torch.from_numpy(inputs)
        normalized_feature_matrix = preprocessor(inputs, (inputs != MISSING_VALUE))

        npt.assert_allclose(
            np.array(
                [
                    [1.0, 1, 0, 0, 1, 0],
                    [2.0, 0, 1, 0, 0, 1],
                    [3.0, 0, 0, 1, 1, 0],
                    [3.0, 0, 0, 1, 0, 0],
                ]
            ),
            normalized_feature_matrix,
        )

    def test_persistency(self) -> None:
        fvalue_map = variate_data()
        normalization_params: dict[int, NormalizationParams] = {}
        for fid, fvalues in fvalue_map.items():
            normalization_params[fid] = identify_param(fid, fvalues)
            fvalues[0] = MISSING_VALUE

        serialized_normalization_params = normalization.serialize(normalization_params)
        deserialized_normalization_params = normalization.deserialize(
            serialized_normalization_params
        )
        assert deserialized_normalization_params.keys() == normalization_params.keys()
        for fid in normalization_params:
            for field in [
                "ftype",
                "possible_values",
                "boxcox_lambda",
                "boxcox_shift",
                "mean",
                "stdev",
                "quantiles",
                "min_value",
                "max_value",
            ]:
                assert getattr(
                    deserialized_normalization_params[fid], field
                ) == getattr(normalization_params[fid], field)

    def test_quantile_boundary(self) -> None:
        x = torch.tensor([[0.0], [80.0], [100.0]])
        norm_params = NormalizationParams(
            Ftype.QUANTILE, None, None, 0, 1, None, [0.0, 80.0, 100.0], 0.0, 100.0
        )
        preprocessor = Preprocessor({1: norm_params})
        preprocessed_input = preprocessor._preprocess_quantile(
            0, x.float(), [norm_params]
        )

        expected = torch.tensor([[0.0], [0.5], [1.0]])

        assert np.allclose(preprocessed_input, expected)

    def test_preprocessing_network(self) -> None:
        fvalue_map = variate_data()

        normalization_params: dict[int, NormalizationParams] = {}
        fid_preprocessed_blob_map: dict[int, np.ndarray] = {}

        for fid, fvalues in fvalue_map.items():
            normalization_params[fid] = identify_param(fid, fvalues)
            fvalues[0] = MISSING_VALUE

            preprocessor = Preprocessor({fid: normalization_params[fid]})
            fvalue_matrix = torch.from_numpy(np.expand_dims(fvalues, -1))
            normalized_fvalues: torch.Tensor = preprocessor(
                fvalue_matrix, (fvalue_matrix != MISSING_VALUE)
            )
            fid_preprocessed_blob_map[fid] = normalized_fvalues.numpy()

        test_features = NumpyFeaturePreprocessor.preprocess(
            fvalue_map, normalization_params
        )

        for fid in fvalue_map:
            normalized_features = fid_preprocessed_blob_map[fid]
            if fid != ENUM_FEATURE_ID:
                normalized_features = np.squeeze(normalized_features, -1)

            tol = 0.01
            if fid == BOXCOX_FEATURE_ID:
                tol = 0.5

            assert np.allclose(
                normalized_features.flatten(),
                test_features[fid].flatten(),
                rtol=tol,
                atol=tol,
            ), f"{fid} doesn't match."

    def test_type_override_binary(self) -> None:
        fvalue_map = variate_data()
        probability_values = fvalue_map[PROBABILITY_FEATURE_ID]

        param = identify_param(-101, probability_values, ftype=Ftype.BINARY)
        assert param.ftype.value == "binary"

    def test_type_override_continuous(self) -> None:
        fvalue_map = variate_data()
        boxcocks_values = fvalue_map[BOXCOX_FEATURE_ID]

        param = identify_param(-101, boxcocks_values, ftype=Ftype.CONTINUOUS)
        assert param.ftype.value == "continuous"

    def test_type_override_boxcox(self) -> None:
        fvalue_map = variate_data()
        continuous_values = fvalue_map[CONTINUOUS_FEATURE_ID]

        param = identify_param(-101, continuous_values, ftype=Ftype.BOXCOX)
        assert param.ftype.value == "boxcox"

    def test_type_override_quantile(self) -> None:
        fvalue_map = variate_data()
        boxcocks_values = fvalue_map[BOXCOX_FEATURE_ID]

        param = identify_param(-101, boxcocks_values, ftype=Ftype.QUANTILE)
        assert param.ftype.value == "quantile"
