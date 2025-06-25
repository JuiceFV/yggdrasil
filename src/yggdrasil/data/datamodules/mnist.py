import torch

from yggdrasil.core.dtypes.base import ExtraData
from yggdrasil.core.dtypes.classification.base import MulticlassPreprocessedInput
from yggdrasil.core.dtypes.dataset import ParquetDataset, TableSpec
from yggdrasil.core.dtypes.options import DatasetOptions, PreprocessingOptions
from yggdrasil.core.dtypes.parameters import (
    NormalizationData,
    NormalizationKey,
)
from yggdrasil.core.dtypes.preprocessing.base import InputColumn
from yggdrasil.data.data_extractor.mnist import DataExtractor
from yggdrasil.data.datamodules.manual import ManualDataModule
from yggdrasil.data.etl import spark
from yggdrasil.preprocessing.batch_preprocessor import BatchPreprocessor
from yggdrasil.preprocessing.normalization import sort_features_by_normalization
from yggdrasil.preprocessing.preprocessor import Preprocessor


class MNISTBatchPreprocessor(BatchPreprocessor):
    def __init__(self, features_preprocessor: Preprocessor) -> None:
        super().__init__()
        self.features_preprocessor = features_preprocessor

    def forward(self, batch: dict[str, torch.Tensor]) -> MulticlassPreprocessedInput:
        r"""
        Applying preprocessing to a batch of tensor data.

        Args:
            batch (dict[str, torch.Tensor]): Batch of tensor data.

        Returns:
            MulticlassPreprocessedInput: Preprocessed batch intantiated from multiclass
                specified :class:`~yggdrasil.core.dtypes.base.TensorDataClass`
        """
        batch_dict = {}
        _, _, indcs = sort_features_by_normalization(
            self.features_preprocessor.normalization_params
        )
        batch_dict["features"] = self.features_preprocessor(
            torch.index_select(
                batch[InputColumn.FEATURES],
                dim=1,
                index=torch.tensor(indcs),
            ),
            torch.index_select(
                batch[f"{InputColumn.FEATURES}_presence"],
                dim=1,
                index=torch.tensor(indcs),
            ),
        )
        batch_dict["extras"] = ExtraData(sample_id=batch[InputColumn.SAMPLE_ID].long())
        batch_dict["target"] = batch[InputColumn.TARGET].unsqueeze(-1).float()
        batch_dict["nclasses"] = 10  # MNIST has 10 classes (0-9)
        return MulticlassPreprocessedInput.from_input(**batch_dict)


class MNISTDataModule(ManualDataModule):
    def __init__(
        self,
        *,
        input_table_spec: TableSpec | None = None,
        data_extractor: DataExtractor | None = None,
        setup_data: dict[str, str] | None = None,
        saved_setup_data: dict[str, str] | None = None,
        dataset_options: DatasetOptions | None = None,
        features_preprocessing_options: PreprocessingOptions | None = None,
    ) -> None:
        super().__init__(
            input_table_spec=input_table_spec,
            data_extractor=data_extractor,
            setup_data=setup_data,
            saved_setup_data=saved_setup_data,
            dataset_options=dataset_options,
        )

        self.features_preprocessing_options = (
            features_preprocessing_options or PreprocessingOptions()
        )

    def run_feature_identification(
        self, table_identifier: str
    ) -> dict[str, NormalizationData]:
        session = spark.init.get_spark_session()
        features_normalization_params = spark.transform.identify_normalization_params(
            session=session,
            table_name=table_identifier,
            col_name=InputColumn.FEATURES,
            preprocessing_options=self.features_preprocessing_options,
            seed=42,  # Fixed seed for reproducibility
        )
        return {
            NormalizationKey.FEATURES: NormalizationData(features_normalization_params)
        }

    def query_data(
        self,
        table_identifier: str,
        sample_range: tuple[float, float],
        data_extractor: DataExtractor,
    ) -> ParquetDataset:
        return data_extractor.query_data(table_identifier, sample_range)

    def build_batch_preprocessor(self) -> MNISTBatchPreprocessor:
        if self.normalization_dict is None:
            msg = "Normalization dict must be defined"
            raise ValueError(msg)
        features_normalization_data = self.normalization_dict[NormalizationKey.FEATURES]
        features_norm_params = features_normalization_data.dense_normalization_params
        features_preprocessor = Preprocessor(normalization_params=features_norm_params)
        return MNISTBatchPreprocessor(features_preprocessor=features_preprocessor)
