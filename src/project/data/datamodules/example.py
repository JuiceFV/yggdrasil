import torch

from project.core.dtypes.base import ExtraData
from project.core.dtypes.classification.base import BinaryPreprocessedInput
from project.core.dtypes.dataset import ParquetDataset, TableSpec
from project.core.dtypes.options import DatasetOptions, PreprocessingOptions
from project.core.dtypes.parameters import (
    NormalizationData,
    NormalizationKey,
)
from project.core.dtypes.preprocessing.base import InputColumn
from project.data.data_extractor.example import DataExtractor
from project.data.datamodules.manual import ManualDataModule
from project.data.etl import spark
from project.preprocessing.batch_preprocessor import BatchPreprocessor
from project.preprocessing.normalization import sort_features_by_normalization
from project.preprocessing.preprocessor import Preprocessor


class ExampleBatchPreprocessor(BatchPreprocessor):
    r"""
    Batch preprocessor for example data. This preprpcessor is an example
    and does nothing except some basic checks. It returns an instance of
    :class:`~project.core.dtypes.base.TensorDataClass` suitable for
    binary classification task.

    1. Unsqueeze target tensor to make it 2D.
    2. Create BinaryPreprocessedInput instance.
    """

    def __init__(self, features_preprocessor: Preprocessor) -> None:
        super().__init__()
        self.features_preprocessor = features_preprocessor

    def forward(self, batch: dict[str, torch.Tensor]) -> BinaryPreprocessedInput:
        r"""
        Applying preprocessing to a batch of tensor data.

        Args:
            batch (dict[str, torch.Tensor]): Batch of tensor data.

        Returns:
            BinaryPreprocessedInput: Preprocessed batch intantiated
            from binary specified :class:`~project.core.dtypes.base.TensorDataClass`
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
        return BinaryPreprocessedInput.from_input(**batch_dict)


class ExampleDataModule(ManualDataModule):
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

    def build_batch_preprocessor(self) -> ExampleBatchPreprocessor:
        if self.normalization_dict is None:
            msg = "Normalization dict must be defined"
            raise ValueError(msg)
        features_normalization_data = self.normalization_dict[NormalizationKey.FEATURES]
        features_norm_params = features_normalization_data.dense_normalization_params
        features_preprocessor = Preprocessor(normalization_params=features_norm_params)
        return ExampleBatchPreprocessor(features_preprocessor=features_preprocessor)
