import torch

from yggdrasil.core.dtypes.base import ExtraData
from yggdrasil.core.dtypes.classification.base import BinaryPreprocessedInput
from yggdrasil.core.dtypes.dataset import ParquetDataset, TableSpec
from yggdrasil.core.dtypes.options import DatasetOptions, PreprocessingOptions
from yggdrasil.core.dtypes.parameters import (
    NormalizationData,
    NormalizationKey,
)
from yggdrasil.core.dtypes.preprocessing.base import InputColumn
from yggdrasil.data.data_extractor.example import DataExtractor
from yggdrasil.data.datamodules.manual import ManualDataModule
from yggdrasil.data.etl import spark
from yggdrasil.preprocessing.batch_preprocessor import BatchPreprocessor
from yggdrasil.preprocessing.normalization import sort_features_by_normalization
from yggdrasil.preprocessing.preprocessor import Preprocessor


class ExampleBatchPreprocessor(BatchPreprocessor):
    r"""
    Batch preprocessor for example data. This preprpcessor is an example
    and does nothing except some basic checks. It returns an instance of
    :class:`~yggdrasil.core.dtypes.base.TensorDataClass` suitable for
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
            from binary specified :class:`~yggdrasil.core.dtypes.base.TensorDataClass`
        """
        batch_dict = {}
        _, _, indcs = sort_features_by_normalization(self.features_preprocessor.normalization_params)
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
    r"""
    ExampleDataModule is a data module for example data. It inherits from
    :class:`~yggdrasil.data.datamodules.manual.ManualDataModule` and
    implements methods for feature identification and data querying.

    Args:
        input_table_spec (TableSpec | None): Specification of the input table.
        data_extractor (DataExtractor | None): Data extractor instance.
        setup_data (dict[str, str] | None): Setup data for the data module.
        saved_setup_data (dict[str, str] | None): Saved setup data for the data module.
        dataset_options (DatasetOptions | None): Options for the dataset.
        features_preprocessing_options (PreprocessingOptions | None): Options for features preprocessing.
    """
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

        self.features_preprocessing_options = features_preprocessing_options or PreprocessingOptions()

    def run_feature_identification(self, table_identifier: str) -> dict[str, NormalizationData]:
        r"""
        Run feature identification for the given table identifier.

        Args:
            table_identifier (str): Identifier of the table to run feature identification on.

        Returns:
            dict[str, NormalizationData]: A dictionary containing normalization data for features.
        """
        session = spark.init.get_spark_session()
        features_normalization_params = spark.transform.identify_normalization_params(
            session=session,
            table_name=table_identifier,
            col_name=InputColumn.FEATURES,
            preprocessing_options=self.features_preprocessing_options,
            seed=42,
        )
        return {NormalizationKey.FEATURES: NormalizationData(features_normalization_params)}

    def query_data(
        self,
        table_identifier: str,
        sample_range: tuple[float, float],
        data_extractor: DataExtractor,
    ) -> ParquetDataset:
        """
        Query data from the data extractor for the given table identifier and sample range.

        Args:
            table_identifier (str): Identifier of the table to query data from.
            sample_range (tuple[float, float]): Range of samples to query.
            data_extractor (DataExtractor): Instance of DataExtractor to use for querying.

        Returns:
            ParquetDataset: A dataset containing the queried data.
        """
        return data_extractor.query_data(table_identifier, sample_range)

    def build_batch_preprocessor(self) -> ExampleBatchPreprocessor:
        r"""
        Build a batch preprocessor for the example data module.

        Raises:
            ValueError: If the normalization dictionary is not defined.

        Returns:
            ExampleBatchPreprocessor: An instance of ExampleBatchPreprocessor.
        """
        if self.normalization_dict is None:
            msg = "Normalization dict must be defined"
            raise ValueError(msg)
        features_normalization_data = self.normalization_dict[NormalizationKey.FEATURES]
        features_norm_params = features_normalization_data.dense_normalization_params
        features_preprocessor = Preprocessor(normalization_params=features_norm_params)
        return ExampleBatchPreprocessor(features_preprocessor=features_preprocessor)
