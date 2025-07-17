import json
import os
from typing import cast
from urllib.parse import urlparse

from datasets import Dataset as HFDataset
from datasets import DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from yggdrasil.core.dataclasses import DataclassJSONEncoder
from yggdrasil.core.dtypes.base import TensorDataClass
from yggdrasil.core.dtypes.dataset import ParquetDataset, TableSpec
from yggdrasil.core.dtypes.options import DatasetOptions
from yggdrasil.core.dtypes.parameters import NormalizationData
from yggdrasil.data.data_extractor.base import DataExtractor
from yggdrasil.data.datamodules.base import (
    BaseDataModule,
    collate_and_preprocess,
    get_sample_range,
)


def _deserialize_setup_data(
    setup_data: dict[str, str] | str | None,
) -> dict[str, str] | None:
    if isinstance(setup_data, str):
        with open(setup_data) as f:
            return json.load(f)
    return setup_data


def _scan_parquet_folder(file_path: str) -> list[str]:
    files = [os.path.join(file_path, f) for f in os.listdir(urlparse(file_path).path) if f.endswith(".parquet")]
    return files


def load_parquet_dataset(
    url: str,
    streaming: bool = False,
    num_proc: int = 1,
) -> DatasetDict | HFDataset | IterableDatasetDict | IterableDataset:
    ds = load_dataset(
        "parquet",
        data_files=_scan_parquet_folder(url),
        split="train",
        streaming=streaming,
        num_proc=None if streaming else num_proc,
    )
    return ds


class ManualDataModule(BaseDataModule):
    """
    Manual data module automates the loading and transformation process, relying on the :ref:`data fromatting`
    and snapshot mechanism defined across the entire Yggdrasil framework. The original interface dictated
    by :class:`~yggdrasil.data.datamodules.base.BaseDataModule` leaves all the methods responsible for data loading,
    transforming, and serving as not implemented.

    .. code-block:: python

        class BaseDataModule(abc.ABC, L.LightningDataModule):
            @abc.abstractmethod
            def prepare_data(self) -> None:
                pass

            @abc.abstractmethod
            def setup(self, stage: str | None) -> None:
                pass

            @abc.abstractmethod
            def train_dataloader(self) -> DataLoader[TensorDataClass]:
                pass

            @abc.abstractmethod
            def val_dataloader(self) -> DataLoader[TensorDataClass] | None:
                pass

            @abc.abstractmethod
            def test_dataloader(self) -> DataLoader[TensorDataClass] | None:
                pass

    The manual datamodule siezes the data format and the ETL process to automate these procedures.
    The implementation is designed to work with a specific input table specification and a data extractor.
    The workfkow is as follows :meth:`~lightning.pytorch.core.hooks.DataHooks.prepare_data`:

    1. :meth:`prepare_data` is called to prepare the data. This method is called on the driver node and responsible
       for the loading (extracting) the data from the source. During the data preparation 3 key processes occur:

       * Data loading: Loads the data from a dataset prepared on the :ref:`ddp` stage.
         The extraction is defined by :meth:`~yggdrasil.data.data_extractor.base.DataExtractor.query_data`.

       * Splitting: Splits the data into train, eval, and test datasets based on the sample ranges.
         See the :func:`~yggdrasil.data.etl.spark.extract.hash_and_subsample` for the splitting logic.

       * Feature identification: Identifies the normalization parameters for the dense columns.
         See the :func:`~yggdrasil.data.etl.spark.transform.identify_normalization_params`
         for the feature identification logic.

    2. :meth:`setup` is called to set up the data module. This method is called on each worker node and responsible
       for the storing the metadata and the datasets in the datamodule instance.

    3. :meth:`get_dataloader` is called to get the dataloader for the train, eval, and test datasets. It loads the
       data iteratively from the parquet files and applies the preprocessing defined by the
       :meth:`~yggdrasil.data.datamodules.BaseDataModule.build_batch_preprocessor`.

    .. code-block:: python

        class ManualDataModule(BaseDataModule):
            def prepare_data(self) -> None:
                ...

            def setup(self, stage: str | None) -> None:
                ...

            def train_dataloader(self) -> DataLoader[TensorDataClass]:
                ...

            def val_dataloader(self) -> DataLoader[TensorDataClass] | None:
                ...

            def test_dataloader(self) -> DataLoader[TensorDataClass] | None:
                ...

    As you can see, the manual data module implements the most rutine of the data loading and transformation
    procedures, while leaving the flexibility to define the data extraction and transformation logic. A user
    needs only the following methods to implement the manual data module:

    1. :meth:`~yggdrasil.data.datamodules.BaseDataModule.run_feature_identification`: This method is responsible
       for identifying the normalization parameters for the dense columns in the dataset. It should return a
       dictionary with the normalization parameters for the dense columns.
    2. :meth:`~yggdrasil.data.datamodules.BaseDataModule.query_data`: This method is responsible for querying the
       data from the data extractor. It should return a :class:`~yggdrasil.core.dtypes.dataset.ParquetDataset`
       instance containing the data.
    3. :meth:`~yggdrasil.data.datamodules.BaseDataModule.build_batch_preprocessor`: This method is responsible
       for building the batch preprocessor for a dataset.

    .. note::
        Usually, the implementations don't vary much from the one module to another.
        They commonly, specify the data structure types, columns to process, extraction logic
        (which is defined by the :class:`~yggdrasil.data.data_extractor.base.DataExtractor` instance).
        However, I leave the flexibility to enrich your data module with the additional parameters and methods.

    Example::

        from yggdrasil.data.datamodules.manual import ManualDataModule
        from yggdrasil.data.data_extractor import DataExtractor
        from yggdrasil.core.dtypes import TableSpec, DatasetOptions

        class MyDataModule(ManualDataModule):
            def __init__(
                self,
                input_table_spec: TableSpec | None = None,
                data_extractor: DataExtractor | None = None,
                setup_data: dict[str, str] | None = None,
                saved_setup_data: dict[str, str] | None = None,
                dataset_options: DatasetOptions | None = None,
                ... # extra parameters if needed
            ) -> None:
                # Initialize the data module with the necessary parameters
                ...

            def run_feature_identification(self, table_identifier: str) -> dict[str, NormalizationData]:
                # Identify normalization parameters for the dense column
                ...

            def query_data(
                self,
                table_identifier: str,
                sample_range: tuple[float, float],
                data_extractor: DataExtractor,
            ) -> ParquetDataset:
                # Here you can validate the parameters or enrich the data extraction process
                return data_extractor.query_data(table_identifier, sample_range)

            def build_batch_preprocessor(self) -> MyBatchPreprocessor:
                # Build the batch preprocessor using the normalization parameters
                ...

    Args:
        input_table_spec (TableSpec | None, optional): Input table specification. Defaults to None.
        data_extractor (DataExtractor | None, optional): Data extractor instance. Defaults to None.
        setup_data (dict[str, str] | str | None, optional): Serialized setup data. Defaults to None.
        saved_setup_data (dict[str, str] | str | None, optional): Serialized saved setup data. Defaults to None.
        dataset_options (DatasetOptions | None, optional): Options for dataset loading. Defaults to None.
    """

    def __init__(
        self,
        *,
        input_table_spec: TableSpec | None = None,
        data_extractor: DataExtractor | None = None,
        setup_data: dict[str, str] | str | None = None,
        saved_setup_data: dict[str, str] | str | None = None,
        dataset_options: DatasetOptions | None = None,
    ) -> None:
        """
        Initialize the manual data module.

        Args:
            input_table_spec (TableSpec | None, optional): Input table specification. Defaults to None.
            data_extractor (DataExtractor | None, optional): Data extractor instance. Defaults to None.
            setup_data (dict[str, str] | str | None, optional): Serialized setup data. Defaults to None.
            saved_setup_data (dict[str, str] | str | None, optional): Serialized saved setup data. Defaults to None.
            dataset_options (DatasetOptions | None, optional): Options for dataset loading. Defaults to None.
        """
        super().__init__()

        self.input_table_spec = input_table_spec
        self.data_extractor = data_extractor
        self.setup_data = _deserialize_setup_data(setup_data)
        self.saved_setup_data = _deserialize_setup_data(saved_setup_data) or {}
        self.dataset_options = dataset_options or DatasetOptions()

    def prepare_data(self) -> None:
        """
        Prepare the data for the manual data module.

        Raises:
            ValueError: If the input table spec or data extractor is not defined.
            ValueError: If the train dataset is not defined.
        """
        if self.setup_data is not None:
            return

        if self.input_table_spec is None:
            msg = "Input table spec must be defined"
            raise ValueError(msg)
        if self.data_extractor is None:
            msg = "Data extractor must be defined"
            raise ValueError(msg)

        sample_range = get_sample_range(self.input_table_spec)
        train_dataset = self._query_ranged_data(sample_range.train_sample_range)
        eval_dataset = self._query_ranged_data(sample_range.eval_sample_range)
        test_dataset = self._query_ranged_data(sample_range.test_sample_range)

        if train_dataset is None:
            msg = "Train dataset must be defined"
            raise ValueError(msg)

        if "normalization_dict" not in self.saved_setup_data:
            normalization_dict = self.run_feature_identification(self.input_table_spec.table_identifier)
        else:
            normalization_dict = {
                nkey: NormalizationData(**ndata)
                for nkey, ndata in json.loads(self.saved_setup_data["normalization_dict"]).items()
            }
        self.setup_data = self._pickle_setup_data(train_dataset, eval_dataset, test_dataset, normalization_dict)

    def setup(self, stage: str | None = None) -> None:
        """
        Set up the manual data module.

        Args:
            stage (str | None, optional): Stage of the setup. Defaults to None.

        Raises:
            ValueError: If the setup data is not defined before calling setup.
        """
        if self._setup_done:
            return

        if self.setup_data is None:
            msg = "Setup data must be defined before calling setup."
            raise ValueError(msg)

        setup_data = {k: json.loads(v) for k, v in self.setup_data.items()}

        self._normalization_dict = {
            nkey: NormalizationData(**ndata) for nkey, ndata in setup_data["normalization_dict"].items()
        }

        self._train_data = ParquetDataset(**setup_data["train_dataset"])
        self._eval_data = ParquetDataset(**setup_data["eval_dataset"]) if setup_data["eval_dataset"] else None
        self._test_data = ParquetDataset(**setup_data["test_dataset"]) if setup_data["test_dataset"] else None

        self._setup_done = True

    def _query_ranged_data(
        self,
        sample_range: tuple[float, float] | None,
    ) -> ParquetDataset | None:
        r"""
        Query the data extractor for the data within the specified sample range.

        Args:
            sample_range (tuple[float, float] | None): Sample range to query the data for.

        Raises:
            ValueError: If the data extractor or input table spec is not defined.

        Returns:
            ParquetDataset | None: Queried data as a ParquetDataset, or None if no sample range is provided.
        """
        if self.data_extractor is None or self.input_table_spec is None:
            msg = "Data extractor and input table spec must be defined"
            raise ValueError(msg)
        return (
            self.query_data(
                table_identifier=self.input_table_spec.table_identifier,
                sample_range=sample_range,
                data_extractor=self.data_extractor,
            )
            if sample_range is not None
            else None
        )

    @staticmethod
    def _pickle_setup_data(
        train_dataset: ParquetDataset,
        eval_dataset: ParquetDataset | None,
        test_dataset: ParquetDataset | None,
        normalization_dict: dict[str, NormalizationData],
    ) -> dict[str, str]:
        """
        Serialize the setup data into a dictionary.

        Args:
            train_dataset (ParquetDataset): Training dataset.
            eval_dataset (ParquetDataset | None): Validation dataset (optional).
            test_dataset (ParquetDataset | None): Test dataset (optional).
            normalization_dict (dict[str, NormalizationData]): Normalization parameters for the dataset.

        Returns:
            dict[str, str]: Serialized setup data dictionary containing the train, eval, and test datasets,
        """
        setup_data = dict(
            train_dataset=json.dumps(train_dataset, cls=DataclassJSONEncoder),
            eval_dataset=json.dumps(eval_dataset, cls=DataclassJSONEncoder),
            test_dataset=json.dumps(test_dataset, cls=DataclassJSONEncoder),
            normalization_dict=json.dumps(normalization_dict, cls=DataclassJSONEncoder),
        )
        return setup_data

    def get_dataloader(self, dataset: ParquetDataset | None, identity: str = "default") -> DataLoader[TensorDataClass]:
        r"""
        Get a DataLoader for the given dataset.

        Args:
            dataset (ParquetDataset | None): Dataset to load.
            identity (str, optional): Identity for the DataLoader. Defaults to "default".

        Raises:
            ValueError: If the dataset URL is not defined.

        Returns:
            DataLoader[TensorDataClass]: DataLoader for the given dataset.
        """
        if dataset is None:
            msg = "Dataset URL must be defined"
            raise ValueError(msg)

        batch_preprocessor = self.build_batch_preprocessor()
        dataset_options = self.dataset_options
        hf_dataset = load_parquet_dataset(
            dataset.dataset_url,
            num_proc=dataset_options.num_proc,
            streaming=dataset_options.streaming,
        ).with_format("numpy")

        dataloader = DataLoader(
            cast(TorchDataset, hf_dataset),
            batch_size=dataset_options.minibatch_size,
            collate_fn=collate_and_preprocess(batch_preprocessor=batch_preprocessor),
        )
        return dataloader

    def train_dataloader(self) -> DataLoader[TensorDataClass]:
        r"""
        Get the DataLoader for the training dataset.

        Returns:
            DataLoader[TensorDataClass]: DataLoader for the training dataset.
        """
        self._num_train_data_loader_calls += 1
        return self.get_dataloader(self.train_data, identity=f"train_{self._num_train_data_loader_calls}")

    def test_dataloader(self) -> DataLoader[TensorDataClass] | None:
        r"""
        Get the DataLoader for the test dataset.

        Returns:
            DataLoader[TensorDataClass] | None: DataLoader for the test dataset, or None if no test dataset is defined.
        """
        self._num_test_data_loader_calls += 1
        return self._get_optional_data(self.test_data, identity=f"test_{self._num_test_data_loader_calls}")

    def val_dataloader(self) -> DataLoader[TensorDataClass] | None:
        r"""
        Get the DataLoader for the validation dataset.

        Returns:
            DataLoader[TensorDataClass] | None: DataLoader for the val dataset, or None if no val dataset is defined.
        """
        self._num_val_data_loader_calls += 1
        return self._get_optional_data(self.eval_data, identity=f"eval_{self._num_val_data_loader_calls}")

    def _get_optional_data(self, dataset: ParquetDataset | None, identity: str) -> DataLoader[TensorDataClass] | None:
        r"""
        Get the DataLoader for an optional dataset.

        Args:
            dataset (ParquetDataset | None): Dataset to load.
            identity (str): Identity for the DataLoader.

        Returns:
            DataLoader[TensorDataClass] | None: DataLoader for the dataset, or None if no dataset is defined.
        """
        return None if not dataset else self.get_dataloader(dataset, identity)
