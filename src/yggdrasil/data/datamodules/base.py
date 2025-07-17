import abc
from collections.abc import Callable

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from yggdrasil.core.dataclasses import dataclass
from yggdrasil.core.dtypes.base import TensorDataClass
from yggdrasil.core.dtypes.dataset import ParquetDataset, TableSpec
from yggdrasil.core.dtypes.parameters import NormalizationData
from yggdrasil.data.data_extractor.base import DataExtractor
from yggdrasil.preprocessing.batch_preprocessor import BatchPreprocessor


def collate_and_preprocess(
    batch_preprocessor: BatchPreprocessor,
) -> Callable[[list[dict]], torch.Tensor]:
    r"""
    Default colate function for DataLoader with batch preprocessing.

    Example::

        from datasets import load_dataset
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            load_dataset(...),
            batch_size=64,
            collate_fn=collate_and_preprocess(
                batch_preprocessor=ExampleBatchPreprocessor()
        )

        # batch == BinaryPreprocessedInput(features=..., target=...)
        batch = next(iter(dataloader))

    .. warning::
        :func:`~torch.utils.data.default_collate` doesn't properly handle Decimals.
        In case of encoding timestamps as Decimals, consider using another collate
        function.

    .. warning::
        In case of distributed training this function needs to be pickled. It's
        imposible to pickle because we define :func:`collate_fn` inside this function.

    Args:
        batch_preprocessor (BatchPreprocessor): Preprocessor applied to a batch.

    Returns:
        ~collections.abc.Callable[[list[dict]], torch.Tensor]:
            Collate function for DataLoader.
    """

    def collate_fn(batch_list: list[dict]) -> torch.Tensor:
        batch = default_collate(batch_list)
        preprocessed_batch: torch.Tensor = batch_preprocessor(batch)
        return preprocessed_batch

    return collate_fn


@dataclass
class TrainEvalTestSampleRanges:
    """
    Dataclass to hold sample ranges for train, eval, and test datasets.
    Train sample range is always required, while eval and test sample ranges can be None.

    Attributes:
        train_sample_range (tuple[float, float]): Range of samples for the training dataset.
        eval_sample_range (tuple[float, float] | None): Range of samples for the evaluation dataset.
        test_sample_range (tuple[float, float] | None): Range of samples for the test dataset.

    Example::

        from yggdrasil.data.datamodules.base import TrainEvalTestSampleRanges
        sample_ranges = TrainEvalTestSampleRanges(
            train_sample_range=(0.0, 70.0),
            eval_sample_range=(70.0, 85.0),
            test_sample_range=(85.0, 100.0)
        )
        print(sample_ranges)

        # Output:
        # TrainEvalTestSampleRanges(
        #     train_sample_range=(0.0, 70.0),
        #     eval_sample_range=(70.0, 85.0),
        #     test_sample_range=(85.0, 100.0)
        # )
    """
    train_sample_range: tuple[float, float]
    eval_sample_range: tuple[float, float] | None
    test_sample_range: tuple[float, float] | None


def get_sample_range(input_table_spec: TableSpec) -> TrainEvalTestSampleRanges:
    """
    Get sample ranges for train, eval, and test datasets based on the input table specification.
    The function takes :class:`~yggdrasil.core.dtypes.dataset.TableSpec` as input and returns.
    As the result it returns the ranges which are used for sampling the datasets.

    Example::

        from yggdrasil.data.datamodules.base import get_sample_range
        from yggdrasil.core.dtypes import TableSpec

        input_table_spec = TableSpec(
            train_table_sample=70.0,
            eval_table_sample=15.0,
            test_table_sample=15.0
        )
        sample_ranges = get_sample_range(input_table_spec)

        # Output:
        # TrainEvalTestSampleRanges(
        #     train_sample_range=(0.0, 70.0),
        #     eval_sample_range=(70.0, 85.0),
        #     test_sample_range=(85.0, 100.0)
        # )

    Args:
        input_table_spec (TableSpec): Specification of the input table containing sample ranges.

    Raises:
        ValueError: If the sum of train, eval, and test sample ranges exceeds 100.

    Returns:
        TrainEvalTestSampleRanges: A dataclass containing the sample ranges for train, eval, and test datasets.
    """
    train_sample = input_table_spec.train_table_sample or 100.0
    eval_sample = input_table_spec.eval_table_sample or 0.0
    test_sample = input_table_spec.test_table_sample or 0.0

    if train_sample + eval_sample + test_sample > 100.0 + 1e-6:
        msg = "Sum of sample ranges must be equal to 100."
        raise ValueError(msg)

    train_sample_range = (0.0, train_sample)
    eval_sample_range = (train_sample, train_sample + eval_sample) if eval_sample else None
    test_sample_range = (train_sample + eval_sample, train_sample + eval_sample + test_sample) if test_sample else None

    return TrainEvalTestSampleRanges(
        train_sample_range=train_sample_range,
        eval_sample_range=eval_sample_range,
        test_sample_range=test_sample_range,
    )


class BaseDataModule(abc.ABC, L.LightningDataModule):
    r"""
    Base class for data modules. See the :class:`~lightning.pytorch.core.datamodule.LightningDataModule`
    for the workflow of the data module. The base data module dictates the interface to tie
    the the lightning data module with the :class:`~yggdrasil.core.dtypes.base.TensorDataClass`
    fromat specified by Yggdrasil.

    Ovrride the following methods:

    1. :meth:`prepare_data` to prepare the data for the data module.
    2. :meth:`setup` to set up the data module for the specified stage.
    3. :meth:`build_batch_preprocessor` to build the batch preprocessor for the data module.
    4. :meth:`query_data` to query the data using the provided data extractor.
    5. :meth:`run_feature_identification` to run feature identification.
    6. :meth:`train_dataloader` to get the training data loader.
    7. :meth:`val_dataloader` to get the validation data loader.
    8. :meth:`test_dataloader` to get the test data loader.
    """

    _normalization_dict: dict[str, NormalizationData] | None
    _train_data: ParquetDataset | None
    _eval_data: ParquetDataset | None
    _test_data: ParquetDataset | None

    def __init__(self) -> None:
        super().__init__()
        self._setup_done = False
        self._num_train_data_loader_calls = 0
        self._num_val_data_loader_calls = 0
        self._num_test_data_loader_calls = 0

    @abc.abstractmethod
    def prepare_data(self) -> None:
        r"""
        Prepare the data for the data module.
        """

    @abc.abstractmethod
    def setup(self, stage: str | None) -> None:
        r"""
        Setup the data module for the specified stage.

        Args:
            stage (str | None): Stage of the data module. If None, setup for all stages.
        """

    @abc.abstractmethod
    def build_batch_preprocessor(self) -> BatchPreprocessor:
        r"""
        Build the batch preprocessor for the data module.

        Returns:
            BatchPreprocessor: Batch preprocessor for the data module.
        """

    @abc.abstractmethod
    def query_data(
        self,
        table_identifier: str,
        sample_range: tuple[float, float],
        data_extractor: DataExtractor,
    ) -> ParquetDataset:
        r"""
        Query the data from the specified table identifier and sample range using the provided data extractor.

        Args:
            table_identifier (str): Identifier of the table to query.
            sample_range (tuple[float, float]): Range of samples to query.
            data_extractor (DataExtractor): Data extractor to use for querying the data.

        Returns:
            ParquetDataset: Dataset containing the queried data.
        """

    @abc.abstractmethod
    def run_feature_identification(self, table_identifier: str) -> dict[str, NormalizationData]:
        """
        Run feature identification on the specified table identifier.

        Args:
            table_identifier (str): Identifier of the table to run feature identification on.

        Returns:
            dict[str, NormalizationData]: Dictionary mapping feature names to their normalization data.
        """

    @property
    def normalization_dict(self) -> dict[str, NormalizationData] | None:
        r"""
        Get the normalization dictionary containing normalization data for features.

        Returns:
            dict[str, NormalizationData] | None: Dictionary mapping feature names to their normalization data,
        """
        return getattr(self, "_normalization_dict", None)

    @property
    def train_data(self) -> ParquetDataset | None:
        r"""
        Get the training data.

        Returns:
            ParquetDataset | None: Training dataset if available, otherwise None.
        """
        return getattr(self, "_train_data", None)

    @property
    def eval_data(self) -> ParquetDataset | None:
        r"""
        Get the evaluation data.

        Returns:
            ParquetDataset | None: Evaluation dataset if available, otherwise None.
        """
        return getattr(self, "_eval_data", None)

    @property
    def test_data(self) -> ParquetDataset | None:
        r"""
        Get the test data.

        Returns:
            ParquetDataset | None: Test dataset if available, otherwise None.
        """
        return getattr(self, "_test_data", None)

    @abc.abstractmethod
    def train_dataloader(self) -> DataLoader[TensorDataClass]:
        r"""
        Get the training data loader.

        Returns:
            DataLoader[TensorDataClass]: Data loader for the training dataset.
        """

    @abc.abstractmethod
    def val_dataloader(self) -> DataLoader[TensorDataClass] | None:
        """
        Get the validation data loader.

        Returns:
            DataLoader[TensorDataClass] | None: Data loader for the val dataset, or None if no val data is available.
        """

    @abc.abstractmethod
    def test_dataloader(self) -> DataLoader[TensorDataClass] | None:
        r"""
        Get the test data loader.

        Returns:
            DataLoader[TensorDataClass] | None: Data loader for the test dataset, or None if no test data is available.
        """
