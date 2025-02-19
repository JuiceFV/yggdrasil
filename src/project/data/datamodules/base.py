import abc
from collections.abc import Callable

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from project.core.dataclasses import dataclass
from project.core.dtypes.base import TensorDataClass
from project.core.dtypes.dataset import ParquetDataset, TableSpec
from project.core.dtypes.parameters import NormalizationData
from project.data.data_extractor.base import DataExtractor
from project.preprocessing.batch_preprocessor import BatchPreprocessor


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
    train_sample_range: tuple[float, float]
    eval_sample_range: tuple[float, float] | None
    test_sample_range: tuple[float, float] | None


def get_sample_range(input_table_spec: TableSpec) -> TrainEvalTestSampleRanges:
    train_sample = input_table_spec.train_table_sample or 100.0
    eval_sample = input_table_spec.eval_table_sample or 0.0
    test_sample = input_table_spec.test_table_sample or 0.0

    if train_sample + eval_sample + test_sample > 100.0 + 1e-6:
        msg = "Sum of sample ranges must be equal to 100."
        raise ValueError(msg)

    train_sample_range = (0.0, train_sample)
    eval_sample_range = (
        (train_sample, train_sample + eval_sample) if eval_sample else None
    )
    test_sample_range = (
        (train_sample + eval_sample, train_sample + eval_sample + test_sample)
        if test_sample
        else None
    )

    return TrainEvalTestSampleRanges(
        train_sample_range=train_sample_range,
        eval_sample_range=eval_sample_range,
        test_sample_range=test_sample_range,
    )


class BaseDataModule(abc.ABC, L.LightningDataModule):
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
        pass

    @abc.abstractmethod
    def setup(self, stage: str | None) -> None:
        pass

    @abc.abstractmethod
    def build_batch_preprocessor(self) -> BatchPreprocessor:
        pass

    @abc.abstractmethod
    def query_data(
        self,
        table_identifier: str,
        sample_range: tuple[float, float],
        data_extractor: DataExtractor,
    ) -> ParquetDataset:
        pass

    @abc.abstractmethod
    def run_feature_identification(
        self, table_identifier: str
    ) -> dict[str, NormalizationData]:
        pass

    @property
    def normalization_dict(self) -> dict[str, NormalizationData] | None:
        return getattr(self, "_normalization_dict", None)

    @property
    def train_data(self) -> ParquetDataset | None:
        return getattr(self, "_train_data", None)

    @property
    def eval_data(self) -> ParquetDataset | None:
        return getattr(self, "_eval_data", None)

    @property
    def test_data(self) -> ParquetDataset | None:
        return getattr(self, "_test_data", None)

    @abc.abstractmethod
    def train_dataloader(self) -> DataLoader[TensorDataClass]:
        pass

    @abc.abstractmethod
    def eval_dataloader(self) -> DataLoader[TensorDataClass] | None:
        pass

    @abc.abstractmethod
    def test_dataloader(self) -> DataLoader[TensorDataClass] | None:
        pass
