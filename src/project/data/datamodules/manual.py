import json
import os
from typing import cast

from datasets import Dataset as HFDataset
from datasets import DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from project.core.dataclasses import DataclassJSONEncoder
from project.core.dtypes.base import TensorDataClass
from project.core.dtypes.dataset import ParquetDataset, TableSpec
from project.core.dtypes.options import DatasetOptions
from project.core.dtypes.parameters import NormalizationData
from project.data.data_extractor.base import DataExtractor
from project.data.datamodules.base import (
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
    files = [
        os.path.join(file_path, f)
        for f in os.listdir(file_path)
        if f.endswith(".parquet")
    ]
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
    def __init__(
        self,
        *,
        input_table_spec: TableSpec | None = None,
        data_extractor: DataExtractor | None = None,
        setup_data: dict[str, str] | str | None = None,
        saved_setup_data: dict[str, str] | str | None = None,
        dataset_options: DatasetOptions | None = None,
    ) -> None:
        super().__init__()

        self.input_table_spec = input_table_spec
        self.data_extractor = data_extractor
        self.setup_data = _deserialize_setup_data(setup_data)
        self.saved_setup_data = _deserialize_setup_data(saved_setup_data) or {}
        self.dataset_options = dataset_options or DatasetOptions()

    def prepare_data(self) -> None:
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
            normalization_dict = self.run_feature_identification(
                self.input_table_spec.table_identifier
            )
        else:
            normalization_dict = {
                nkey: NormalizationData(**ndata)
                for nkey, ndata in json.loads(
                    self.saved_setup_data["normalization_dict"]
                ).items()
            }
        self.setup_data = self._pickle_setup_data(
            train_dataset, eval_dataset, test_dataset, normalization_dict
        )

    def setup(self, stage: str | None = None) -> None:
        if self._setup_done:
            return

        if self.setup_data is None:
            msg = "Setup data must be defined before calling setup."
            raise ValueError(msg)

        setup_data = {k: json.loads(v) for k, v in self.setup_data.items()}

        self._normalization_dict = {
            nkey: NormalizationData(**ndata)
            for nkey, ndata in setup_data["normalization_dict"].items()
        }

        self._train_data = ParquetDataset(**setup_data["train_dataset"])
        self._eval_data = (
            ParquetDataset(**setup_data["eval_dataset"])
            if setup_data["eval_dataset"]
            else None
        )
        self._test_data = (
            ParquetDataset(**setup_data["test_dataset"])
            if setup_data["test_dataset"]
            else None
        )

        self._setup_done = True

    def _query_ranged_data(
        self,
        sample_range: tuple[float, float] | None,
    ) -> ParquetDataset | None:
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
        setup_data = dict(
            train_dataset=json.dumps(train_dataset, cls=DataclassJSONEncoder),
            eval_dataset=json.dumps(eval_dataset, cls=DataclassJSONEncoder),
            test_dataset=json.dumps(test_dataset, cls=DataclassJSONEncoder),
            normalization_dict=json.dumps(normalization_dict, cls=DataclassJSONEncoder),
        )
        return setup_data

    def get_dataloader(
        self, dataset: ParquetDataset | None, identity: str = "default"
    ) -> DataLoader[TensorDataClass]:
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
        self._num_train_data_loader_calls += 1
        return self.get_dataloader(
            self.train_data, identity=f"train_{self._num_train_data_loader_calls}"
        )

    def test_dataloader(self) -> DataLoader[TensorDataClass] | None:
        self._num_test_data_loader_calls += 1
        return self._get_optional_data(
            self.test_data, identity=f"test_{self._num_test_data_loader_calls}"
        )

    def val_dataloader(self) -> DataLoader[TensorDataClass] | None:
        self._num_val_data_loader_calls += 1
        return self._get_optional_data(
            self.eval_data, identity=f"eval_{self._num_val_data_loader_calls}"
        )

    def _get_optional_data(
        self, dataset: ParquetDataset | None, identity: str
    ) -> DataLoader[TensorDataClass] | None:
        return None if not dataset else self.get_dataloader(dataset, identity)
