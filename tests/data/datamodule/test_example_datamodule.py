import pytest
import torch
from torch.utils.data import DataLoader

from project.core.dtypes.classification import BinaryPreprocessedInput
from project.data.datamodules.example import (
    ExampleBatchPreprocessor,
    ExampleDataModule,
    collate_and_preprocess,
)
from project.data.dataset.random import RandomDataset


class TestExampleBatchPreprocessor:
    def test_forward(self) -> None:
        batch_preprocessor = ExampleBatchPreprocessor()
        batch = {"features": torch.randn(16, 64), "target": torch.randint(0, 2, (16,))}
        preprocessed_batch = batch_preprocessor.forward(batch)
        assert isinstance(preprocessed_batch, BinaryPreprocessedInput)
        assert preprocessed_batch.features.shape == (16, 64)
        assert preprocessed_batch.target.shape == (16, 1)


class TestCollateAndPreprocess:
    def test_collate_and_preprocess(self) -> None:
        batch_preprocessor = ExampleBatchPreprocessor()
        collate_fn = collate_and_preprocess(batch_preprocessor)
        batch_list = [
            {"features": torch.randn(64), "target": torch.tensor(1)},
            {"features": torch.randn(64), "target": torch.tensor(0)},
        ]
        preprocessed_batch = collate_fn(batch_list)
        assert isinstance(preprocessed_batch, BinaryPreprocessedInput)
        assert preprocessed_batch.features.shape == (2, 64)
        assert preprocessed_batch.target.shape == (2, 1)


class TestExampleDataModule:
    @pytest.fixture
    def data_module(self) -> ExampleDataModule:
        dataset = RandomDataset(num_features=64, length=1024)
        return ExampleDataModule(dataset=dataset, batch_size=16, num_workers=0)

    def test_train_dataloader(self, data_module: ExampleDataModule) -> None:
        dataloader = data_module.train_dataloader()
        assert isinstance(dataloader, DataLoader)
        batch = next(iter(dataloader))
        assert isinstance(batch, BinaryPreprocessedInput)
        assert batch.features.shape == (16, 64)
        assert batch.target.shape == (16, 1)

    def test_val_dataloader(self, data_module: ExampleDataModule) -> None:
        dataloader = data_module.val_dataloader()
        assert isinstance(dataloader, DataLoader)
        batch = next(iter(dataloader))
        assert isinstance(batch, BinaryPreprocessedInput)
        assert batch.features.shape == (16, 64)
        assert batch.target.shape == (16, 1)

    def test_test_dataloader(self, data_module: ExampleDataModule) -> None:
        dataloader = data_module.test_dataloader()
        assert isinstance(dataloader, DataLoader)
        batch = next(iter(dataloader))
        assert isinstance(batch, BinaryPreprocessedInput)
        assert batch.features.shape == (16, 64)
        assert batch.target.shape == (16, 1)

    def test_example_datamodule(self, data_module: ExampleDataModule) -> None:
        datamodule = data_module
        datamodule.prepare_data()
        datamodule.setup("fit")

        num_steps = sum(1 for _ in datamodule.train_dataloader())
        assert num_steps == 1024 // 16
