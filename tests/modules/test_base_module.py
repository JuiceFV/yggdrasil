import logging
from collections.abc import Generator
from unittest.mock import PropertyMock, patch

import pytest
import torch

from project.core.dtypes.base import TensorDataClass
from project.modules.base import BaseModule


class DummyTensorDataClass(TensorDataClass): ...


class DummyBaseModule(BaseModule):
    def __init__(self) -> None:
        super().__init__()
        self._automatic_optimization = True
        self._training_step_gen = None
        self._verified_steps = False
        self.train_batches_processed_this_epoch = 0
        self.val_batches_processed_this_epoch = 0
        self.test_batches_processed_this_epoch = 0
        self.all_batches_processed = 0

    def train_step_gen(
        self, training_batch: TensorDataClass, batch_idx: int
    ) -> Generator[torch.Tensor, None, None]:
        yield torch.tensor(0.0)


class TestBaseModuleTests:
    @pytest.fixture
    def base_module(self) -> DummyBaseModule:
        return DummyBaseModule()

    def test_initialization(self, base_module: DummyBaseModule) -> None:
        assert base_module._automatic_optimization is True
        assert base_module._training_step_gen is None
        assert base_module._verified_steps is False
        assert base_module.train_batches_processed_this_epoch == 0
        assert base_module.val_batches_processed_this_epoch == 0
        assert base_module.test_batches_processed_this_epoch == 0
        assert base_module.all_batches_processed == 0

    def test_training_step(self, base_module: DummyBaseModule) -> None:
        batch = DummyTensorDataClass()
        batch_idx = 0
        output = base_module.training_step(batch, batch_idx)
        assert output.item() == 0.0

    @patch.object(DummyBaseModule, "current_epoch", new_callable=PropertyMock)
    def test_on_train_epoch_end(
        self,
        mock_epoch: PropertyMock,
        base_module: DummyBaseModule,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_epoch.return_value = 1
        base_module.train_batches_processed_this_epoch = 10
        with caplog.at_level(logging.INFO):
            base_module.on_train_epoch_end()
        assert base_module.train_batches_processed_this_epoch == 0
        assert "Finished train epoch 1 with 10 batches processed" in caplog.text

    @patch.object(DummyBaseModule, "current_epoch", new_callable=PropertyMock)
    def test_on_validation_epoch_end(
        self,
        mock_epoch: PropertyMock,
        base_module: DummyBaseModule,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_epoch.return_value = 1
        base_module.val_batches_processed_this_epoch = 5
        with caplog.at_level(logging.INFO):
            base_module.on_validation_epoch_end()
        assert base_module.val_batches_processed_this_epoch == 0
        assert "Finished validation epoch 1 with 5 batches processed" in caplog.text

    @patch.object(DummyBaseModule, "current_epoch", new_callable=PropertyMock)
    def test_on_test_epoch_end(
        self,
        mock_epoch: PropertyMock,
        base_module: DummyBaseModule,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mock_epoch.return_value = 1
        base_module.test_batches_processed_this_epoch = 3
        with caplog.at_level(logging.INFO):
            base_module.on_test_epoch_end()
        assert base_module.test_batches_processed_this_epoch == 0
        assert "Finished test epoch 1 with 3 batches processed" in caplog.text

    def test_on_train_batch_end(self, base_module: DummyBaseModule) -> None:
        base_module.on_train_batch_end()
        assert base_module.train_batches_processed_this_epoch == 1
        assert base_module.all_batches_processed == 1

    def test_on_validation_batch_end(self, base_module: DummyBaseModule) -> None:
        base_module.on_validation_batch_end()
        assert base_module.val_batches_processed_this_epoch == 1
        assert base_module.all_batches_processed == 1

    def test_on_test_batch_end(self, base_module: DummyBaseModule) -> None:
        base_module.on_test_batch_end()
        assert base_module.test_batches_processed_this_epoch == 1
        assert base_module.all_batches_processed == 1
