from copy import deepcopy
from functools import partial
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from project.core.config import param_hash
from project.core.dataclasses import dataclass
from project.core.dtypes.classification.base import (
    BinaryOutput,
    BinaryPreprocessedInput,
)
from project.models.base import BaseModel
from project.modules.binary_classification import BinaryClassificationModule


@dataclass
class DummyNetwork(BaseModel):
    __hash__ = param_hash

    def __post_init__(self) -> None:
        super().__init__()
        self.model = self._build_model()

    def _build_model(self) -> nn.Module:
        return nn.Sequential(nn.Linear(64, 1))

    def forward(self, batch: BinaryPreprocessedInput) -> BinaryOutput:
        logits = self.model(batch.features.dense_features)
        return BinaryOutput(probabilities=torch.sigmoid(logits), logits=logits)


class TestBinaryClassificationModule:
    @pytest.fixture
    def setup_model(self) -> BinaryClassificationModule:
        net = DummyNetwork()
        return BinaryClassificationModule(net)

    def test_optimizer_initialization(
        self, setup_model: BinaryClassificationModule
    ) -> None:
        model = setup_model
        optimizers = model.configure_optimizers()
        optimizer = optimizers["optimizer"]

        assert isinstance(optimizer, Adam)
        params = list(optimizer.param_groups[0]["params"])
        assert len(params) == len(list(model.parameters()))

    def test_loss_computation(self, setup_model: BinaryClassificationModule) -> None:
        model = setup_model
        batch = BinaryPreprocessedInput.from_tensors(
            features=torch.randn(8, 64),
            target=torch.randint(0, 2, (8, 1)).float(),
        )
        loss = model._single_step(batch)["loss"]
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])

    def test_weight_update_after_backward(
        self, setup_model: BinaryClassificationModule
    ) -> None:
        model = setup_model
        optimizer = model.configure_optimizers()["optimizer"]
        initial_weights = deepcopy(list(model.net.parameters()))
        batch = BinaryPreprocessedInput.from_tensors(
            features=torch.randn(8, 64),
            target=torch.randint(0, 2, (8, 1)).float(),
        )
        optimizer.zero_grad()
        with patch.object(model, "log"):
            loss = model._single_step(batch)["loss"]
        loss.backward()
        optimizer.step()

        updated_weights = list(model.net.parameters())
        for initial, updated in zip(initial_weights, updated_weights, strict=False):
            assert not torch.equal(
                initial, updated
            ), "Weights did not update after optimization step."


class TestBinaryClassificationModuleLRScheduling:
    def test_lr_scheduler_step(self) -> None:
        net = DummyNetwork()
        optimizer = Adam(net.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        model = BinaryClassificationModule(
            net,
            optimizer=partial(Adam, lr=1e-3),
            scheduler=partial(StepLR, step_size=10, gamma=0.1),
        )

        optimizers = model.configure_optimizers()
        lr_scheduler = optimizers["lr_scheduler"]["scheduler"]

        assert isinstance(lr_scheduler, StepLR)
        initial_lr = optimizer.param_groups[0]["lr"]

        for _ in range(11):
            optimizer.step()
            scheduler.step()

        assert optimizer.param_groups[0]["lr"] == initial_lr * 0.1
