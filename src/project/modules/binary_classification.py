from collections.abc import Generator
from copy import deepcopy
from functools import partial

import torch
from torch import nn

from project.core.dtypes import (
    BinaryOutput,
    BinaryPreprocessedInput,
    MetricInput,
)
from project.models.base import BaseModel
from project.modules.base import STEP_OUTPUT, BaseModule


class BinaryClassificationModule(BaseModule):
    def __init__(
        self,
        net: BaseModel,
        optimizer: partial[torch.optim.Optimizer] | None = None,
        scheduler: partial[torch.optim.lr_scheduler.LRScheduler] | None = None,
        lr_scheduler_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.optimizer = optimizer or partial(torch.optim.Adam, lr=1e-3)
        self.scheduler = scheduler
        self.lr_scheduler_config = lr_scheduler_config or {}
        self.criterion = nn.BCEWithLogitsLoss()

    def configure_optimizers(self):  # type: ignore # noqa: ANN201
        config = {}
        optimizer = self.optimizer(self.parameters())
        config["optimizer"] = optimizer

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            lr_scheduler_config = deepcopy(self.lr_scheduler_config)
            lr_scheduler_config["scheduler"] = scheduler
            config["lr_scheduler"] = lr_scheduler_config
        return config

    def _single_step(
        self,
        batch: BinaryPreprocessedInput,
    ) -> dict[str, torch.Tensor | MetricInput]:
        # NOTE: Binary classification task is restricted by the data
        # acceptable for this task. Probably, it's better to add
        # isinstance checks for the input data.
        out: BinaryOutput = self.net(batch)
        loss = self.criterion(out.logits, batch.target)
        metric_input = MetricInput(preds=out.probabilities, target=batch.target)
        return {"loss": loss, "metric_input": metric_input}

    def train_step_gen(  # type: ignore
        self, training_batch: BinaryPreprocessedInput, batch_idx: int
    ) -> Generator[STEP_OUTPUT, None, None]:
        yield self._single_step(training_batch)

    def validation_step(
        self, valid_batch: BinaryPreprocessedInput, batch_idx: int
    ) -> STEP_OUTPUT:
        return self._single_step(valid_batch)

    def test_step(
        self, test_batch: BinaryPreprocessedInput, batch_idx: int
    ) -> STEP_OUTPUT:
        return self._single_step(test_batch)
