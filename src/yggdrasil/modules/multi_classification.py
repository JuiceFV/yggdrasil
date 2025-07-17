from collections.abc import Callable, Generator, Iterable
from copy import deepcopy
from functools import partial

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from yggdrasil.core.dtypes import (
    ClassificationOutput,
    MetricInput,
    MulticlassPreprocessedInput,
)
from yggdrasil.models.base import BaseModel
from yggdrasil.modules.base import BaseModule

_STEP_T = dict[str, torch.Tensor | MetricInput]


class MultiClassificationModule(BaseModule[MulticlassPreprocessedInput, _STEP_T]):
    def __init__(
        self,
        net: BaseModel,
        optimizer: Callable[[Iterable[torch.nn.Parameter]], Optimizer] | None = None,
        scheduler: Callable[[Optimizer], LRScheduler] | None = None,
        lr_scheduler_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.optimizer = optimizer or partial(torch.optim.Adam, lr=1e-3)
        self.scheduler = scheduler
        self.lr_scheduler_config = lr_scheduler_config or {}
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):  # type: ignore # noqa: ANN201
        config = {}
        optimizer = self.optimizer(self.parameters())
        config["optimizer"] = optimizer

        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer)
            lr_scheduler_config = deepcopy(self.lr_scheduler_config)
            lr_scheduler_config["scheduler"] = scheduler
            config["lr_scheduler"] = lr_scheduler_config
        return config

    def _single_step(
        self,
        batch: MulticlassPreprocessedInput,
    ) -> _STEP_T:
        out: ClassificationOutput = self.net(batch)
        loss: torch.Tensor = self.criterion(out.logits, batch.labeled_target)
        metric_input = MetricInput(preds=out.probabilities, target=batch.labeled_target)
        return {"loss": loss, "metric_input": metric_input}

    def train_step_gen(
        self, training_batch: MulticlassPreprocessedInput, batch_idx: int
    ) -> Generator[_STEP_T, None, None]:
        yield self._single_step(training_batch)

    def validation_step(self, valid_batch: MulticlassPreprocessedInput, batch_idx: int) -> _STEP_T:
        return self._single_step(valid_batch)

    def test_step(self, test_batch: MulticlassPreprocessedInput, batch_idx: int) -> _STEP_T:
        return self._single_step(test_batch)
