from copy import deepcopy
from functools import partial
from typing import Any

import lightning as L
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MaxMetric, MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
)

_DEFAULT_BIN_CLASSIFICATION_METRICS = MetricCollection(
    [
        BinaryAccuracy(),
        BinaryF1Score(),
        BinaryAUROC(),
    ]
)

_CRITERION = nn.BCEWithLogitsLoss()
_POPTIMIZER = partial(torch.optim.Adam, lr=1e-3)


class BinaryClassificationModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,  # we are expecting here a network outputing logits
        criterion: nn.Module = _CRITERION,
        optimizer: partial[torch.optim.Optimizer] = _POPTIMIZER,
        scheduler: partial[LRScheduler] | None = None,
        lr_scheduler_config: dict | None = None,
        metrics: MetricCollection = _DEFAULT_BIN_CLASSIFICATION_METRICS,
    ):
        super().__init__()
        # lightning automatically saves and logs all module arguments to `self.hparams`
        # attribute, except the ignored ones. We also disable logging of the
        # hyperparameters since we perform it manually for better control
        self.save_hyperparameters(logger=False, ignore=["net", "criterion", "metrics"])

        self.net = net
        self.criterion = criterion
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        # object to manually track the max matrics of our metrics over all epochs
        self.max_metrics = nn.ModuleDict(
            {"max_" + name: MaxMetric() for name in metrics}
        )

    def forward(self, *args: Any, **kwargs) -> Any:
        return self.net(*args, **kwargs)

    def configure_optimizers(self) -> dict[str, Any]:
        """
        Required function configuring the optimizer and the learning rate scheduler
        """
        config = {}
        optimizer = self.hparams.optimizer(self.parameters())
        config["optimizer"] = optimizer

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            lr_scheduler_config = deepcopy({} or self.hparams.lr_scheduler_config)
            lr_scheduler_config["scheduler"] = scheduler
            config["lr_scheduler"] = lr_scheduler_config
        return config

    def _unpack_losses(
        self,
        losses: torch.Tensor | dict[str, torch.Tensor],
        prefix: str = "",
        **log_args,
    ) -> torch.Tensor:
        """
        Utility function to log multiple loss components as separate metrics
        if the model forward step produces a dictionary of losses
        """
        if isinstance(losses, dict):
            self.log_dict({prefix + k: v for k, v in losses.items()}, **log_args)
            return sum(losses.values())
        return losses

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks
        # before training starts, so it's worth to make sure validation
        # metrics don't store results from these checks
        self.train_metrics.reset()
        self.valid_metrics.reset()
        for m in self.max_metrics.values():
            m.reset()

    def _single_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        metrics: MetricCollection,
        prefix: str,
    ) -> torch.Tensor:
        x, y = batch["features"], batch["target"]
        pred = self(x)
        loss = self.criterion(pred, y)
        loss = self._unpack_losses(
            loss, prefix=prefix + "_", on_step=True, on_epoch=True
        )

        metrics.update(torch.sigmoid(pred), y)
        self.log(prefix, loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step, on_step=True, on_epoch=False)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._single_step(batch, batch_idx, self.train_metrics, "train_loss")

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._single_step(batch, batch_idx, self.valid_metrics, "val_loss")

    def on_validation_epoch_end(self) -> None:
        metrics = self.valid_metrics.compute()
        # max metrics are accumulated across all epochs
        # that's why we need to update them manually
        for max_metric_mod, metric_val in zip(
            self.max_metrics.values(), metrics.values(), strict=True
        ):
            max_metric_mod.update(metric_val)
        # log through `.compute()` method instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log_dict(
            {k: v.compute() for k, v in self.max_metrics.items()},
            sync_dist=True,
        )

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary with the model predictions alongside all
        the other possible metadata passed inside the inference batch
        """
        pred = torch.sigmoid(self(batch["features"]))
        out = {
            **{k: v for k, v in batch.items() if k != "features"},
            "pred": pred,
        }
        return out
