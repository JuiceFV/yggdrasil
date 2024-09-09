from copy import deepcopy
from functools import partial

import lightning as L
import torch
import torch.nn as nn
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


class BinaryClassificationModel(L.LightningModule):
    def __init__(
        self,
        net: nn.Module,  # we are expecting here a network outputing logits
        criterion=nn.BCEWithLogitsLoss(),
        optimizer: partial[torch.optim.Optimizer] = partial(torch.optim.Adam, lr=1e-3),
        scheduler: partial[LRScheduler] | None = None,
        lr_scheduler_config: dict[str, str] | None = {
            "monitor": "val_loss_epoch",
            "interval": "epoch",
            "frequency": 1,
        },
        metrics=_DEFAULT_BIN_CLASSIFICATION_METRICS,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "criterion", "metrics"])

        self.net = net
        self.criterion = criterion
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.max_metrics = nn.ModuleDict(
            {"max_" + name: MaxMetric() for name in metrics}
        )

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def configure_optimizers(self):
        config = {}
        optimizer = self.hparams.optimizer(self.parameters())
        config["optimizer"] = optimizer

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            lr_scheduler_config = deepcopy(self.hparams.lr_scheduler_config)
            lr_scheduler_config["scheduler"] = scheduler
            config["lr_scheduler"] = lr_scheduler_config
        return config

    def _unpack_losses(
        self, losses: torch.Tensor | dict[str, torch.Tensor], prefix="", **log_args
    ):
        if isinstance(losses, dict):
            self.log_dict({prefix + k: v for k, v in losses.items()}, **log_args)
            return sum(losses.values())
        return losses

    def on_train_start(self):
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
    ):
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

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._single_step(batch, batch_idx, self.train_metrics, "train_loss")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        return self._single_step(batch, batch_idx, self.valid_metrics, "val_loss")

    def on_validation_epoch_end(self):
        metrics = self.valid_metrics.compute()
        # min metrics are accumulated across all epochs
        # that's why we need to update them manually
        for min_metric_mod, metric_val in zip(
            self.max_metrics.values(), metrics.values()
        ):
            min_metric_mod.update(metric_val)
        # log through `.compute()` method instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log_dict(
            {k: v.compute() for k, v in self.max_metrics.items()},
            sync_dist=True,
        )

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        pred = torch.sigmoid(self(batch["features"]))
        out = {
            **{k: v for k, v in batch.items() if k != "features"},
            "pred": pred,
        }
        return out
