from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import lightning as L
import torch
from torch import nn
from torchmetrics import MaxMetric, MetricCollection

from yggdrasil.core.dtypes.base import TensorDataClass
from yggdrasil.core.dtypes.metrics import MetricInput
from yggdrasil.utils.logging import init_logger

log = init_logger(__name__)


def _dataclass_asdict_nonrecursive(obj: object) -> dict[str, Any]:
    # TODO: Take this logic out of here
    # Probably, as native method of dataclasses
    if not is_dataclass(obj):
        msg = f"{obj} is not a dataclass"
        raise TypeError(msg)
    res = {}
    for field in fields(obj):
        if (value := getattr(obj, field.name)) is not None:
            res[field.name] = value
    return res


class MetricCallback(L.Callback):
    def __init__(
        self,
        train_metrics: MetricCollection | None = None,
        val_metrics: MetricCollection | None = None,
        test_metrics: MetricCollection | None = None,
    ) -> None:
        if train_metrics is not None:
            self.train_metrics = train_metrics
        elif val_metrics is not None:
            self.train_metrics = val_metrics.clone(prefix="train_")
        elif test_metrics is not None:
            self.train_metrics = test_metrics.clone(prefix="train_")
        else:
            msg = "At least one of the metrics should be provided"
            raise ValueError(msg)

        self.val_metrics = val_metrics or self.train_metrics.clone(prefix="val_")
        self.test_metrics = test_metrics or self.train_metrics.clone(prefix="test_")

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        r"""
        Setup lightning module with relevant metrics. By doing this we
        move the metrics to the correct device and make them trackable.

        Args:
            trainer (L.Trainer): Lightning Trainer instance.
            pl_module (L.LightningModule): Lightning Module instance.
            stage (str): The stage of the training loop.
        """
        if stage == "fit":
            pl_module.train_metrics = self.train_metrics
            pl_module.val_metrics = self.val_metrics
        elif stage == "validate":
            pl_module.val_metrics = self.val_metrics
        elif stage == "test":
            pl_module.test_metrics = self.test_metrics

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: TensorDataClass,
        batch_idx: int,
    ) -> None:
        r"""
        Log metrics for the given batch during the training phase.

        Args:
            trainer (L.Trainer): Lightning Trainer instance.
            pl_module (L.LightningModule): Lightning Module instance.
            outputs (torch.Tensor | Mapping[str, Any] | None): Output of the step.
            batch (TensorDataClass): Batch data.
            batch_idx (int): Batch index in case of distributed testing.
        """
        self._log_metrics(pl_module, outputs, batch, pl_module.train_metrics)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: TensorDataClass,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""
        Log metrics for the given batch during the validation phase.

        Args:
            trainer (L.Trainer): Lightning Trainer instance.
            pl_module (L.LightningModule): Lightning Module instance.
            outputs (torch.Tensor | Mapping[str, Any] | None): Output of the step.
            batch (TensorDataClass): Batch data.
            batch_idx (int): Batch index in case of distributed testing.
            dataloader_idx (int, optional): Dataloader index in case of multiple data.
                Defaults to 0.
        """
        self._log_metrics(pl_module, outputs, batch, pl_module.val_metrics)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: TensorDataClass,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""
        Log metrics for the given batch during the testing phase.

        Args:
            trainer (L.Trainer): Lightning Trainer instance.
            pl_module (L.LightningModule): Lightning Module instance.
            outputs (torch.Tensor | Mapping[str, Any] | None): Output of the step.
            batch (TensorDataClass): Batch data.
            batch_idx (int): Batch index in case of distributed testing.
            dataloader_idx (int, optional): Dataloader index in case of multiple data.
                Defaults to 0.
        """
        self._log_metrics(pl_module, outputs, batch, pl_module.test_metrics)

    def _log_metrics(
        self,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: TensorDataClass,
        metrics: MetricCollection,
    ) -> None:
        r"""
        Log metrics for the given batch.

        Args:
            pl_module (L.LightningModule): Lightning Module instance.
            outputs (torch.Tensor | Mapping[str, Any] | None): Output of the step.
            batch (TensorDataClass): Batch data.
            metrics (MetricCollection): Collection of metrics to log.

        Raises:
            TypeError: If the outputs are not a Mapping.
            KeyError: If the outputs do not contain 'metric_input'.
            TypeError: If the 'metric_input' is not an instance of MetricInput.
        """
        if not isinstance(outputs, Mapping):
            msg = f"Outpu expected Mapping, got {type(outputs)}"
            raise TypeError(msg)
        if "metric_input" not in outputs:
            msg = f"Expected 'metric_input' in outputs, got {outputs.keys()}"
            raise KeyError(msg)

        inp: MetricInput = outputs["metric_input"]
        if not isinstance(inp, MetricInput):
            msg = f"Expected MetricInput, got {type(inp)}"
            raise TypeError(msg)
        metrics.update(**_dataclass_asdict_nonrecursive(inp))
        pl_module.log_dict(metrics, on_step=False, on_epoch=True)


class MaxMetricCallback(L.Callback):
    """
    Callback for logging the maximum value of the metrics.

    .. important::
        By default the callback logs the maximum value of the metrics
        during validation. However, the behavior can be reconsidered
        in the future.

    .. todo::
        - Reconsider the stage of the logging the maximum value of the metrics.
        - Come up with more generic metrics definition.
    """

    def __init__(self, metric_names: list[str]) -> None:
        """
        Initialize the callback.

        Args:
            metric_names (list[str]): List of metric names to log the maximum value.
        """
        self.max_metrics = nn.ModuleDict(
            {f"max_{name}": MaxMetric() for name in metric_names}
        )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """
        Log the maximum value of the metrics at the end of the validation epoch

        .. note::
            At the end of the validation epoch, we also can log training metrics.

        Args:
            trainer (L.Trainer): Lightning Trainer instance.
            pl_module (L.LightningModule): Lightning Module instance.
        """
        for metric_name in self.max_metrics:
            # Retrieve the metric value from the callback_metrics
            metric_value = trainer.callback_metrics.get(metric_name[len("max_") :])

            if metric_value is not None:
                # According to the execution order validation epoch runs
                # after the training epoch. So, at the end of the validation
                # `trainer.callback_metrics` contains the metrics from the both.
                # Som this callback can also accumulate the max of training metrics.
                self.max_metrics[metric_name].update(metric_value)
            else:
                log.warning(f"Metric {metric_name} not found in callback_metrics")
        pl_module.log_dict(
            {name: value.compute() for name, value in self.max_metrics.items()},
            sync_dist=True,
        )
