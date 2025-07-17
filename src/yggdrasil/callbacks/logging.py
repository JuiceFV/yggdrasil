import os
import shutil
import tempfile
import time
from collections.abc import Mapping
from typing import Any

import lightning as L
import torch
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.model_summary.model_summary import ModelSummary

from yggdrasil.utils.mlflow import MLFlowLoggerCheckpointer


class MLFlowModelRegistryHook(L.Callback):
    """
    Hack to make instance of `trainer` and `pl_module` available to
    MLFlowLoggerCheckpointer. Required for the checkpointer to be able to save
    models to MLFlow model registry and not as just artifacts to the experiment run
    """

    def __init__(self) -> None:
        super().__init__()

    @rank_zero_only
    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLoggerCheckpointer):
                logger.trainer = trainer
                logger.module = pl_module


class SummaryLogger(L.Callback):
    """
    Callback saving txt model description to MLFlow experiment as artifact
    """

    def __init__(self, max_depth: int = -1) -> None:
        self.max_depth = max_depth

    @rank_zero_only
    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if isinstance(trainer.model, L.LightningModule):
            summary = str(ModelSummary(trainer.model, max_depth=self.max_depth))
        else:
            msg = "trainer.model is not of type LightningModule"
            raise TypeError(msg)

        tempdir = tempfile.mkdtemp()
        try:
            summary_file = os.path.join("file:/", tempdir, "model_summary.txt")
            with open(summary_file, "w") as f:
                f.write(summary)

            for logger in trainer.loggers:
                if isinstance(logger, MLFlowLogger):
                    logger.experiment.log_artifact(logger.run_id, local_path=summary_file)
        finally:
            shutil.rmtree(tempdir)


class TimingCallback(L.Callback):
    """
    Callback measuring and logging the time taken for each train step and epoch
    """

    @rank_zero_only
    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.epoch_start_time = time.time()

    @rank_zero_only
    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.batch_start_time = time.time()

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        step_time = time.time() - self.batch_start_time
        if trainer.logger:
            trainer.logger.log_metrics({"step_time": step_time}, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, *args: Any) -> None:
        epoch_time = time.time() - self.epoch_start_time
        if trainer.logger:
            trainer.logger.log_metrics({"epoch_time": epoch_time}, step=trainer.global_step)


class StepLoggingCallback(L.Callback):
    r"""
    Callback logging step related extra measurements to the loggers.

    .. note::
        Some loggers demand the step to be logged explicitly. For example,
        :class:`~lightning.loggers.mlflow.MLFlowLogger` requires the global step
        to be logged explicitly.

    .. note::
        It's recommended to log only the mandatory metrics here in purpose of
        reducing the amount of logged data. For example, the loss is logged.
    """

    def _log_metrics(
        self,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        prefix: str,
    ) -> None:
        r"""
        Log metrics to the loggers.

        Args:
            pl_module (L.LightningModule): Trainer module.
            outputs (torch.Tensor | Mapping[str, Any] | None): Output of a step.
            prefix (str): Stage identifier.

        Raises:
            ValueError: If the output dictionary does not contain the 'loss' key.
        """
        # NOTE: Assume that the loss is the only mandatory metric, otherwise
        # reconsider the implementation
        outputs = outputs if isinstance(outputs, Mapping) else {"loss": outputs}
        if "loss" not in outputs:
            msg = "Output dictionary must contain 'loss' key"
            raise ValueError(msg)

        pl_module.log(
            f"{prefix}_loss",
            outputs["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        pl_module.log("global_step", pl_module.global_step, on_step=True, on_epoch=False)

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        r"""Log step related metrics after the training batch.

        .. note::
            The callback is not restricted by model or input data types.
            Therefore, batch data can be of any type. And the callback
            could be used with any model, even though it's not instantiated from
            :class:`~yggdrasil.modules.base.BaseModule`.

        Args:
            trainer (L.Trainer): Lightning Trainer instance.
            pl_module (L.LightningModule): Lightning Module instance.
            outputs (torch.Tensor | Mapping[str, Any] | None): Output of the step.
            batch (Any): Batch data.
            batch_idx (int): Batch index in case of distributed training.
        """
        self._log_metrics(pl_module, outputs, prefix="train")

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""Log step related metrics after the validation batch.

        .. note::
            The callback is not restricted by model or input data types.
            Therefore, batch data can be of any type. And the callback
            could be used with any model, even though it's not instantiated from
            :class:`~yggdrasil.modules.base.BaseModule`.

        Args:
            trainer (L.Trainer): Lightning Trainer instance.
            pl_module (L.LightningModule): Lightning Module instance.
            outputs (torch.Tensor | Mapping[str, Any] | None): Output of the step.
            batch (Any): Batch data.
            batch_idx (int): Batch index in case of distributed validation.
            dataloader_idx (int, optional): Dataloader index in case of multiple data.
                Defaults to 0.
        """
        self._log_metrics(pl_module, outputs, prefix="val")

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        r"""Log step related metrics after the test batch.

        .. note::
            The callback is not restricted by model or input data types.
            Therefore, batch data can be of any type. And the callback
            could be used with any model, even though it's not instantiated from
            :class:`~yggdrasil.modules.base.BaseModule`.

        Args:
            trainer (L.Trainer): Lightning Trainer instance.
            pl_module (L.LightningModule): Lightning Module instance.
            outputs (torch.Tensor | Mapping[str, Any] | None): Output of the step.
            batch (Any): Batch data.
            batch_idx (int): Batch index in case of distributed validation.
            dataloader_idx (int, optional): Dataloader index in case of multiple data.
                Defaults to 0.
        """
        self._log_metrics(pl_module, outputs, prefix="test")
