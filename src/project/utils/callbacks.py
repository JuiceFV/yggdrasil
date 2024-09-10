import os
import shutil
import tempfile
import time
from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.utilities.model_summary import ModelSummary
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from project.utils import init_logger

from .mlflow import MLFlowLoggerCheckpointer

log = init_logger(__name__)


class MLFlowModelRegistryHook(Callback):
    def __init__(self) -> None:
        super().__init__()

    @rank_zero_only
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLoggerCheckpointer):
                logger.trainer = trainer
                logger.module = pl_module


class SummaryLogger(Callback):
    def __init__(self, max_depth: int = -1) -> None:
        self.max_depth = max_depth

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        summary = str(ModelSummary(trainer.model, max_depth=self.max_depth))

        tempdir = tempfile.mkdtemp()
        try:
            summary_file = os.path.join("file:/", tempdir, "model_summary.txt")
            with open(summary_file, "w") as f:
                f.write(summary)

            for logger in trainer.loggers:
                if isinstance(logger, MLFlowLogger):
                    logger.experiment.log_artifact(
                        logger.run_id, local_path=summary_file
                    )
        finally:
            shutil.rmtree(tempdir)


class TimingCallback(Callback):
    @rank_zero_only
    def on_train_epoch_start(self, *args: Any) -> None:
        self.epoch_start_time = time.time()

    @rank_zero_only
    def on_train_batch_start(self, *args: Any) -> None:
        self.batch_start_time = time.time()

    @rank_zero_only
    def on_train_batch_end(self, trainer: Trainer, *args: Any) -> None:
        step_time = time.time() - self.batch_start_time
        trainer.logger.log_metrics({"step_time": step_time}, step=trainer.global_step)

    @rank_zero_only
    def on_train_epoch_end(self, trainer: Trainer, *args: Any) -> None:
        epoch_time = time.time() - self.epoch_start_time
        trainer.logger.log_metrics({"epoch_time": epoch_time}, step=trainer.global_step)
