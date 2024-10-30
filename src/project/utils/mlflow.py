import os
from typing import TYPE_CHECKING, Any

import mlflow
from lightning.fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from mlflow.system_metrics.system_metrics_monitor import SystemMetricsMonitor
from mlflow.utils.autologging_utils import get_autologging_config

from project.utils import init_logger

if TYPE_CHECKING:
    from lightning import LightningModule, Trainer
    from mlflow.tracking.fluent import ActiveRun

log = init_logger(__name__)


class MLFlowLoggerCheckpointer(MLFlowLogger):
    """
    Extension of MLFlowLogger that logs model checkpoints to MLFlow model registry.
    It also extends the MLFlowLogger to log system metrics.
    """

    def __init__(
        self,
        experiment_name: str = "lightning_logs",
        run_name: str | None = None,
        tracking_uri: str | None = os.getenv("MLFLOW_TRACKING_URI"),
        tags: dict[str, Any] | None = None,
        save_dir: str | None = "./mlruns",
        prefix: str = "",
        artifact_location: str | None = None,
        run_id: str | None = None,
        custom_log_model: bool = True,
        register_model: bool = True,
        extra_files: list | None = None,
        log_system_metrics: bool = True,
    ):
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
            tags=tags,
            save_dir=save_dir,
            log_model=False,
            prefix=prefix,
            artifact_location=artifact_location,
            run_id=run_id,
        )

        self.custom_log_model = custom_log_model
        self.register_model = register_model

        self.extra_files = extra_files

        self.trainer: Trainer | None = None
        self.module: LightningModule | None = None

        self._active_run: ActiveRun | None = None
        self.log_system_metrics = log_system_metrics
        self._system_monitor: SystemMetricsMonitor | None = None
        self._existing_run = run_id is not None

    @property
    def system_monitor(self) -> SystemMetricsMonitor | None:
        if self.log_system_metrics and self._system_monitor is None:
            try:
                self._system_monitor = SystemMetricsMonitor(
                    self._run_id,
                    resume_logging=self._existing_run,
                )
                if self._system_monitor is not None:
                    self._system_monitor.start()
            except Exception:
                log.exception("Failed to start system metrics monitoring: %e")
                self.log_system_metrics = False

        return self._system_monitor

    @property
    def run_id(self) -> str | None:
        mlflow.set_tracking_uri(self.experiment.tracking_uri)
        _ = self.system_monitor
        return self._run_id

    @property
    def active_run(self) -> mlflow.ActiveRun:
        if self._active_run is None:
            # Hack to force MLFlow to 'know' about this run
            self._active_run = mlflow.start_run(
                self.run_id, log_system_metrics=False
            ).__enter__()

        return self._active_run

    def _register_model(self, name: str) -> None:
        if self.trainer:
            try:
                registered_model_name = get_autologging_config(
                    mlflow.pytorch.FLAVOR_NAME,
                    "registered_model_name",
                    None,
                )

                mlflow.pytorch.log_model(
                    pytorch_model=self.trainer.model.net,  # type: ignore
                    artifact_path=name,
                    registered_model_name=registered_model_name,
                    extra_files=self.extra_files,
                )

            except Exception:
                log.exception("Saving model to MLFLow failed with exception: %s")
        else:
            log.warning(
                "`trainer` reference not registered via `MLFlowModelRegistryHook`"
            )

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        """
        Called after model checkpoint callback saves a new checkpoint.
        """

        if self.custom_log_model:
            self.experiment.log_artifact(
                self.run_id, checkpoint_callback.best_model_path
            )

        if self.register_model:
            _ = self.active_run
            if checkpoint_callback.filename is not None:
                self._register_model(checkpoint_callback.filename)

    @rank_zero_only
    def finalize(self, status: str = "success") -> None:
        if self._active_run:
            self.active_run.__exit__(None, None, None)
        if self._system_monitor:
            self._system_monitor.finish()
        super().finalize(status)
