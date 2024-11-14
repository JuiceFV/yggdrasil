import os

from hydra_zen import make_config
from omegaconf import MISSING

from project.configs.callbacks.base import MLFlowCallbacksCfg
from project.utils.logging import init_logger
from project.utils.mlflow import MLFlowLoggerCheckpointer

from .utils import ZENSTORE, fbuilds

logger = init_logger(__name__)

try:
    from aim.pytorch_lightning import AimLogger
except ImportError:
    AimLogger = None
    logger.warning(
        msg=("AimLogger is not available. All related config groups will be disabled.")
    )

logging_store = ZENSTORE(group="loggers", package="_global_")

MLFlowLogCfg = fbuilds(
    MLFlowLoggerCheckpointer,
    experiment_name=MISSING,  # required argument to be provided
    run_name=None,
    tracking_uri="${oc.env:MLFLOW_TRACKING_URI,file:${paths.log_dir}/mlflow/mlruns}",
    tags=None,
    save_dir="./mlruns",
    prefix="",
    artifact_location=None,
    custom_log_model=True,
    register_model=True,
    extra_files=None,
)

MLFlowLogCheckCfg = make_config(
    loggers={"mlflow": MLFlowLogCfg},
    callbacks=MLFlowCallbacksCfg,
)

logging_store(MLFlowLogCheckCfg, name="mlflow")


if AimLogger is not None:
    AimLoggerConf = fbuilds(
        AimLogger, experiment=MISSING, repo=os.environ.get("AIM_ENDPOINT", None)
    )
    aim_logger_conf = make_config(
        loggers={"aim": AimLoggerConf},
    )
    logging_store(aim_logger_conf, name="aim")
