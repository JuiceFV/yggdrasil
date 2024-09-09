from hydra_zen import make_config
from omegaconf import MISSING

from project.utils.mlflow import MLFlowLoggerCheckpointer

from .callbacks import MLFlowCallbacksCfg
from .utils import ZENSTORE, fbuilds

logging_store = ZENSTORE(group="loggers", package="_global_")

MLFlowLogCfg = fbuilds(
    MLFlowLoggerCheckpointer,
    experiment_name=MISSING,
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
