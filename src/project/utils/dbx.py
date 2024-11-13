import logging
import os

import hydra
import mlflow

from project.utils import init_logger

log = init_logger(__name__)


def setup_loggin_for_dbx_notebook(log_name: str) -> logging.Logger:
    log = init_logger(log_name)

    log_config = logging.config.dictConfig(
        {
            "version": 1,
            "loggers": {
                "py4j": {"level": "ERROR"},
                "urllib3.connectionpool": {"level": "ERROR"},
            },
        }
    )
    hydra.core.utils.configure_log(log_config)
    logging.captureWarnings(True)

    return log


def _retrive_mlflow_credentials() -> dict | None:
    creds = None

    try:
        from mlflow.utils.databricks_utils import get_databricks_env_vars

        creds = get_databricks_env_vars("databricks")
    except mlflow.MlflowException as e:
        log.warning(f"Failed to retrieve Databricks credentials: {e}")

    return creds


def propagate_credentials() -> dict | None:
    """
    Retrieves Databricks credentials if available and propagates them to the ray
    remote jobs which are executed in the separete python processes and don't have
    a direct access to them.
    If launched locally, the script will also propagate your own local credentials
    specified in the `~/.databrickscfg` and will connect to mlflow server on DBX.
    """
    do_propagate = int(os.environ.get("PROPAGATE_DBX_CREDS", "0"))
    if do_propagate:
        return _retrive_mlflow_credentials()
    return None
