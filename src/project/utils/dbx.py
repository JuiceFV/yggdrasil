import logging

import hydra

from project.utils import init_logger


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
