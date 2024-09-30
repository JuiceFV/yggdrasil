# Databricks notebook source
! [ -d "/tmp/pkg_build" ] && rm -r "/tmp/pkg_build"
%mkdir /tmp/pkg_build && cd .. && cp -R * /tmp/pkg_build 2>/dev/null
%cd /tmp/pkg_build/
%pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import logging
import hydra

from project.utils import init_logger

log = init_logger("dbx_train")

log_config = logging.config.dictConfig(
    {
        "version": 1,
        "loggers": {
            "py4j": {"level": "WARNING"},
            "urllib3.connectionpool": {"level": "ERROR"},
        },
    }
)
hydra.core.utils.configure_log(log_config)
logging.captureWarnings(True)

# COMMAND ----------

from hydra_zen import  launch

from project.train import run, RunCfg, register_config

# COMMAND ----------

register_config(RunCfg)

# COMMAND ----------

default_overrides = "datamodule.dataset.num_features=128 model.net.input_dim=128 paths.log_dir=/dbfs/FileStore/tmp/project/logs"

dbx_params = dbutils.notebook.entry_point.getCurrentBindings()
overrides = dbx_params.get("overrides", default_overrides).split()
log.info(f"Detected overrides: {overrides}")

# COMMAND ----------

job = launch(RunCfg, run, version_base="1.3", overrides=overrides)

# COMMAND ----------


