# Databricks notebook source
# MAGIC !cd .. && source scripts/install_on_dbx.sh

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# fixes issue with editable package discovery on dbx
from site import addsitepackages

addsitepackages(None)

# COMMAND ----------

from project.utils.dbx import setup_loggin_for_dbx_notebook

log = setup_loggin_for_dbx_notebook("dbx_train")

# COMMAND ----------

from hydra_zen import launch

from project.train import ZENSTORE, RunCfg, register_config, run

# COMMAND ----------

register_config(RunCfg, ZENSTORE)

# COMMAND ----------

default_overrides = (
    "datamodule.dataset.num_features=128 model.net.input_dim=128 "
    "paths.log_dir=/tmp/logs"
)

dbx_params = dbutils.notebook.entry_point.getCurrentBindings()  # noqa: F821
overrides = dbx_params.get("overrides", default_overrides).split()
log.info(f"Detected overrides: {overrides}")

# COMMAND ----------

job = launch(RunCfg, run, version_base="1.3", overrides=overrides)
