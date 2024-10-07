# Databricks notebook source
# MAGIC !ln -s $(realpath ..) $(mktemp -d)
# MAGIC %pip install ..

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from project.utils.dbx import setup_loggin_for_dbx_notebook

log = setup_loggin_for_dbx_notebook("dbx_train")

# COMMAND ----------

from hydra_zen import launch

from project.train import RunCfg, register_config, run

# COMMAND ----------

register_config(RunCfg)

# COMMAND ----------

default_overrides = (
    "datamodule.dataset.num_features=128 model.net.input_dim=128 "
    "paths.log_dir=/dbfs/FileStore/tmp/project/logs"
)

dbx_params = dbutils.notebook.entry_point.getCurrentBindings()  # noqa: F821
overrides = dbx_params.get("overrides", default_overrides).split()
log.info(f"Detected overrides: {overrides}")

# COMMAND ----------

job = launch(RunCfg, run, version_base="1.3", overrides=overrides)

# COMMAND ----------
