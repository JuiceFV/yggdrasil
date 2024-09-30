# Databricks notebook source
# MAGIC %mkdir /tmp/pkg_build && cd .. && cp -R * /tmp/pkg_build 2>/dev/null
# MAGIC %cd /tmp/pkg_build/
# MAGIC %pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from project.utils import init_logger

log = init_logger("dbx_train")

# COMMAND ----------

from hydra_zen import  launch

from project.train import run, RunCfg, register_config

# COMMAND ----------

register_config(RunCfg)

# COMMAND ----------

dbx_args = dict(dbutils.notebook.entry_point.getCurrentBindings())
default_overrides = "datamodule.dataset.num_features=128 model.net.input_dim=128"

overrides = dbx_args.get("overrides", default_overrides).split()
log.info(f"Detected overrides: {overrides}")

# COMMAND ----------

job = launch(RunCfg, run, overrides=overrides)

# COMMAND ----------


