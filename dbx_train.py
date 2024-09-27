# Databricks notebook source
# MAGIC %pip install .

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

from hydra_zen import  launch

from project.train import run, RunCfg, register_config

# COMMAND ----------

register_config(RunCfg)

# COMMAND ----------

overrides = ["datamodule.dataset.num_features=128", "model.net.input_dim=128"]

# COMMAND ----------

job = launch(RunCfg, run, overrides=overrides)

# COMMAND ----------


