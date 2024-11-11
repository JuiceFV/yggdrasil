# Databricks notebook source
!cd .. && source scripts/install_on_dbx.sh

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# enables autoreloading of all imporeted modules without
# relaunching notebook. Should be executed just once
%load_ext autoreload
%autoreload 2

# COMMAND ----------

# fixes issue with editable package discovery on dbx
from site import addsitepackages

addsitepackages(None)

# COMMAND ----------

from project.utils.dbx import setup_loggin_for_dbx_notebook

log = setup_loggin_for_dbx_notebook("dbx_train")

# COMMAND ----------

from hydra_zen import launch

from project.train import RunCfg, register_config, run, ZENSTORE

# COMMAND ----------

register_config(RunCfg, ZENSTORE)

# COMMAND ----------

overrides = ["datamodule.dataset.num_features=128", "model.net.input_dim=128"]

job = launch(RunCfg, run, version_base="1.3", overrides=overrides)

# COMMAND ----------


