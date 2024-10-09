# Databricks notebook source
# MAGIC !cd .. && source scripts/install_on_dbx.sh

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# enables autoreloading of all imporeted modules without
# relaunching notebook. Should be executed just once
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# fixes issue with editable package discovery on dbx

import os
import sys

src_dir = os.path.realpath("../src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# COMMAND ----------

from project.utils.dbx import setup_loggin_for_dbx_notebook

log = setup_loggin_for_dbx_notebook("dbx_train")

# COMMAND ----------

from hydra_zen import launch

from project.train import RunCfg, register_config, run

# COMMAND ----------

register_config(RunCfg)

# COMMAND ----------

overrides = ["datamodule.dataset.num_features=128", "model.net.input_dim=128"]

job = launch(RunCfg, run, version_base="1.3", overrides=overrides)
