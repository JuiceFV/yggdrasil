yggdrasil.data
==============

.. automodule:: yggdrasil.data

.. contents:: yggdrasil.data
    :depth: 2
    :local:
    :backlinks: top


ETL
---

Spark
~~~~~

.. currentmodule:: yggdrasil.data.etl.spark

.. autosummary::
    :toctree: ../generated
    :nosignatures:

    init.get_spark_session
    extract.query_original_table
    extract.get_distinct_keys
    extract.hash_and_subsample
    transform.make_sparse_vector
    transform.make_sparse2dense
    transform.stratified_sampling_norm_spec
    transform.identify_normalization_params
    load.get_table_url
    load.upload_as_parquet
