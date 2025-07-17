from pyspark.sql import SparkSession


def get_spark_session(config: dict[str, str] | None = None) -> SparkSession:
    r"""
    Create and return a Spark session with Hive support enabled.

    Args:
        config (dict[str, str] | None, optional): Configuration options for the Spark session.
            Defaults to None.

    Returns:
        SparkSession: A Spark session with Hive support enabled and configured according to the provided options.
    """
    spark = SparkSession.Builder().enableHiveSupport()
    if config is not None:
        for k, v in config.items():
            spark = spark.config(k, v)
    spark = spark.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
