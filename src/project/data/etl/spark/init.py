from pyspark.sql import SparkSession


def get_spark_session(config: dict[str, str] | None = None) -> SparkSession:
    spark = SparkSession.Builder().enableHiveSupport()
    if config is not None:
        for k, v in config.items():
            spark = spark.config(k, v)
    spark = spark.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark
