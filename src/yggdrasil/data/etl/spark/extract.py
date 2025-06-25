import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from yggdrasil.core.dtypes.preprocessing.base import InputColumn


def query_original_table(session: SparkSession, table_identifier: str) -> DataFrame:
    return session.read.table(table_identifier)


def get_distinct_keys(df: DataFrame, col_name: str) -> list[int]:
    df = df.select(F.explode(F.map_keys(col_name)).alias("key"))
    keys = [row["key"] for row in df.distinct().collect()]
    if not all(isinstance(k, int) for k in keys):
        msg = f"Expected all keys to be of type int; got {keys}"
        raise ValueError(msg)
    return sorted(keys)


def hash_and_subsample(
    df: DataFrame,
    sample_range: tuple[float, float] | None = None,
    hash_cols: list[str] | None = None,
    num_partitions: int = 100,
) -> DataFrame:
    if sample_range and not (0.0 <= sample_range[0] <= sample_range[1] <= 100.0):
        msg = f"Sample range must be between 0.0 and 100.0; got {sample_range}"
        raise ValueError(msg)

    hash_key = F.concat(*(hash_cols or df.columns))
    df = df.withColumn(InputColumn.SAMPLE_ID, F.crc32(hash_key))
    if sample_range:
        df = df.withColumn(
            "partition_id", F.col(InputColumn.SAMPLE_ID) % num_partitions
        )
        lower_partition = int(sample_range[0] / 100.0 * num_partitions)
        upper_partition = int(sample_range[1] / 100.0 * num_partitions)
        df = df.filter(
            (F.col("partition_id") >= lower_partition)
            & (F.col("partition_id") < upper_partition)
        )
        df = df.drop("partition_id")
    return df
