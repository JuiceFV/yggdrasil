import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession

from yggdrasil.core.dtypes.preprocessing.base import InputColumn


def query_original_table(session: SparkSession, table_identifier: str) -> DataFrame:
    r"""
    Query the original table from the Spark session.

    Args:
        session (SparkSession): Spark session to use for querying.
        table_identifier (str): Identifier of the table to query.

    Returns:
        DataFrame: DataFrame containing the queried table.
    """
    return session.read.table(table_identifier)


def get_distinct_keys(df: DataFrame, col_name: str) -> list[int]:
    r"""
    Get distinct keys from a column in the DataFrame.

    Example::

        df = spark.createDataFrame(
            [{"key": 1}, {"key": 2}, {"key": 3}, {"key": 1}]
        )
        keys = get_distinct_keys(df, "key")
        print(keys)  # Output: [1, 2, 3]

    Args:
        df (DataFrame): DataFrame to extract keys from.
        col_name (str): Name of the column containing the keys.

    Raises:
        ValueError: If any of the keys are not of type int.

    Returns:
        list[int]: Sorted list of distinct keys from the specified column.
    """
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
    r"""
    Hash the DataFrame and subsample it based on the provided sample range.

    .. important::
        The function will be modified s.t. cover as much splitting logic as possible.

    .. note::
        This function produces a new column :attr:`~yggdrasil.core.dtypes.preprocessing.InputColumn.SAMPLE_ID`
        containing the hash of the specified columns or all columns if none are specified. This logic retains
        a unique identifier but makes it consumable by PyTorch.

    .. important::
        The number of partitions defines how grainular the subsampling is. If you pass ``num_partitions=2``,
        then the distribution within a partition will be less sharp than if you pass ``num_partitions=100``.
        You can consider this as number of rectangles in the Riemann sum approximation of the distribution.

    Example::

        df = spark.createDataFrame(
            [
                {"id": "id-str1", "feature1": 1.0, "feature2": 2.0},
                {"id": "id-str2", "feature1": 3.0, "feature2": 4.0},
                ...,
                {"id": "id-str100", "feature1": 0.5, "feature2": 11.0},
            ]
        )
        print(df.count())  # Output: 100
        print(df.columns)  # Output: ['id', 'feature1', 'feature2']

        train_df = hash_and_subsample(df, sample_range=(0.0, 70.0), hash_cols=["id"])
        print(train_df.count())  # Output: 70
        print(train_df.columns)  # Output: ['id', 'feature1', 'feature2', 'sample_id']

    Args:
        df (DataFrame): DataFrame to be processed.
        sample_range (tuple[float, float] | None, optional): Range of the sample to keep, specified as a tuple
            of two floats between 0.0 and 100.0. If None, no subsampling is applied. Defaults to None.
        hash_cols (list[str] | None, optional): Columns to use for hashing.If None, all columns will be used.
            Defaults to None.
        num_partitions (int, optional): Number of partitions to use for subsampling. Defaults to 100.

    Raises:
        ValueError: If the sample range is not between 0.0 and 100.0.

    Returns:
        DataFrame: DataFrame with hashed sample IDs and subsampled based on the range.
    """
    if sample_range and not (0.0 <= sample_range[0] <= sample_range[1] <= 100.0):
        msg = f"Sample range must be between 0.0 and 100.0; got {sample_range}"
        raise ValueError(msg)

    hash_key = F.concat(*(hash_cols or df.columns))
    df = df.withColumn(InputColumn.SAMPLE_ID, F.crc32(hash_key))
    if sample_range:
        df = df.withColumn("partition_id", F.col(InputColumn.SAMPLE_ID) % num_partitions)
        lower_partition = int(sample_range[0] / 100.0 * num_partitions)
        upper_partition = int(sample_range[1] / 100.0 * num_partitions)
        df = df.filter((F.col("partition_id") >= lower_partition) & (F.col("partition_id") < upper_partition))
        df = df.drop("partition_id")
    return df
