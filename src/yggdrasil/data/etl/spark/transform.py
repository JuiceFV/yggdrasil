import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame, SparkSession

from yggdrasil.core.dtypes.options import PreprocessingOptions
from yggdrasil.core.dtypes.parameters import NormalizationParams
from yggdrasil.data.etl.spark.extract import query_original_table
from yggdrasil.preprocessing.normalization import infer_normalization


def make_sparse_vector(
    df: DataFrame,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[DataFrame, dict[int, str]]:
    r"""
    Convert selected columns of a DataFrame into a sparse vector format.

    .. note::
        Final column will be named `sparse_vector` and will contain a map
        of column indices to their values. In other words, it will be a
        dictionary-like structure where keys feature IDs (integers)
        and values are the corresponding feature values (floats).

    Example::

        df.show()  # Before transformation:

        +---+---+---+
        | a | b | c |
        +---+---+---+
        | 1 | 2 | 3 |
        | 4 | 5 | 6 |
        +---+---+---+

        new_df, fid2fname = make_sparse_vector(df)
        new_df.show()  # After transformation:

        +------------------+
        | sparse_vector    |
        +------------------+
        | {0: 1.0, 2: 3.0} |
        | {0: 4.0, 2: 6.0} |
        +------------------+

        print(fid2fname)  # Output: {0: 'a', 1: 'b', 2: 'c'}

    Args:
        df (DataFrame): DataFrame to be transformed.
        include (list[str] | None, optional): List of columns to include in the sparse vector.
            If None, all columns not in `exclude` will be included. Defaults to None.
        exclude (list[str] | None, optional): List of columns to exclude from the sparse vector.
            If None, no columns will be excluded. Defaults to None.

    Raises:
        ValueError: If both `include` and `exclude` are provided, as they are mutually exclusive.

    Returns:
        tuple[DataFrame, dict[int, str]]: New DataFrame with a `sparse_vector` column,
            and a dictionary mapping feature IDs to their corresponding column names.
    """
    if include and exclude:
        msg = "`include` and `exclude` are mutually exclusive"
        raise ValueError(msg)
    select_cols = list(set(df.columns) & set(include)) if include else list(set(df.columns) - set(exclude or []))
    fid2fname = dict(enumerate(select_cols))
    col_pairs = list(
        sum(
            [(F.lit(fid).cast("int"), F.col(f"`{fname}`").cast("double")) for fid, fname in fid2fname.items()],
            (),
        )
    )
    df = df.withColumn("sparse_vector", F.create_map(*col_pairs)).drop(*select_cols)
    return df, fid2fname


def make_sparse2dense(
    df: DataFrame,
    col_name: str,
    possible_keys: list[int],
) -> DataFrame:
    r"""
    Convert a sparse vector column in a DataFrame to a dense format.
    The sparse vector is expected to be a map with integer keys and float values.
    The dense format will consist of two arrays: one indicating the presence
    of each key and another containing the corresponding values.

    Example::

        df.show() # Before transformation:

        # Output:
        # +---------------------+
        # | sparse_vector       |
        # +---------------------+
        # | {0: 1.0, 2: 3.0}    |
        # +---------------------+
        # | {1: 2.0, 3: 4.0}    |
        # +---------------------+

        make_sparse2dense(df, "sparse_vector", [0, 1, 2, 3]).show() # After transformation:

        # Output:
        # +---------------------+------------------------+---------------------+
        # | sparse_vector       | sparse_vector_presence | sparse_vector_dense |
        # +---------------------+------------------------+---------------------+
        # | {0: 1.0, 2: 3.0}    | [True, False, True]    | [1.0, 0.0, 3.0]     |
        # +---------------------+------------------------+---------------------+
        # | {1: 2.0, 3: 4.0}    | [False, True, True]    | [0.0, 2.0, 4.0]     |
        # +---------------------+------------------------+---------------------+

    Args:
        df (DataFrame): DataFrame containing the sparse vector column.
        col_name (str): Name of the column containing the sparse vector.
        possible_keys (list[int]): List of possible keys that should be present in the sparse vector.

    Raises:
        TypeError: If the column is not a dictionary type or if the keys are not integers.

    Returns:
        DataFrame: DataFrame with the sparse vector column transformed into a dense format.
    """
    output_type = T.StructType(
        [
            T.StructField("presence", T.ArrayType(T.BooleanType()), False),
            T.StructField("dense", T.ArrayType(T.FloatType()), False),
        ]
    )

    def map_sparse2dense(
        map_col: dict[int, float],
    ) -> tuple[list[bool], list[float]]:
        if not isinstance(map_col, dict):
            msg = f"{map_col} has type {type(map_col)} and is not a dict."
            raise TypeError(msg)
        presence = [False] * len(possible_keys)
        dense = [0.0] * len(possible_keys)
        for i, key in enumerate(possible_keys):
            val = map_col.get(key)
            if val is not None:
                presence[i] = True
                dense[i] = float(val)
        return presence, dense

    sparse2dense_udf = F.udf(map_sparse2dense, output_type)
    df = df.withColumn(col_name, sparse2dense_udf(col_name))
    df = df.withColumn(f"{col_name}_presence", F.col(f"{col_name}.presence"))
    df = df.withColumn(col_name, F.col(f"{col_name}.dense"))
    return df


def stratified_sampling_norm_spec(
    df: DataFrame,
    col_name: str,
    nsamples: int,
    seed: int | None = None,
) -> DataFrame:
    r"""
    Perform stratified sampling on a DataFrame to create a normalization specification.

    Example::

        df.show()

        # Output:
        # +-------------------------------------+
        # | sparse_vector                       |
        # +-------------------------------------+
        # | {0: 1.0, 1: 3.5, 2: 3.0}            |
        # | {0: 2.0, 1: 2.0, 2: 0.5, 3: 4.0}    |
        # +-------------------------------------+

        stratified_sampling_norm_spec(df, "sparse_vector", 2).show()

        # Output:
        # +---+------------------+
        # |fid| fvalues          |
        # +---+------------------+
        # | 0 | [1.0, 2.0]       |
        # | 1 | [3.5, 2.0]       |
        # | 2 | [3.0, 0.5]       |
        # | 3 | [4.0]            |
        # +---+------------------+

    Args:
        df (DataFrame): DataFrame to sample from.
        col_name (str): Name of the column containing the sparse vector.
        nsamples (int): Number of samples to take from each feature.
        seed (int | None, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        DataFrame: DataFrame containing the sampled features and their values.
    """
    if isinstance(df.schema[col_name].dataType, T.ArrayType):
        df = df.select(F.explode(F.col(col_name)).alias(col_name))

    df = df.select(F.explode(F.col(col_name)).alias("fid", "fvalue"))

    counts_df: DataFrame = df.groupBy("fid").count()
    fracs = {}
    for row in counts_df.collect():
        fracs[row["fid"]] = min(nsamples / row["count"], 1.0)

    df = df.sampleBy("fid", fractions=fracs, seed=seed)
    df = df.groupBy("fid").agg(F.collect_list("fvalue").alias("fvalues"))
    return df


def identify_normalization_params(
    session: SparkSession,
    table_name: str,
    col_name: str,
    preprocessing_options: PreprocessingOptions,
    seed: int | None = None,
) -> dict[int, NormalizationParams]:
    r"""
    Identify normalization parameters for a specified column in a Spark DataFrame.
    The dedicated column is expected to contain a sparse vector format. I.e. it could
    be either a ``map<int, float>`` or a ``list<map<int, float>>``.

    Args:
        session (SparkSession): Spark session to use for querying the table.
        table_name (str): Name of the table to query.
        col_name (str): Name of the column to identify normalization parameters for.
        preprocessing_options (PreprocessingOptions): Preprocessing options to use for normalization.
        seed (int | None, optional): Random seed for sampling. Defaults to None.

    Returns:
        dict[int, NormalizationParams]: Dictionary mapping feature IDs to their corresponding normalization parameters.
    """
    df = query_original_table(session, table_name)
    df = stratified_sampling_norm_spec(df, col_name, preprocessing_options.nsamples, seed)
    rows = df.collect()

    normalization_processor = infer_normalization(
        max_unique_enum_values=preprocessing_options.max_unique_enum_values,
        qunatile_size=preprocessing_options.quantile_size,
        quantile_k2_threshold=preprocessing_options.quantile_k2_threshold,
        skip_box_cox=preprocessing_options.skip_boxcox,
        skip_quantiles=preprocessing_options.skip_quantiles,
        feature_overrides=preprocessing_options.feature_overrides,
        allowed_features=preprocessing_options.allowed_features,
        assert_allowlist_feature_coverage=preprocessing_options.assert_allowlist_feature_coverage,
    )
    return normalization_processor(rows)
