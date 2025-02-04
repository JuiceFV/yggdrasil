import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame, SparkSession

from project.core.dtypes.options import PreprocessingOptions
from project.core.dtypes.parameters import NormalizationParams
from project.data.etl.spark.extract import query_original_table
from project.preprocessing.normalization import infer_normalization


def make_sparse_vector(
    df: DataFrame,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> tuple[DataFrame, dict[int, str]]:
    if include and exclude:
        msg = "`include` and `exclude` are mutually exclusive"
        raise ValueError(msg)
    select_cols = (
        list(set(df.columns) & set(include))
        if include
        else list(set(df.columns) - set(exclude or []))
    )
    fid2fname = dict(enumerate(select_cols))
    col_pairs = list(
        sum(
            [
                (F.lit(fid).cast("int"), F.col(f"`{fname}`").cast("double"))
                for fid, fname in fid2fname.items()
            ],
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
    df = query_original_table(session, table_name)
    df = stratified_sampling_norm_spec(
        df, col_name, preprocessing_options.nsamples, seed
    )
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
