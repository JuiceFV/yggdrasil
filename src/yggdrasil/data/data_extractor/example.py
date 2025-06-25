import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from yggdrasil.core.dtypes.dataset import ParquetDataset
from yggdrasil.core.dtypes.preprocessing.base import InputColumn
from yggdrasil.data.data_extractor.base import DataExtractor
from yggdrasil.data.etl import spark


def select_relevant_columns(df: DataFrame) -> DataFrame:
    select_cols = [
        F.col(InputColumn.SAMPLE_ID).cast(T.LongType()),
        F.col(InputColumn.FEATURES).cast(T.ArrayType(T.FloatType())),
        F.col(f"{InputColumn.FEATURES}_presence").cast(T.ArrayType(T.BooleanType())),
        F.col(InputColumn.TARGET).cast(T.FloatType()),
    ]
    return df.select(*select_cols)


class ExampleDataExtractor(DataExtractor):
    r"""
    Example data extractor for testing purposes.
    """

    def query_data(
        self, table_identifier: str, sample_range: tuple[float, float]
    ) -> ParquetDataset:
        session = spark.init.get_spark_session()
        df = spark.extract.query_original_table(session, table_identifier)
        df = spark.extract.hash_and_subsample(
            df, sample_range=sample_range, hash_cols=["uid"]
        )
        df = spark.transform.make_sparse2dense(
            df,
            "features",
            possible_keys=spark.extract.get_distinct_keys(df, "features"),
        )
        df = select_relevant_columns(df)
        dataset_url = spark.load.upload_as_parquet(session, df)
        return ParquetDataset(dataset_url=dataset_url)
