import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import DataFrame, SQLContext
from pyspark.sql.functions import asc

from tests.data.sql_test_base import SQLTestBase
from yggdrasil.core.dtypes.preprocessing.base import InputColumn
from yggdrasil.data.data_extractor.example import ExampleDataExtractor

if TYPE_CHECKING:
    from yggdrasil.core.dtypes.dataset import ParquetDataset

logger = logging.getLogger(__name__)


def gen_pandas_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "uid": ["0", "1", "2", "3"],
            "features": [{0: 1}, {1: 1}, {4: 1}, {5: 1}],
            "target": [0, 1, 0, 1],
        }
    )
    return df


def gen_example_data(ctx: SQLContext, table_name: str) -> None:
    pdf = gen_pandas_df()
    df: DataFrame = ctx.createDataFrame(pdf)
    logger.info("Created dataframe")
    df.show()
    df.createOrReplaceTempView(table_name)


class TestExampleData(SQLTestBase):
    def setUp(self) -> None:
        super().setUp()
        logging.getLogger(__name__).setLevel(logging.INFO)
        self.table_identifier = "test_table"
        logger.info(f"Table name is {self.table_identifier}")

    def gen_data(self) -> None:
        gen_example_data(self.sqlCtx, table_name=self.table_identifier)

    def read_data(self) -> DataFrame:
        data_extractor = ExampleDataExtractor()
        dataset: ParquetDataset = data_extractor.query_data(
            table_identifier=self.table_identifier, sample_range=(0.0, 100.0)
        )
        df = self.sqlCtx.read.parquet(dataset.dataset_url)
        df = df.orderBy(asc(InputColumn.SAMPLE_ID))
        logger.info("Read parquet dataframe: ")
        df.show()
        return df

    @pytest.mark.serial
    def test_query_data(self) -> None:
        self.gen_data()
        df = self.read_data()
        df = df.toPandas()
        self.assert_all(df)
        logger.info("Example data extraction seems fine")

    def assert_all(self, df: pd.DataFrame) -> None:
        features_presence = np.array(
            [
                [False, False, True, False],
                [False, False, False, True],
                [False, True, False, False],
                [True, False, False, False],
            ],
            dtype=bool,
        )
        self.assert_eq(pd.Series(df["features_presence"]), features_presence)
        features = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self.assert_eq_with_presence(pd.Series(df["features"]), features_presence, features)
