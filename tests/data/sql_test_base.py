import logging
import os
import shutil

import lightning as L
import numpy as np
import pandas as pd
from sparktestingbase.sqltestcase import SQLTestCase

SPARK_CONFIG = {
    "spark.app.name": "YggdrasilTest",
    "spark.sql.session.timeZone": "UTC",
    # use local host
    "spark.driver.host": "127.0.0.1",
    # use as many worker threads as possible on machine
    "spark.master": "local[*]",
    # default local warehouse for Hive
    "spark.sql.warehouse.dir": os.path.abspath("spark-warehouse"),
    # Set shuffle partitions to a low number, e.g. <= cores * 2 to speed
    # things up, otherwise the tests will use the default 200 partitions
    # and it will take a lot more time to complete
    "spark.sql.shuffle.partitions": "12",
    # Same effect as builder.enableHiveSupport() [useful for test framework]
    "spark.sql.catalogImplementation": "hive",
}

HIVE_METASTORE = "metastore_db"
TEST_CLASS_PTR = 0

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class SQLTestBase(SQLTestCase):
    def getConf(self) -> dict[str, str]:  # noqa: N802
        return SPARK_CONFIG

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        global TEST_CLASS_PTR  # noqa: PLW0603
        cls.test_class_seed = TEST_CLASS_PTR
        logger.info(f"Allocating seed {cls.test_class_seed} to {cls.__name__}")
        TEST_CLASS_PTR += 1

    def setUp(self) -> None:
        super().setUp()
        assert not os.path.isdir(HIVE_METASTORE), f"Delete {HIVE_METASTORE} first!"

        L.seed_everything(self.test_class_seed)
        logging.basicConfig()

    def assert_eq(self, pd_series: pd.Series, arr: np.ndarray) -> None:
        series_as_arr = np.array(pd_series.tolist())
        np.testing.assert_equal(series_as_arr, arr)

    def assert_all_close(self, pd_series: pd.Series, arr: np.ndarray) -> None:
        series_as_arr = np.array(pd_series.tolist())
        np.testing.assert_allclose(series_as_arr, arr)

    def assert_eq_with_presence(self, pd_series: pd.Series, presence: np.ndarray, arr: np.ndarray) -> None:
        series_as_arr = np.array(pd_series.tolist())
        present_sa = series_as_arr[presence]
        present_arr = arr[presence]
        np.testing.assert_equal(present_sa, present_arr)

    def tearDown(self) -> None:
        super().tearDown()

        if os.path.isdir(HIVE_METASTORE):
            shutil.rmtree(HIVE_METASTORE)
