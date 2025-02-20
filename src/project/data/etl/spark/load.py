import secrets
import string

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame, SparkSession

MAX_UPLOAD_PARQUET_TRIES = 10


def get_table_url(session: SparkSession, table_name: str) -> str:
    row = (
        session.sql(f"DESCRIBE FORMATTED {table_name}")
        .filter(F.col("col_name") == "Location")
        .select(F.col("data_type").cast(T.StringType()))
        .collect()
    )
    url = row[0]["data_type"]
    schema, path = str(url).split(":")
    return f"{schema}://{path}" if schema != "dbfs" else f"/{schema}/{path}"


def upload_as_parquet(session: SparkSession, df: DataFrame) -> str:
    success = False
    rand_name = "tmp_parquet_default"
    for _ in range(MAX_UPLOAD_PARQUET_TRIES):
        suffix = "".join(secrets.choice(string.ascii_letters) for _ in range(10))
        rand_name = f"tmp_parquet_{suffix}"
        if not session.catalog._jcatalog.tableExists(rand_name):
            success = True
            break
    if not success:
        msg = f"Failed to find name after {MAX_UPLOAD_PARQUET_TRIES} tries."
        raise RuntimeError(msg)

    df.write.mode("errorifexists").format("parquet").saveAsTable(rand_name)
    parquet_url = get_table_url(session, rand_name)
    return parquet_url
