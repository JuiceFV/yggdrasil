from yggdrasil.core.dataclasses import dataclass


@dataclass
class TableSpec:
    r"""
    Represents a specification for a table in a dataset.

    Args:
        table_identifier (str): The identifier for the table, typically a table name.
        train_table_sample (float | None): The sample size for the training set.
        eval_table_sample (float | None): The sample size for the evaluation set.
        test_table_sample (float | None): The sample size for the test set.
    """
    table_identifier: str
    train_table_sample: float | None = None
    eval_table_sample: float | None = None
    test_table_sample: float | None = None


@dataclass
class ParquetDataset:
    r"""
    Represents a dataset stored in Parquet format.

    Args:
        dataset_url (str): The URL or path to the Parquet dataset.
    """

    dataset_url: str
