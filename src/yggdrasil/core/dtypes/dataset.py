from yggdrasil.core.dataclasses import dataclass


@dataclass
class TableSpec:
    table_identifier: str
    train_table_sample: float | None = None
    eval_table_sample: float | None = None
    test_table_sample: float | None = None


@dataclass
class ParquetDataset:
    dataset_url: str
