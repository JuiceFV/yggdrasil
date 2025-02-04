from omegaconf import MISSING

from project.core.dtypes.dataset import TableSpec
from project.core.dtypes.options import DatasetOptions
from project.core.dtypes.preprocessing.options import PreprocessingOptions
from project.data.datamodules.example import ExampleDataModule

from .utils import ZENSTORE, fbuilds

dm_store = ZENSTORE(group="datamodule")

InputTableSpecConf = fbuilds(
    TableSpec,
    table_identifier=MISSING,
    train_table_sample=80.0,
    eval_table_sample=10.0,
    test_table_sample=10.0,
)

ExampleDMConf = fbuilds(
    ExampleDataModule,
    input_table_spec=InputTableSpecConf,
    dataset_options=fbuilds(DatasetOptions),
    features_preprocessing_options=fbuilds(PreprocessingOptions),
)


dm_store(ExampleDMConf, name="example")
