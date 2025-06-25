from omegaconf import MISSING

import yggdrasil.data.data_extractor as de
from yggdrasil.core.dtypes.dataset import TableSpec
from yggdrasil.core.dtypes.options import DatasetOptions
from yggdrasil.core.dtypes.preprocessing.options import PreprocessingOptions
from yggdrasil.data.datamodules.example import ExampleDataModule
from yggdrasil.data.datamodules.mnist import MNISTDataModule

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
    data_extractor=fbuilds(de.ExampleDataExtractor),
    features_preprocessing_options=fbuilds(PreprocessingOptions),
)

MNiSTDMConf = fbuilds(
    MNISTDataModule,
    input_table_spec=InputTableSpecConf,
    dataset_options=fbuilds(DatasetOptions),
    data_extractor=fbuilds(de.MNISTDataExtractor),
    features_preprocessing_options=fbuilds(PreprocessingOptions),
)


dm_store(ExampleDMConf, name="example")
dm_store(MNiSTDMConf, name="mnist")
