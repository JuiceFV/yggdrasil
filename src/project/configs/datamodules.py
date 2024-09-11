from project.data.datamodules.example import ExampleDataModule
from project.data.dataset.random import RandomData

from .utils import ZENSTORE, fbuilds

dm_store = ZENSTORE(group="datamodule")

RandomDataConf = fbuilds(
    RandomData,
    length=128,
)

ExampleDMConf = fbuilds(
    ExampleDataModule,
    dataset=RandomDataConf,
    batch_size=16,
    num_workers=0,
)

dm_store(ExampleDMConf, name="example")
