from project.data.datamodules.example import ExampleDataModule
from project.data.dataset.random import RandomData


def test_example_datamodule() -> None:
    num_features = 64
    length = 128
    batch_size = 16

    dataset = RandomData(num_features=num_features, length=length)
    dm = ExampleDataModule(dataset=dataset, batch_size=batch_size, num_workers=0)

    dm.prepare_data()
    dm.setup("fit")

    num_steps = sum(1 for _ in dm.train_dataloader())
    assert num_steps == length // batch_size
