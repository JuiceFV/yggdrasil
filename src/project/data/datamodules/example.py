import lightning as L
from torch.utils.data import DataLoader

from project.data.dataset.random import RandomData


class ExampleDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: RandomData,
        batch_size: int = 16,
        num_workers: int = 1,
        **dataloader_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset = dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataloader_kwargs = dataloader_kwargs

    def _build_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **self.dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return self._build_dataloader()

    # keep in mind that in this example, for simplicity,
    # validation and testing uses the same generated data as training
    def val_dataloader(self) -> DataLoader:
        return self._build_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self._build_dataloader()
