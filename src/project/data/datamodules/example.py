from collections.abc import Callable

import lightning as L
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from project.core.dtypes.classification import BinaryPreprocessedInput
from project.data.dataset.random import RandomDataset
from project.preprocessing.batch_preprocessor import BatchPreprocessor


def collate_and_preprocess(
    batch_preprocessor: BatchPreprocessor,
) -> Callable[[list[dict]], torch.Tensor]:
    r"""
    Default colate function for DataLoader with batch preprocessing.

    Example::

        from torch.utils.data import DataLoader
        from project.data.dataset.random import RandomData

        dataloader = DataLoader(
            RandomData(num_features=64, length=1024),
            batch_size=64,
            collate_fn=collate_and_preprocess(
                batch_preprocessor=ExampleBatchPreprocessor()
        )

        # batch == BinaryPreprocessedInput(features=..., target=...)
        batch = next(iter(dataloader))

    .. todo::
        * Further this function will be defined in another module.

    .. warning::
        :func:`~torch.utils.data.default_collate` doesn't properly handle Decimals.
        In case of encoding timestamps as Decimals, consider using another collate
        function.

    .. warning::
        In case of distributed training this function needs to be pickled. It's
        imposible to pickle because we define :func:`collate_fn` inside this function.

    Args:
        batch_preprocessor (BatchPreprocessor): Preprocessor applied to a batch of
            ``dict[str, torch.Tensor]``.

    Returns:
        ~collections.abc.Callable[[list[dict]], torch.Tensor]:
            Collate function for DataLoader.
    """

    def collate_fn(batch_list: list[dict]) -> torch.Tensor:
        batch = default_collate(batch_list)
        preprocessed_batch: torch.Tensor = batch_preprocessor(batch)
        return preprocessed_batch

    return collate_fn


class ExampleBatchPreprocessor(BatchPreprocessor):
    r"""
    Batch preprocessor for example data. This preprpcessor is an example
    and does nothing except some basic checks. It returns an instance of
    :class:`~project.core.dtypes.base.TensorDataClass` suitable for
    binary classification task.

    1. Unsqueeze target tensor to make it 2D.
    2. Create BinaryPreprocessedInput instance.
    """

    def forward(self, batch: dict[str, torch.Tensor]) -> BinaryPreprocessedInput:
        r"""
        Applying preprocessing to a batch of tensor data.

        Args:
            batch (dict[str, torch.Tensor]): Batch of tensor data.

        Returns:
            BinaryPreprocessedInput: Preprocessed batch intantiated
            from binary specified :class:`~project.core.dtypes.base.TensorDataClass`
        """
        return BinaryPreprocessedInput.from_input(
            target=batch["target"].unsqueeze(-1), features=batch["features"]
        )


class ExampleDataModule(L.LightningDataModule):
    r"""
    Example data module for binary classification task. The module instantiates the
    :class:`~lightning.pytorch.core.datamodule.LightningDataModule` class. This module
    implements only:

    1. :meth:`~lightning.pytorch.core.hooks.DataHooks.train_dataloader`
       - training dataloader with batch preprocessing. The dataloader returns a value
       of :class:`~project.core.dtypes.classification.base.BinaryPreprocessedInput`.
    2. :meth:`~lightning.pytorch.core.hooks.DataHooks.val_dataloader`
       - validation dataloader with batch preprocessing. The dataloader returns a value
       of :class:`~project.core.dtypes.classification.base.BinaryPreprocessedInput`.
    3. :meth:`~lightning.pytorch.core.hooks.DataHooks.test_dataloader`
       - test dataloader with batch preprocessing. The dataloader returns a value
       of :class:`~project.core.dtypes.classification.base.BinaryPreprocessedInput`.

    Args:
        dataset (RandomData): :class:`~torch.utils.data.Dataset` instance.
            Randomly generated data.
        batch_size (int, optional): Batch size passed to
            :class:`~torch.utils.data.DataLoader`, by default 16
        num_workers (int, optional): Number of shardes for data-parallel training,
            by default 1
    """

    def __init__(
        self,
        dataset: RandomDataset,
        batch_size: int = 16,
        num_workers: int = 1,
        **dataloader_kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.dataset = dataset

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataloader_kwargs = dataloader_kwargs

    def _build_dataloader(self) -> DataLoader:
        r"""
        Common method for building dataloaders.

        Returns:
            torch.utils.data.DataLoader: DataLoader instance.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_and_preprocess(
                batch_preprocessor=ExampleBatchPreprocessor()
            ),
            **self.dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        r"""
        Implements the training dataloader.

        Returns:
            torch.utils.data.DataLoader: Training dataloader.
        """
        return self._build_dataloader()

    def val_dataloader(self) -> DataLoader:
        r"""
        Implements the validation dataloader.

        .. note:: Validation dataloader is the same as the training dataloader.

        Returns:
            torch.utils.data.DataLoader: Validation dataloader.
        """
        return self._build_dataloader()

    def test_dataloader(self) -> DataLoader:
        r"""
        Implements the test dataloader.

        .. note:: test dataloader is the same as the training dataloader.

        Returns:
            torch.utils.data.DataLoader: Test dataloader.
        """
        return self._build_dataloader()
