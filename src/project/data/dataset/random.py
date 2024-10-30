import torch
from torch.utils.data import Dataset


class RandomDataset(Dataset):
    r"""
    Randomly generated dataset for testing purposes.

    Args:
        num_features (int): Dimensionality of the feature space.
        length (int, optional): Dataset size, by default 128
    """

    def __init__(
        self,
        num_features: int,
        length: int = 128,
    ) -> None:
        super().__init__()

        self.length = length
        self.num_features = num_features

        self.data = torch.rand(self.length, self.num_features)
        self.targets = torch.randint(0, 2, (self.length,), dtype=torch.float32)

    def __len__(self) -> int:
        r"""
        Length of the dataset.

        Returns:
            int: Length of the entire dataset.
        """
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        r"""
        Get record by index in suitable format.

        Args:
            index (int): Index of record in the dataset.

        Returns:
            dict[str, torch.Tensor]: Dictionary with features and target tensors.
        """
        return {
            "features": self.data[index],
            "target": self.targets[index],
        }
