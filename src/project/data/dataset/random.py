import torch
from torch.utils.data import Dataset


class RandomData(Dataset):
    def __init__(
        self,
        num_features: int,
        length: int = 128,
    ):
        super().__init__()

        self.length = length
        self.num_features = num_features

        self.data = torch.rand(self.length, self.num_features)
        self.targets = torch.randint(0, 2, (self.length,), dtype=torch.float32)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.data[index],
            "target": self.targets[index].unsqueeze(dim=-1),
        }
