import torch

from project.data.dataset.random import RandomDataset


class TestExampleDataset:
    def test_random_data_length(self) -> None:
        dataset_length = 50
        dataset = RandomDataset(num_features=10, length=dataset_length)
        assert len(dataset) == dataset_length

    def test_random_data_features_shape(self) -> None:
        num_features = 10
        dataset_length = 50
        dataset = RandomDataset(num_features=num_features, length=dataset_length)
        sample = dataset[0]
        assert sample["features"].shape == (num_features,)

    def test_random_data_target_type(self) -> None:
        dataset = RandomDataset(num_features=10, length=50)
        sample = dataset[0]
        assert isinstance(sample["target"], torch.Tensor)

    def test_random_data_target_values(self) -> None:
        dataset = RandomDataset(num_features=10, length=50)
        sample = dataset[0]
        assert sample["target"].item() in [0.0, 1.0]

    def test_random_data_indexing(self) -> None:
        dataset = RandomDataset(num_features=10, length=50)
        sample1 = dataset[0]
        sample2 = dataset[1]
        assert not torch.equal(sample1["features"], sample2["features"])
