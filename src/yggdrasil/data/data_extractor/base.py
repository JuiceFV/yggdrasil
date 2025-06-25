import abc

from yggdrasil.core.dtypes.dataset import ParquetDataset


class DataExtractor(abc.ABC):
    @abc.abstractmethod
    def query_data(
        self,
        table_identifier: str,
        sample_range: tuple[float, float],
    ) -> ParquetDataset:
        pass
