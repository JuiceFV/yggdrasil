import abc

from project.core.dtypes.dataset import ParquetDataset


class DataExtractor:
    @abc.abstractmethod
    def query_data(
        self,
        table_identifier: str,
        sample_range: tuple[float, float],
    ) -> ParquetDataset:
        pass
