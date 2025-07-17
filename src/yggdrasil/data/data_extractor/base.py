import abc

from yggdrasil.core.dtypes.dataset import ParquetDataset


class DataExtractor(abc.ABC):
    r"""
    Interface for data extractors. It defines methods for querying data from a
    data source, such as a database or a file system, then applies transformations
    to the data and returns temporary dataset.
    """

    @abc.abstractmethod
    def query_data(
        self,
        table_identifier: str,
        sample_range: tuple[float, float],
    ) -> ParquetDataset:
        r"""
        Queries data from a specified table and sample range.

        Args:
            table_identifier (str): Original table identifier.
            sample_range (tuple[float, float]): Range of samples to query.

        Returns:
            ParquetDataset: A dataset containing the queried data in Parquet format.
        """
