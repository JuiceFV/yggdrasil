from dataclasses import dataclass

from yggdrasil.core.dtypes.preprocessing.options import PreprocessingOptions

__all__ = ["DatasetOptions", "PreprocessingOptions"]


@dataclass
class DatasetOptions:
    r"""
    Batch reader options.

    Args:
        minibatch_size (int): Batch size for reading the dataset.
        num_proc (int): Number of processes to use for reading the dataset.
        streaming (bool): Use streaming mode for reading a dataset (Useful for large datasets).
    """

    minibatch_size: int = 1024
    num_proc: int = 1
    streaming: bool = False
