from dataclasses import dataclass

from project.core.dtypes.preprocessing.options import PreprocessingOptions

__all__ = ["DatasetOptions", "PreprocessingOptions"]


@dataclass
class DatasetOptions:
    r"""
    Batch reader options.
    """

    #: Batch size.
    minibatch_size: int = 1024

    #: Number of processes to use for reading the dataset.
    num_proc: int = 1

    #: Use streaming mode for reading a dataset (Useful for large datasets).
    streaming: bool = False
