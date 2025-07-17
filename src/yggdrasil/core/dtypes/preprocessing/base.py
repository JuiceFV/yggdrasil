class InputColumn:
    r"""
    A class to represent the input columns of a dataset.
    These columns are being used in transfromation pipelines.

    .. note::
        List all the columns that could be shared across different
        datasets and transformations.

    Attributes:
        SAMPLE_ID (str): Unique integer identifier for the input column.
        FEATURES (str): Features column in :math:`f(X) = y` problems.
        TARGET (str): Target column in :math:`f(X) = y` problems.
    """
    SAMPLE_ID: str = "sample_id"
    FEATURES: str = "features"
    TARGET: str = "target"
