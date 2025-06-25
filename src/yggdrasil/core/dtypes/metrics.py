import torch

from yggdrasil.core.dataclasses import dataclass
from yggdrasil.core.dtypes.base import TensorDataClass


@dataclass
class MetricInput(TensorDataClass):
    r"""
    This is a helper dataclass to ensure in a proper input for metrics.
    To make the workflow highly configurable we consider metrics computation
    outside of an optimization step. For these purposes we define the generic
    :class:`~yggdrasil.callbacks.metrics.MetricCallback`. It accepts any metric
    instantiated from :class:`~torchmetrics.Metric` class. The different metrics
    implement their own :meth:`~torchmetrics.Metric.update` methods. The method
    accepts different inputs, but they're classified into several groups.
    Most of them accepts two tensors: ``preds`` and ``target``.

    ..note::
        This class has to be a union of arguments that are accepted by the
        :meth:`~torchmetrics.Metric.update` method. Ideally, it should be
        configured dynamically based on the metrics that are used.

    ..warning::
        This is the first version of this dataclass. It's oversimplified and
        which is more important it's static.
    """

    preds: torch.Tensor | None = None
    target: torch.Tensor | None = None
