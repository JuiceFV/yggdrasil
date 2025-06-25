from dataclasses import dataclass

import torch

from yggdrasil.core.dtypes.base import ExtraData, Feature, TensorDataClass
from yggdrasil.utils import init_logger

log = init_logger(__name__)


@dataclass
class BinaryPreprocessedInput(TensorDataClass):
    r"""
    Dataclass for binary classification tasks.
    """

    #: Target tensor. It must consist of two unique values. Typically, 0 and 1.
    target: torch.Tensor

    #: Features tensor. Normalized and preprocessed features.
    features: Feature

    #: Extra data for the model.
    extras: ExtraData | None = None

    @classmethod
    def from_input(
        cls,
        target: torch.Tensor,
        features: torch.Tensor,
        extras: ExtraData | None = None,
    ) -> "BinaryPreprocessedInput":
        r"""
        Create a BinaryPreprocessedInput instance from target and features tensors.

        Args:
            target (torch.Tensor): Preprocessed target tensor.
            features (torch.Tensor): Preprocessed features tensor.

        Raises:
            ValueError: If target and features have different number of samples.
            ValueError: If target is not binary.

        Returns:
            BinaryPreprocessedInput: Returns the data in suitable format for
        """
        if target.shape[0] != features.shape[0]:
            msg = "Target and features must have the same number of samples"
            raise ValueError(msg)
        if not (
            torch.all(
                torch.logical_or(target == torch.tensor(0), target == torch.tensor(1))
            )
            or torch.min(target) == torch.max(target)
        ):
            msg = "Target must be binary"
            raise ValueError(msg)
        return cls.from_tensors(
            target=target.to(torch.float32),
            features=features.to(torch.float32),
            extras=extras,
        )

    @classmethod
    def from_tensors(
        cls,
        target: torch.Tensor,
        features: torch.Tensor,
        extras: ExtraData | None = None,
    ) -> "BinaryPreprocessedInput":
        def annotation_checking(inp: torch.Tensor | None) -> None:
            if inp is not None and not isinstance(inp, torch.Tensor):
                msg = f"Expected {torch.Tensor | None}; but got {type(inp)}"
                raise TypeError(msg)

        annotation_checking(target)
        annotation_checking(features)

        return cls(
            target=target,
            features=Feature(dense_features=features),
            extras=extras,
        )


@dataclass
class MulticlassPreprocessedInput(TensorDataClass):
    target: torch.Tensor
    features: Feature
    nclasses: int | None = None
    extras: ExtraData | None = None

    @classmethod
    def from_input(
        cls,
        target: torch.Tensor,
        features: torch.Tensor,
        nclasses: int | None = None,
        extras: ExtraData | None = None,
    ) -> "MulticlassPreprocessedInput":
        if target.shape[0] != features.shape[0]:
            msg = "Target and features must have the same number of samples"
            raise ValueError(msg)
        if target.ndim > 2:
            msg = (
                "Multiclass target should be either class labels (nsamples x 1) "
                "or distribution (nsamples x nclasses) tensor."
            )
            raise ValueError(msg)
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        if nclasses is not None and nclasses <= 0:
            msg = "Number of classes must be a positive integer"
            raise ValueError(msg)
        if target.size(1) > 1:
            if nclasses is not None:
                log.warning(
                    "The number of classes will be infered from the target tensor."
                )
            nclasses = target.size(1)

        return cls.from_tensors(
            target=target.to(torch.float32),
            features=features.to(torch.float32),
            nclasses=nclasses,
            extras=extras,
        )

    @classmethod
    def from_tensors(
        cls,
        target: torch.Tensor,
        features: torch.Tensor,
        nclasses: int | None = None,
        extras: ExtraData | None = None,
    ) -> "MulticlassPreprocessedInput":
        def annotation_checking(inp: torch.Tensor | None) -> None:
            if inp is not None and not isinstance(inp, torch.Tensor):
                msg = f"Expected {torch.Tensor | None}; but got {type(inp)}"
                raise TypeError(msg)

        annotation_checking(target)
        annotation_checking(features)

        return cls(
            target=target,
            features=Feature(dense_features=features),
            nclasses=nclasses,
            extras=extras,
        )

    @property
    def probabilistic_target(self) -> torch.Tensor:
        r"""
        Returns the target tensor as probabilities.
        If the target is a class label, it converts it to a one-hot encoded tensor.

        Raises:
            ValueError: If the number of classes is not defined.

        Returns:
            torch.Tensor: The target tensor as probabilities.
            If the target is a class label, it converts it to a one-hot encoded tensor.
        """
        if self.nclasses is None:
            msg = "Undefined total number of classes."
            raise ValueError(msg)
        if self.target.size(1) == 1:
            return torch.nn.functional.one_hot(
                self.target, num_classes=self.nclasses
            ).to(torch.float32)
        return self.target

    @property
    def labeled_target(self) -> torch.Tensor:
        r"""
        Returns the target tensor as class labels.
        If the target is already a class label, it returns it as is.

        Returns:
            torch.Tensor: The target tensor as class labels.
        """
        if self.target.size(1) == 1:
            return self.target.squeeze(-1).to(torch.int64)
        return torch.argmax(self.target, dim=-1).to(torch.int64)


@dataclass
class ClassificationOutput(TensorDataClass):
    r"""
    Dataclass for classification outputs.
    """

    #: Logits tensor.
    logits: torch.Tensor

    #: Dimension which is considered as the class dimension.
    #: Defaults to -1, which means the last dimension is used.
    dim: int = -1

    def __post_init__(self) -> None:
        if self.logits.ndim < 2:
            msg = (
                "Logits tensor has less than 2 dimensions. "
                "Adding a batch dimension to the logits tensor."
            )
            log.warning(msg)
            self.logits = self.logits.unsqueeze(0)

    @property
    def nclasses(self) -> int:
        r"""
        Number of classes in the classification task.
        """
        return self.logits.shape[1]

    @nclasses.setter
    def nclasses(self, value: int) -> None:
        r""" "
        Set the number of classes in the classification task.
        This property is read-only and cannot be set directly.
        It is determined by the shape of the logits tensor.

        Args:
            value (int): The number of classes to set.

        Raises:
            RuntimeError: If trying to set nclasses, as it is a read-only property.
        """
        msg = (
            "nclasses is a read-only property. "
            "It is determined by the shape of the logits tensor."
        )
        raise RuntimeError(msg)

    @property
    def probabilities(self) -> torch.Tensor:
        r"""
        Returns the probabilities of each class.
        """
        if self.nclasses == 1:
            return torch.sigmoid(self.logits)
        return torch.softmax(self.logits, dim=self.dim)
