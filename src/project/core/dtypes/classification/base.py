from dataclasses import dataclass

import torch

from project.core.dtypes.base import ExtraData, Feature, TensorDataClass


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
        return cls.from_tensors(target=target, features=features, extras=extras)

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
class BinaryOutput(TensorDataClass):
    r"""
    Dataclass for binary classification outputs.
    """

    #: Predicted probabilities tensor.
    probabilities: torch.Tensor

    #: Logits tensor.
    logits: torch.Tensor
