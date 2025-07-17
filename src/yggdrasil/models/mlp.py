import torch
from torch import nn

from yggdrasil.core.config import param_hash
from yggdrasil.core.dataclasses import dataclass
from yggdrasil.core.dtypes.classification import (
    BinaryPreprocessedInput,
    ClassificationOutput,
    MulticlassPreprocessedInput,
)
from yggdrasil.models.base import BaseModel


class MLP(nn.Module):
    r"""
    Multilayer perceptron (MLP) model.

    .. math::

        \mathbf{h}^{(l)} = \sigma \left(
            \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}
        \right), \quad \text{for } l = 1, 2, \dots, L

    where:

    * :math:`\mathbf{h}^{(0)} = \mathbf{x}` is the input vector.
    * :math:`\mathbf{h}^{(l)}` is the output (or activation) of layer :math:`l`.
    * :math:`\mathbf{W}^{(l)}` is the weight matrix for layer :math:`l`.
    * :math:`\mathbf{b}^{(l)}` is the bias vector for layer :math:`l`.
    * :math:`\sigma(\cdot)` is the activation function (e.g., ReLU, Sigmoid, etc.).
    * :math:`L` is the number of layers.

    The output y of the MLP after the last layer (without applying any activation)
    can be written as:

    .. math::

        \mathbf{y} = \mathbf{W}^{(L)} \mathbf{h}^{(L-1)} + \mathbf{b}^{(L)}

    :math:`y` is the final output, which can be followed by a softmax or other
    activation depending on the task (classification, regression, etc.).

    Args:
        input_dim (int): Dimensionality of the input vector.
        output_dim (int): Dimensionality of the output vector.
        hidden_dim (int, optional): Dimensionality of the weight matrix, by default 64
        num_layers (int, optional): Number of layers MLP consists of, by default 2
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        input_dim if i == 0 else hidden_dim,
                        output_dim if i == num_layers - 1 else hidden_dim,
                    ),
                    nn.ReLU() if i != num_layers - 1 else nn.Identity(),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): Input tensor.

        Shape:
            - x: :math:`(*, H_{in})`
            - output: :math:`(*, H_{out})`

        Notations:
            - :math:`H_{in}` - Input vector's dimensionality.
            - :math:`H_{out}` - Output vector's dimensionality.
            - :math:`*` - Any number of dimensions including none.

        Returns:
            torch.Tensor: Output tensor (last layer of weight matrix).
        """
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class MLPNetwork(BaseModel):
    """
    MLPNetwork is a dataclass that encapsulates the :class:`MLP` model and its parameters.

    Args:
        input_dim (int): Dimensionality of the input vector.
        output_dim (int): Dimensionality of the output vector.
        hidden_dim (int, optional): Dimensionality of the weight matrix.
        num_layers (int, optional): Number of layers MLP consists of.
    """

    __hash__ = param_hash

    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int

    def __post_init__(self) -> None:
        super().__init__()
        self.mlp = self._build_model()

    def _build_model(self) -> nn.Module:
        r"""
        Builds the MLP model based on the parameters defined in the dataclass.

        Returns:
            nn.Module: An instance of the MLP model.
        """
        return MLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )


@dataclass
class BinaryClassificationMLPNetwork(MLPNetwork):
    r"""
    BinaryClassificationMLPNetwork is a dataclass that encapsulates the :class:`MLP` model
    specifically for binary classification tasks.

    Example::

        from yggdrasil.models.mlp import BinaryClassificationMLPNetwork
        bin_mlp_network = BinaryClassificationMLPNetwork(input_dim=10, hidden_dim=64, num_layers=2)
        print(bin_mlp_network)
        # Output: BinaryClassificationMLPNetwork(input_dim=10, output_dim=1, hidden_dim=64, num_layers=2)

    Config Example::

        BinMLPConf = fbuilds(
            BinaryClassificationMLPNetwork,
            input_dim=MISSING,
            output_dim=1,  # Shouldn't be changed, as this is a binary classification model
            num_layers=2,
            hidden_dim=64,
        )

    Args:
        input_dim (int): Dimensionality of the input vector.
        output_dim (int): Dimensionality of the output vector, should be 1 for binary classification.
        hidden_dim (int, optional): Dimensionality of the weight matrix.
        num_layers (int, optional): Number of layers MLP consists of.

    Raises:
        ValueError: If `output_dim` is not equal to 1, as this is a binary classification model.
    """

    __hash__ = param_hash

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.output_dim != 1:
            msg = f"BinaryClassificationMLPNetwork requires output_dim=1, got {self.output_dim}"
            raise ValueError(msg)

    def input_prototype(self) -> BinaryPreprocessedInput:
        """
        Returns a prototype input for the binary classification model.

        Returns:
            BinaryPreprocessedInput: An instance of BinaryPreprocessedInput with random tensors.
        """
        return BinaryPreprocessedInput.from_tensors(
            target=torch.randint(0, 2, (1, 1), dtype=torch.float32),
            features=torch.randn(1, self.input_dim, dtype=torch.float32),
        )

    def forward(self, batch: BinaryPreprocessedInput) -> ClassificationOutput:
        """
        Forward pass of the binary classification MLP model.

        Args:
            batch (BinaryPreprocessedInput): Batch of preprocessed input data.

        Returns:
            ClassificationOutput: Output of the model containing logits.
        """
        logits = self.mlp(batch.features.dense_features)
        return ClassificationOutput(logits=logits)


@dataclass
class MultiClassificationMLPNetwork(MLPNetwork):
    r"""
    MultiClassificationMLPNetwork is a dataclass that encapsulates the :class:`MLP` model
    specifically for multi-class classification tasks.

    Example::

        from yggdrasil.models.mlp import MultiClassificationMLPNetwork
        multiclass_mlp_network = MultiClassificationMLPNetwork(input_dim=10, output_dim=5, hidden_dim=64, num_layers=2)
        print(multiclass_mlp_network)
        # Output: MultiClassificationMLPNetwork(input_dim=10, output_dim=5, hidden_dim=64, num_layers=2)

    Config Example::

        MulticlassMLPConf = fbuilds(
            MultiClassificationMLPNetwork,
            input_dim=MISSING,
            output_dim=MISSING,  # should be set to the number of classes
            num_layers=2,
            hidden_dim=64,
        )

    Args:
        input_dim (int): Dimensionality of the input vector.
        output_dim (int): Dimensionality of the output vector, should be equal to the number of classes.
        hidden_dim (int, optional): Dimensionality of the weight matrix.
        num_layers (int, optional): Number of layers MLP consists of.
    """

    __hash__ = param_hash

    def input_prototype(self) -> MulticlassPreprocessedInput:
        r"""
        Returns a prototype input for the multi-class classification model.

        Returns:
            MulticlassPreprocessedInput: An instance of MulticlassPreprocessedInput with random tensors.
        """
        return MulticlassPreprocessedInput.from_tensors(
            target=torch.randint(0, self.output_dim, (1, 1), dtype=torch.float32),
            features=torch.randn(1, self.input_dim, dtype=torch.float32),
            nclasses=self.output_dim,
        )

    def forward(self, batch: MulticlassPreprocessedInput) -> ClassificationOutput:
        r"""
        Forward pass of the multi-class classification MLP model.

        Args:
            batch (MulticlassPreprocessedInput): Batch of preprocessed input data.

        Returns:
            ClassificationOutput: Output of the model containing logits.
        """
        logits = self.mlp(batch.features.dense_features)
        return ClassificationOutput(logits=logits)
