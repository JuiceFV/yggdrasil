import torch
from torch import nn

from project.core.config import param_hash
from project.core.dataclasses import dataclass
from project.core.dtypes.classification import BinaryOutput, BinaryPreprocessedInput
from project.models.base import BaseModel


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

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2
    ) -> None:
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
    __hash__ = param_hash

    input_dim: int
    output_dim: int
    hidden_dim: int
    num_layers: int

    def __post_init__(self) -> None:
        super().__init__()
        self.mlp = self._build_model()

    def _build_model(self) -> nn.Module:
        return MLP(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )


@dataclass
class BinaryClassificationMLPNetwork(MLPNetwork):
    __hash__ = param_hash

    def input_prototype(self) -> BinaryPreprocessedInput:
        return BinaryPreprocessedInput.from_tensors(
            target=torch.randint(0, 2, (1, 1), dtype=torch.float32),
            features=torch.randn(1, self.input_dim, dtype=torch.float32),
        )

    def forward(self, batch: BinaryPreprocessedInput) -> BinaryOutput:
        logits = self.mlp(batch.features)
        return BinaryOutput(probabilities=torch.sigmoid(logits), logits=logits)
