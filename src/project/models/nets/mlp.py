import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 64, num_layers: int = 2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList(
            [
                nn.Linear(
                    input_dim if i == 0 else hidden_dim,
                    output_dim if i == num_layers - 1 else hidden_dim,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
