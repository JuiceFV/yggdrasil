from collections.abc import Callable
from functools import partial
from typing import Any

import torch
from torch import nn

from project.nn.activations import ActivationType
from project.nn.initializations import InitializationType


class HeterogeneousEncoder(nn.Module):
    initialization: str
    input_dim: int
    output_dim: int

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation: str = "relu",
        initialization: str = "kaiming_normal",
        batch_norm: bool = False,
        dropout_rate: float = 0.0,
        activation_cfg: dict[str, Any] | None = None,
        initialization_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        activation_cfg = activation_cfg or {}
        activation_fn = ActivationType[activation].value(**activation_cfg)
        layers: list[nn.Module] = []
        current_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(activation_fn)
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            current_dim = dim

        self.layers = nn.Sequential(*layers)
        self.initialization = initialization
        self.input_dim = input_dim
        self.output_dim = current_dim

        initialization_cfg = initialization_cfg or {}
        initialization_fn = partial(
            InitializationType[initialization].value, **initialization_cfg
        )
        self._initialize_learnable_params(initialization_fn)

    def _initialize_learnable_params(
        self, initialization_fn: Callable[[torch.Tensor], Any]
    ) -> None:
        for param in self.parameters():
            if param.dim() > 1:
                initialization_fn(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def extra_repr(self) -> str:
        return f"init_method={self.initialization}"
