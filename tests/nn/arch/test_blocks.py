import pytest
import torch
from torch import nn

from project.nn.arch.blocks import HeterogeneousEncoder


class TestHeterogeneousEncoder:
    def test_heterogeneous_encoder_forward(self) -> None:
        input_dim = 10
        hidden_dims = [20, 30, 40]
        batch_size = 5
        model = HeterogeneousEncoder(input_dim, hidden_dims)
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        assert output.shape == (batch_size, hidden_dims[-1])

    def test_unsupported_activation(self) -> None:
        with pytest.raises(KeyError, match="'unsupported_activation'"):
            HeterogeneousEncoder(10, [20, 30], activation="unsupported_activation")

    def test_unsupported_initialization(self) -> None:
        with pytest.raises(
            KeyError,
            match="'unsupported_initialization'",
        ):
            HeterogeneousEncoder(
                10, [20, 30], initialization="unsupported_initialization"
            )

    def test_initialization(self) -> None:
        input_dim = 10
        hidden_dims = [20, 30]
        model = HeterogeneousEncoder(
            input_dim, hidden_dims, initialization="kaiming_uniform"
        )
        for param in model.parameters():
            if param.dim() > 1:
                assert torch.all(param != 0), "Parameters should be initialized"

    def test_batch_norm(self) -> None:
        input_dim = 10
        hidden_dims = [20, 30]
        model = HeterogeneousEncoder(input_dim, hidden_dims, batch_norm=True)
        assert any(
            isinstance(layer, nn.BatchNorm1d) for layer in model.layers
        ), "BatchNorm1d should be in layers"

    def test_dropout(self) -> None:
        input_dim = 10
        hidden_dims = [20, 30]
        dropout_rate = 0.5
        model = HeterogeneousEncoder(input_dim, hidden_dims, dropout_rate=dropout_rate)
        assert any(
            isinstance(layer, nn.Dropout) for layer in model.layers
        ), "Dropout should be in layers"

    def test_extra_repr(self) -> None:
        model = HeterogeneousEncoder(10, [20, 30], initialization="kaiming_normal")
        assert "init_method=kaiming_normal" in model.extra_repr()

    def test_activation_configuration(self) -> None:
        input_dim = 10
        hidden_dims = [20, 30]
        activation_cfg = {"negative_slope": 0.1}
        model = HeterogeneousEncoder(
            input_dim,
            hidden_dims,
            activation="leaky_relu",
            activation_cfg=activation_cfg,
        )
        assert any(
            isinstance(layer, nn.LeakyReLU) for layer in model.layers
        ), "LeakyReLU should be in layers"

    def test_initialization_configuration(self) -> None:
        input_dim = 10
        hidden_dims = [20, 30]
        initialization_cfg = {"a": 0.1}
        model = HeterogeneousEncoder(
            input_dim,
            hidden_dims,
            initialization="kaiming_uniform",
            initialization_cfg=initialization_cfg,
        )
        for param in model.parameters():
            if param.dim() > 1:
                assert torch.all(
                    param != 0
                ), "Parameters should be initialized with kaiming_uniform"

    def test_no_hidden_layers(self) -> None:
        input_dim = 10
        hidden_dims: list[int] = []
        batch_size = 5
        model = HeterogeneousEncoder(input_dim, hidden_dims)
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        assert output.shape == (
            batch_size,
            input_dim,
        ), "Output shape should match input shape when no hidden layers"

    def test_single_hidden_layer(self) -> None:
        input_dim = 10
        hidden_dims = [20]
        batch_size = 5
        model = HeterogeneousEncoder(input_dim, hidden_dims)
        x = torch.randn(batch_size, input_dim)
        output = model(x)
        assert output.shape == (
            batch_size,
            hidden_dims[-1],
        ), "Output shape should match the single hidden layer size"
