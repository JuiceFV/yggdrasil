import torch

from project.models.nets.mlp import MLP


def test_mlp():
    input_dim = 64
    output_dim = 16

    net = MLP(input_dim=input_dim, output_dim=output_dim)
    batch_size = 32
    batch = torch.randn(batch_size, input_dim)
    out = net(batch)

    assert out.shape == (batch_size, output_dim)
