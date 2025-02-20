import pytest
import torch
from torch import nn

from project.core.dtypes.classification import BinaryOutput, BinaryPreprocessedInput
from project.models.mlp import MLP, BinaryClassificationMLPNetwork


class TestUtilityMLP:
    @pytest.fixture
    def setup_mlp(self) -> MLP:
        input_dim = 64
        output_dim = 10
        hidden_dim = 32
        num_layers = 3
        return MLP(input_dim, output_dim, hidden_dim, num_layers)

    def test_mlp_output_shape(self, setup_mlp: MLP) -> None:
        mlp = setup_mlp
        x = torch.randn(8, mlp.input_dim)
        output = mlp(x)
        assert output.shape == (8, mlp.output_dim)

    def test_weight_initialization(self, setup_mlp: MLP) -> None:
        mlp = setup_mlp
        for i, layer in enumerate(mlp.layers):
            linear_layer = layer.get_submodule("0")
            if i == 0:
                assert linear_layer.weight.shape == (mlp.hidden_dim, mlp.input_dim)
            elif i == mlp.num_layers - 1:
                assert linear_layer.weight.shape == (mlp.output_dim, mlp.hidden_dim)
            else:
                assert linear_layer.weight.shape == (mlp.hidden_dim, mlp.hidden_dim)

    def test_mlp_non_linearity(self, setup_mlp: MLP) -> None:
        mlp = setup_mlp
        x = torch.randn(8, mlp.input_dim)
        linear_output = x
        for layer in mlp.layers:
            linear_layer = layer.get_submodule("0")
            linear_output = linear_output @ linear_layer.weight.T + linear_layer.bias
        output_mlp = mlp(x)
        assert not torch.allclose(
            output_mlp, linear_output, atol=1e-6
        ), "MLP is not applying non-linearity"


class TestForwardPassMLP:
    def test_forward_pass_computation(self) -> None:
        input_dim = 3
        output_dim = 2
        hidden_dim = 4
        num_layers = 2
        mlp = MLP(input_dim, output_dim, hidden_dim, num_layers)

        with torch.no_grad():
            mlp.layers[0].get_submodule("0").weight = nn.Parameter(
                torch.tensor(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0],
                    ]
                )
            )
            mlp.layers[0].get_submodule("0").bias = nn.Parameter(
                torch.tensor([1.0, 1.0, 1.0, 1.0])
            )
            mlp.layers[1].get_submodule("0").weight = nn.Parameter(
                torch.tensor([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
            )
            mlp.layers[1].get_submodule("0").bias = nn.Parameter(
                torch.tensor([0.5, 0.5])
            )

        x = torch.tensor([[1.0, 2.0, 3.0]])
        expected_hidden = torch.relu(
            torch.tensor(
                [
                    [
                        1.0 * 1 + 2.0 * 2 + 3.0 * 3 + 1,
                        1.0 * 4 + 2.0 * 5 + 3.0 * 6 + 1,
                        1.0 * 7 + 2.0 * 8 + 3.0 * 9 + 1,
                        1.0 * 10 + 2.0 * 11 + 3.0 * 12 + 1,
                    ]
                ]
            )
        )
        expected_output = torch.tensor(
            [
                [
                    1.0 * expected_hidden[0][0]
                    + 1.0 * expected_hidden[0][1]
                    + 1.0 * expected_hidden[0][2]
                    + 1.0 * expected_hidden[0][3]
                    + 0.5,
                    2.0 * expected_hidden[0][0]
                    + 2.0 * expected_hidden[0][1]
                    + 2.0 * expected_hidden[0][2]
                    + 2.0 * expected_hidden[0][3]
                    + 0.5,
                ]
            ]
        )
        output = mlp(x)
        assert torch.allclose(output, expected_output, atol=1e-5)


class TestBinaryClassificationMLPNetwork:
    @pytest.fixture
    def setup_bc_mlp_network(self) -> BinaryClassificationMLPNetwork:
        input_dim = 64
        output_dim = 10
        hidden_dim = 32
        num_layers = 3
        return BinaryClassificationMLPNetwork(
            input_dim, output_dim, hidden_dim, num_layers
        )

    def test_bc_mlp_network_output_shape(
        self, setup_bc_mlp_network: BinaryClassificationMLPNetwork
    ) -> None:
        mlp_network = setup_bc_mlp_network
        batch = BinaryPreprocessedInput.from_tensors(
            target=torch.randint(0, 2, (8, 1), dtype=torch.float32),
            features=torch.randn(8, mlp_network.input_dim, dtype=torch.float32),
        )
        output: BinaryOutput = mlp_network.forward(batch)
        assert output.logits.shape == (8, mlp_network.output_dim)

    def test_bc_mlp_network_input_prototype(
        self, setup_bc_mlp_network: BinaryClassificationMLPNetwork
    ) -> None:
        mlp_network = setup_bc_mlp_network
        prototype = mlp_network.input_prototype()
        assert prototype.features.dense_features.shape == (1, mlp_network.input_dim)
        assert prototype.target.shape == (1, 1)
