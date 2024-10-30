from omegaconf import MISSING

from project.configs.utils import NET_STORE, fbuilds
from project.models.mlp import BinaryClassificationMLPNetwork

# multiple configurations for the same model with different
# hyperparameters can be stored for an easy access later
MLPConf = fbuilds(
    BinaryClassificationMLPNetwork,
    input_dim=MISSING,
    output_dim=1,
    num_layers=2,
    hidden_dim=64,
)

DeepMLPConf = fbuilds(
    BinaryClassificationMLPNetwork,
    input_dim=MISSING,
    output_dim=1,
    num_layers=6,
    hidden_dim=64,
)

WideMLPConf = fbuilds(
    BinaryClassificationMLPNetwork,
    input_dim=MISSING,
    output_dim=1,
    num_layers=2,
    hidden_dim=256,
)


def register_mlps() -> None:
    NET_STORE(MLPConf, name="mlp")
    NET_STORE(DeepMLPConf, name="deep-mlp")
    NET_STORE(WideMLPConf, name="wide-mlp")
