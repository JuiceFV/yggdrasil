from project.configs.utils import NET_STORE, fbuilds
from project.models.nets.mlp import MLP

MLPConf = fbuilds(MLP, output_dim=1, num_layers=2, hidden_dim=64)

DeepMLPConf = fbuilds(
    MLP,
    output_dim=1,
    num_layers=6,
    hidden_dim=64,
)

WideMLPConf = fbuilds(
    MLP,
    output_dim=1,
    num_layers=2,
    hidden_dim=256,
)


def register_mlps():
    NET_STORE(MLPConf, name="mlp")
    NET_STORE(DeepMLPConf, name="deep-mlp")
    NET_STORE(WideMLPConf, name="wide-mlp")
