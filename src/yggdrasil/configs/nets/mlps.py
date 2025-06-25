from omegaconf import MISSING

from yggdrasil.configs.utils import NET_STORE, fbuilds
from yggdrasil.models.mlp import (
    BinaryClassificationMLPNetwork,
    MultiClassificationMLPNetwork,
)

# multiple configurations for the same model with different
# hyperparameters can be stored for an easy access later
BinMLPConf = fbuilds(
    BinaryClassificationMLPNetwork,
    input_dim=MISSING,
    output_dim=1,  # Shouldn't be changed, as this is a binary classification model
    num_layers=2,
    hidden_dim=64,
)

MulticlassMLPConf = fbuilds(
    MultiClassificationMLPNetwork,
    input_dim=MISSING,
    output_dim=MISSING,  # should be set to the number of classes
    num_layers=2,
    hidden_dim=64,
)


def register_mlps() -> None:
    NET_STORE(BinMLPConf, name="bin-mlp")
    NET_STORE(MulticlassMLPConf, name="multiclass-mlp")
