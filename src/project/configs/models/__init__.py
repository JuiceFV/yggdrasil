from hydra_zen import make_config
from omegaconf import MISSING

from project.configs.utils import ZENSTORE, fbuilds
from project.models.binary_classification import BinaryClassificationModel

model_store = ZENSTORE(group="model")

# dynamic generation of dataclass config for the model. All its arguments will be
# included automaticall except for `criterion` and `metrics`. We will provide
# `criterion` manually and leave `metrics` as a default for now
BinClassModelConf = fbuilds(
    BinaryClassificationModel, zen_exclude=["criterion", "metrics"]
)


# defaults list here refering to the local config group
# since the model substore is not a _global_ store
binary_classifier_conf = make_config(
    hydra_defaults=[
        "_self_",
        {"net": "mlp"},
        {"criterion": "bce_logits"},
        {"optimizer": "adam"},
        {"scheduler": None},
    ],
    bases=(BinClassModelConf,),
)


model_store(binary_classifier_conf, name="binary_classifier")
