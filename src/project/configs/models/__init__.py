from hydra_zen import make_config

from project.configs.utils import ZENSTORE, fbuilds
from project.modules.binary_classification import BinaryClassificationModule

model_store = ZENSTORE(group="model")

# dynamic generation of dataclass config for the model. All its arguments will be
# included automaticall except for `criterion` and `metrics`. We will provide
# `criterion` manually and leave `metrics` as a default for now
BinClassModelConf = fbuilds(BinaryClassificationModule, zen_exclude=["metrics"])


# defaults list here refering to the local config group
# since the model substore is not a _global_ store
binary_classifier_conf = make_config(
    hydra_defaults=[
        "_self_",
        {"net": "mlp"},
        {"optimizer": "adam"},
        {"scheduler": None},
    ],
    bases=(BinClassModelConf,),
)


model_store(binary_classifier_conf, name="binary_classifier")
