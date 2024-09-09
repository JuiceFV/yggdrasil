from hydra_zen import make_config
from omegaconf import MISSING

from project.configs.utils import ZENSTORE, fbuilds
from project.models.binary_classification import BinaryClassificationModel

model_store = ZENSTORE(group="model")

BinClassModelConf = fbuilds(
    BinaryClassificationModel, zen_exclude=["criterion", "metrics"]
)

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
