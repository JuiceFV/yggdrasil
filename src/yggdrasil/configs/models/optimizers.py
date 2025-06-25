from torch.optim import Adam, AdamW

from yggdrasil.configs.utils import ZENSTORE, pfbuilds

optim_store = ZENSTORE(group="model/optimizer")


AdamConf = pfbuilds(
    Adam,
    lr=0.001,
    weight_decay=0.0,
)
optim_store(AdamConf, name="adam")


AdamWConf = pfbuilds(
    AdamW,
    lr=0.001,
    weight_decay=0.01,
)
optim_store(AdamWConf, name="adamw")
