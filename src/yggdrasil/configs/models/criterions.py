from torch import nn

from yggdrasil.configs.utils import ZENSTORE, fbuilds

crit_store = ZENSTORE(group="model/criterion")

BCELogitsLossConf = fbuilds(nn.BCEWithLogitsLoss)
crit_store(BCELogitsLossConf, name="bce_logits")
