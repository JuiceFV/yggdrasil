from torch import nn

from project.configs.utils import ZENSTORE, fbuilds

crit_store = ZENSTORE(group="model/criterion")

BCELogitsLossConf = fbuilds(nn.BCEWithLogitsLoss)
crit_store(BCELogitsLossConf, name="bce_logits")
