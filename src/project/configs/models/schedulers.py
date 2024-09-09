from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau

from project.configs.utils import ZENSTORE, pfbuilds

sched_store = ZENSTORE(group="model/scheduler")

ReduceLROnPlateauConf = pfbuilds(
    ReduceLROnPlateau,
    mode="min",
    factor=0.5,
    patience=10,
    min_lr=1e-5,
)
sched_store(ReduceLROnPlateauConf, name="plateau")


CyclicLRConf = pfbuilds(
    CyclicLR,
    mode="triangular2",
    base_lr=1e-5,
    max_lr=1e-2,
    step_size_up=2000,
    cycle_momentum=False,
)
sched_store(CyclicLRConf, name="cyclic")
