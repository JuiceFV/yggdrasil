from hydra_zen import make_config
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau

from project.configs.utils import ZENSTORE, pfbuilds

sched_store = ZENSTORE(group="model/scheduler", package="_global_")

ReduceLROnPlateauConf = pfbuilds(
    ReduceLROnPlateau,
    mode="min",
    factor=0.5,
    patience=10,
    min_lr=1e-5,
)

plateau_conf = make_config(
    model=dict(
        scheduler=ReduceLROnPlateauConf,
        lr_scheduler_config=dict(
            monitor="val_loss_epoch", interval="epoch", frequency=1
        ),
    ),
)

sched_store(plateau_conf, name="plateau")


CyclicLRConf = pfbuilds(
    CyclicLR,
    mode="triangular2",
    base_lr=1e-5,
    max_lr=1e-2,
    step_size_up=2000,
    cycle_momentum=False,
)
cyclic_conf = make_config(model=dict(scheduler=CyclicLRConf))
sched_store(cyclic_conf, name="cyclic")
