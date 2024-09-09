from lightning import Trainer

from .utils import ZENSTORE, builds

trainer_store = ZENSTORE(group="trainer")

DefTrainerConf = builds(
    Trainer,
    callbacks="${oc.dict.values: callbacks}",
    logger="${oc.dict.values: loggers}",
    default_root_dir="${paths.output_dir}",
    min_epochs=1,  # prevents early stopping
    max_epochs=100,
    log_every_n_steps=10,
    accelerator="auto",
    devices=1,
    check_val_every_n_epoch=1,
    enable_progress_bar=True,
    deterministic=False,
)

trainer_store(DefTrainerConf, name="default")
