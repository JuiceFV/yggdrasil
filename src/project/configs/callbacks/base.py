from hydra_zen import make_config
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from omegaconf import MISSING

from project.callbacks.logging import (
    MLFlowModelRegistryHook,
    SummaryLogger,
    TimingCallback,
)
from project.configs.callbacks.logging import StepLoggingCallbackConf
from project.configs.utils import CB_STORE, fbuilds

RichProgressBarConf = fbuilds(RichProgressBar)
rich_pb_conf = {"richpb": RichProgressBarConf}

ModelSummaryConf = fbuilds(RichModelSummary, max_depth=-1)
model_summary_conf = {"model_summ": ModelSummaryConf}

TimingCallbackConf = fbuilds(TimingCallback)
timing_cb_conf = {"timer": TimingCallbackConf}

EarlyStoppingConf = fbuilds(
    EarlyStopping,
    monitor=MISSING,
    min_delta=0.0,
    patience=3,
    verbose=True,
    mode="min",
    strict=True,
    check_finite=True,
    stopping_threshold=None,
    divergence_threshold=None,
    check_on_train_epoch_end=None,
    log_rank_zero_only=True,
)
early_stopping_conf = {"estopping": EarlyStoppingConf}

CheckpointerConf = fbuilds(
    ModelCheckpoint,
    dirpath=None,
    filename=None,
    monitor=None,
    verbose=True,
    save_last=None,
    save_top_k=1,
    mode="min",
    auto_insert_metric_name=False,
    save_weights_only=False,
    save_on_train_epoch_end=None,
)
BestCheckpointerConf = CheckpointerConf(
    filename="best_checkpoint",
    monitor="val_loss_epoch",
    mode="min",
    save_on_train_epoch_end=True,
)
best_ckpt_conf = {"best_ckpt": BestCheckpointerConf}

LastCheckpointerConf = CheckpointerConf(
    filename="last",
    monitor="global_step",
    mode="max",
    save_on_train_epoch_end=False,
)
last_ckpt_conf = {"last_ckpt": LastCheckpointerConf}
ckpt_callbacks = dict(**best_ckpt_conf, **last_ckpt_conf)

LRMonitorConf = fbuilds(
    LearningRateMonitor,
    logging_interval="step",
    log_momentum=True,
)
lr_monitor_conf = {"lr_monitor": LRMonitorConf}


DefCallbacksConf = make_config(
    **model_summary_conf,
    **rich_pb_conf,
)

MLFlowCallbacksCfg = make_config(
    summ_logger=fbuilds(SummaryLogger),
    registry_hook=fbuilds(MLFlowModelRegistryHook),
    step_logging=StepLoggingCallbackConf,  # required for logging
    **timing_cb_conf,
    **lr_monitor_conf,
    **ckpt_callbacks,
)


def register_base_callbacks() -> None:
    CB_STORE(model_summary_conf, name="model_summ")
    CB_STORE(timing_cb_conf, name="timer")
    CB_STORE(early_stopping_conf, name="estopping")
    CB_STORE(rich_pb_conf, name="richpb")
    CB_STORE(best_ckpt_conf, name="best_ckpt")
    CB_STORE(last_ckpt_conf, name="last_ckpt")
    CB_STORE(ckpt_callbacks, name="ckpts")
    CB_STORE(lr_monitor_conf, name="lr_monitor")
    CB_STORE(DefCallbacksConf, name="default")
