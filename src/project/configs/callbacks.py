from hydra_zen import make_config
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from omegaconf import MISSING

from project.utils.callbacks import (
    MLFlowModelRegistryHook,
    SummaryLogger,
    TimingCallback,
)

from .utils import ZENSTORE, fbuilds

cb_store = ZENSTORE(group="callbacks")

RichProgressBarConf = fbuilds(RichProgressBar)
rich_pb_conf = {"richpb": RichProgressBarConf}
cb_store(rich_pb_conf, name="richpb")

ModelSummaryConf = fbuilds(RichModelSummary, max_depth=-1)
model_summary_conf = {"model_summ": ModelSummaryConf}
cb_store(model_summary_conf, name="model_summ")


LRMonitorConf = fbuilds(
    LearningRateMonitor,
    logging_interval="step",
    log_momentum=True,
)

lr_monitor_conf = {"lr_monitor": LRMonitorConf}
cb_store(lr_monitor_conf, name="lr_monitor")

TimingCallbackConf = fbuilds(TimingCallback)
timing_cb_conf = {"timer": TimingCallbackConf}
cb_store(timing_cb_conf, name="timer")

base_logging_callbacks = dict(**lr_monitor_conf, **timing_cb_conf)

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
cb_store(early_stopping_conf, name="estopping")

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
cb_store(best_ckpt_conf, name="best_ckpt")

LastCheckpointerConf = CheckpointerConf(
    filename="last",
    monitor="global_step",
    mode="max",
    save_on_train_epoch_end=False,
)
last_ckpt_conf = {"last_ckpt": LastCheckpointerConf}
cb_store(last_ckpt_conf, name="last_ckpt")

DefCallbacksConf = make_config(
    **model_summary_conf,
    **rich_pb_conf,
    **best_ckpt_conf,
    **last_ckpt_conf,
)
cb_store(DefCallbacksConf, name="default")

MLFlowCallbacksCfg = make_config(
    summ_logger=fbuilds(SummaryLogger),
    registry_hook=fbuilds(MLFlowModelRegistryHook),
    **base_logging_callbacks,
)
