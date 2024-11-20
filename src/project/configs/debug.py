from hydra_zen import make_config
from hydra_zen.typing._implementations import DefaultsList
from lightning.pytorch.profilers import SimpleProfiler

from .builder import BaseRunCfg
from .utils import ZENSTORE, fbuilds

debug_store = ZENSTORE(group="debug", package="_global_")

_DEBUG_DEFAULTS: DefaultsList = [
    "_self_",
    {"override /callbacks": ["model_summ", "richpb"]},
]

# default debug configuration
DefDebugCfg = make_config(
    hydra_defaults=_DEBUG_DEFAULTS,
    task_name="debug",
    loggers={},
    ignore_warnings=False,
    enforce_tags=False,
    print_config=False,
    trainer=dict(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        detect_anomaly=True,  # detect NaNs in the model
        enable_checkpointing=False,
    ),
    datamodule=dict(
        num_workers=0,  # avoids issues with the debugger in distributed setting
        pin_memory=False,
    ),
    hydra=dict(job_logging=dict(root={"level": "DEBUG"}), verbose=True),
    bases=(BaseRunCfg,),
)

debug_store(DefDebugCfg, name="default")

# fast dev run config for single step of each phase: train, val, test
FastDevRunCfg = make_config(
    hydra_defaults=_DEBUG_DEFAULTS,
    trainer=dict(
        fast_dev_run=True,
        enable_checkpointing=False,
    ),
    bases=(DefDebugCfg,),
)

debug_store(FastDevRunCfg, name="fdr")

LimitBatchesCfg = make_config(
    hydra_defaults=_DEBUG_DEFAULTS,
    trainer=dict(
        max_epochs=3,
        limit_train_batches=3,
        limit_val_batches=3,
        limit_test_batches=3,
        enable_checkpointing=False,
    ),
    bases=(DefDebugCfg,),
)

debug_store(LimitBatchesCfg, name="limit")


OverfitCfg = make_config(
    trainer=dict(
        max_epochs=20,
        overfit_batches=3,
        enable_checkpointing=False,
    ),
    bases=(DefDebugCfg,),
)

debug_store(OverfitCfg, name="overfit")

ProfilerCfg = make_config(
    trainer=dict(
        max_epochs=1,
        profiler=fbuilds(SimpleProfiler, filename="profile"),
    )
)

debug_store(ProfilerCfg, name="profiler")
