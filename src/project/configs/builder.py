from dataclasses import dataclass

from hydra.conf import HydraConf, RunDir, SweepDir
from hydra_plugins.hydra_ray_launcher._config import RayConf
from hydra_zen import make_config

from .utils import HYDRASTORE

# config for the `hydra` behavior
HydraOverwriteCfg = make_config(
    # specifies the naming patterns of the directories created by hydra for each launch
    run=RunDir(
        dir="${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}"
    ),
    sweep=SweepDir(
        dir="${paths.log_dir}/${task_name}/multiruns/${now:%Y-%m-%d}_${now:%H-%M-%S}",
        subdir="${hydra.job.num}",
    ),
    # we reuse all defaults lists from the hydra core except enabling color logging
    defaults=[
        {"output": "default"},
        {"launcher": "basic"},
        {"sweeper": "basic"},
        {"help": "default"},
        {"hydra_help": "default"},
        {"hydra_logging": "colorlog"},
        {"job_logging": "colorlog"},
        {"callbacks": None},
        "_self_",
    ],
    # config inherits from the default `hydra` config
    bases=(HydraConf,),
)

HYDRASTORE(HydraOverwriteCfg)

# config for the `ray` launcher connecting to an existing ray cluster on DBX
DBXRayLauncherCfg = RayConf(
    init={
        "address": None,
        "object_store_memory": None,  # required while connecting to existing cluster
    },
)
HYDRASTORE(DBXRayLauncherCfg, name="dbx_ray", group="hydra/launcher")


@dataclass
class PathsCfg:
    # path to the local logs directory
    log_dir: str = "${oc.env:LOGS_DIR}"
    # path to output directory, created dynamically by hydra
    # use it to store all files generated during the run, like ckpts and metrics
    output_dir: str = "${hydra:runtime.output_dir}"


# default config for the training run function
BaseRunCfg = make_config(
    hydra_defaults=[
        "_self_",
        {"trainer": "default"},
        {"loggers": None},
        {"callbacks": "default"},
        {"datamodule": "example"},
        {"model": "binary_classifier"},
        {"experiment": None},
        {"debug": None},
    ],
    callbacks={},
    loggers={},
    task_name="train",
    # tags to help you identify your experiments
    # you can overwrite this in experiment configs
    # overwrite from command line with `python train.py tags="[first_tag, second_tag]"`
    # appending lists from command line is currently not supported :(
    # https://github.com/facebookresearch/hydra/issues/1547
    tags=["dev"],
    train=True,  # set False to skip model training
    test=False,
    ckpt_path=None,  # path to the torch checkpoint to resume training from
    seed=42,
    paths=PathsCfg(),
    ignore_warnings=False,
    enforce_tags=True,
    print_config=True,
    hydra=dict(job=dict(name="training")),
)
