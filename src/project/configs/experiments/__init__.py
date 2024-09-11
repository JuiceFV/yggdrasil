from hydra_zen import make_config

from project.configs.builder import BaseRunCfg
from project.configs.utils import ZENSTORE

# substore for experiments configs inside general store, `package="_global_"` is
# required here to be able to overwrite all config groups in the store, not just local
EXP_STORE = ZENSTORE(group="experiment", package="_global_")

# `_global_` level config requires for overwrites in the defaults list
# to be prefixed with `/`
# `"_self_"` is a keyword refering to the current config and its position determines
# the priority of the overrides in the `hydra_defaults` list
# https://hydra.cc/docs/advanced/defaults_list/
DEEP_EXAMPLE_EXP_CONF = make_config(
    hydra_defaults=[
        {"override /datamodule": "example"},
        {"override /model/net": "deep-mlp"},
        {"override /loggers": "mlflow"},
        "_self_",
    ],
    datamodule=dict(dataset=dict(num_features=64)),
    model=dict(net=dict(input_dim=64)),
    trainer=dict(max_epochs=5),
    loggers={"mlflow": dict(experiment_name="bin-class-example")},
    bases=(BaseRunCfg,),
)


WIDE_EXAMPLE_EXP_CONF = make_config(
    hydra_defaults=[
        {"override /datamodule": "example"},
        {"override /model/net": "wide-mlp"},
        {"override /loggers": "mlflow"},
        "_self_",
    ],
    bases=(DEEP_EXAMPLE_EXP_CONF,),
)

EXP_STORE(DEEP_EXAMPLE_EXP_CONF, name="example-deep")
EXP_STORE(WIDE_EXAMPLE_EXP_CONF, name="example-wide")
