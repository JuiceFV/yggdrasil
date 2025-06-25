from hydra_zen import make_config
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryCalibrationError,
    BinaryF1Score,
)

from yggdrasil.configs.builder import BaseRunCfg
from yggdrasil.configs.callbacks.metrics import MetricCollectionConf
from yggdrasil.configs.utils import EXP_STORE, fbuilds
from yggdrasil.data.data_extractor.example import ExampleDataExtractor

# `_global_` level config requires for overwrites in the defaults list
# to be prefixed with `/`
# `"_self_"` is a keyword refering to the current config and its position determines
# the priority of the overrides in the `hydra_defaults` list
# https://hydra.cc/docs/advanced/defaults_list/
DEEP_EXAMPLE_EXP_CONF = make_config(
    hydra_defaults=[
        {"override /datamodule": "example"},
        {"override /model/net": "bin-mlp"},
        {"override /loggers": "mlflow"},
        {"override /callbacks": ["default", "metrics"]},
        "_self_",
    ],
    datamodule=dict(
        input_table_spec=dict(
            table_identifier="demo_processed",
            train_table_sample=80.0,
            eval_table_sample=10.0,
            test_table_sample=10.0,
        ),
        # TODO: Replace with config override
        data_extractor=fbuilds(ExampleDataExtractor),
        features_preprocessing_options=dict(nsamples=100),
    ),
    model=dict(
        net=dict(
            input_dim=64,
            num_layers=6,
            hidden_dim=64,
        )
    ),
    trainer=dict(max_epochs=5),
    callbacks={
        "metrics": dict(
            train_metrics=MetricCollectionConf(
                metrics=[
                    fbuilds(BinaryAccuracy),
                    fbuilds(BinaryF1Score),
                    fbuilds(BinaryAUROC),
                    fbuilds(BinaryCalibrationError),
                ],
                prefix="train_",
            ),
        ),
        # TODO: Come up with more generic names for max metrics
        "max_metrics": dict(
            metric_names=[
                "val_BinaryAccuracy",
                "val_BinaryF1Score",
                "val_BinaryAUROC",
                "val_BinaryCalibrationError",
            ]
        ),
    },
    loggers={"mlflow": dict(experiment_name="/Shared/bin-class-example")},
    bases=(BaseRunCfg,),
)


WIDE_EXAMPLE_EXP_CONF = make_config(
    hydra_defaults=[
        {"override /datamodule": "example"},
        {"override /model/net": "bin-mlp"},
        {"override /loggers": "mlflow"},
        "_self_",
    ],
    model=dict(
        net=dict(
            input_dim=64,
            num_layers=2,
            hidden_dim=256,
        )
    ),
    bases=(DEEP_EXAMPLE_EXP_CONF,),
)


def register_example_experiments() -> None:
    r"""
    Register example experiments configurations in the experiment store.
    """
    EXP_STORE(DEEP_EXAMPLE_EXP_CONF, name="example-deep")
    EXP_STORE(WIDE_EXAMPLE_EXP_CONF, name="example-wide")
