from hydra_zen import make_config
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCalibrationError,
    MulticlassF1Score,
)

from yggdrasil.configs.builder import BaseRunCfg
from yggdrasil.configs.callbacks.metrics import MetricCollectionConf
from yggdrasil.configs.utils import EXP_STORE, fbuilds

MNIST_MLP_EXP_CONF = make_config(
    hydra_defaults=[
        {"override /datamodule": "mnist"},
        {"override /model": "multi_classifier"},
        {"override /model/net": "multiclass-mlp"},
        {"override /loggers": "mlflow"},
        {"override /callbacks": ["default", "metrics"]},
        "_self_",
    ],
    datamodule=dict(
        input_table_spec=dict(
            table_identifier="mnist_784",
            train_table_sample=80.0,
            eval_table_sample=10.0,
            test_table_sample=10.0,
        ),
        features_preprocessing_options=dict(nsamples=100),
    ),
    model=dict(
        net=dict(
            input_dim=2008,
            output_dim=10,
            num_layers=10,
            hidden_dim=512,
        )
    ),
    trainer=dict(max_epochs=30),
    callbacks={
        "metrics": dict(
            train_metrics=MetricCollectionConf(
                metrics=[
                    fbuilds(MulticlassAccuracy, num_classes=10),
                    fbuilds(MulticlassAUROC, num_classes=10),
                    fbuilds(MulticlassF1Score, num_classes=10),
                    fbuilds(MulticlassCalibrationError, num_classes=10),
                ],
                prefix="train_",
            ),
        ),
        # TODO: Come up with more generic names for max metrics
        "max_metrics": dict(
            metric_names=[
                "val_MulticlassAccuracy",
                "val_MulticlassAUROC",
                "val_MulticlassF1Score",
                "val_MulticlassCalibrationError",
            ]
        ),
    },
    loggers={"mlflow": dict(experiment_name="/Shared/mnist-mlp-multiclassification")},
    bases=(BaseRunCfg,),
)


def register_mnist_experiments() -> None:
    r"""
    Register example experiments configurations in the experiment store.
    """
    EXP_STORE(MNIST_MLP_EXP_CONF, name="mnist-mlp-multiclassification")
