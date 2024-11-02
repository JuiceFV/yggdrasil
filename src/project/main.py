import os
from typing import Any

import torch
from hydra_zen import make_config
from lightning import LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig

from project.configs import BaseRunCfg
from project.configs.utils import fbuilds
from project.utils import (
    get_metric_value,
    init_logger,
    log_hyperparameters,
    task_wrapper,
)
from project.utils.dbx import propagate_credentials

log = init_logger(__name__)


def train_func(
    datamodule: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    train: bool = True,
    test: bool = False,
    ckpt_path: str | None = None,
    cfg: DictConfig | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset,
    using best weights obtained during training.

    Args:
        datamodule (LightningDataModule): DataModule object.
        model (LightningModule): Model to train.
        trainer (Trainer): Trainer object.
        train (bool, optional): Whether to train the model. Defaults to True.
        test (bool, optional): Whether to test the model. Defaults to False.
        ckpt_path (str, optional): Path to the checkpoint. Defaults to None.
        cfg (DictConfig, optional): raw Hydra config. Defaults to None.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    object_dict = {
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    if cfg is not None:
        log_hyperparameters(trainer, model, cfg)

    if train:
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    train_metrics = trainer.callback_metrics

    # TODO: support loading checkpoint if trained before or loading from path if not
    if test:
        log.info("Starting testing!")
        ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", "")
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


_DBX_CREDS = propagate_credentials()


@task_wrapper
def main(
    datamodule: LightningDataModule,
    model: LightningModule,
    trainer: Trainer,
    train: bool = True,
    test: bool = False,
    ckpt_path: str | None = None,
    optimized_metric: str | None = None,
    zen_cfg: DictConfig | None = None,  # stores full resolved hydra config
) -> float | None:
    r"""
    Main function which is a wrapper around the training function
    with additional utilities before and after the training

    Parameters
    ----------
    datamodule : LightningDataModule
        _description_
    model : LightningModule
        _description_
    trainer : Trainer
        _description_
    train : bool, optional
        _description_, by default True
    test : bool, optional
        _description_, by default False
    ckpt_path : str | None, optional
        _description_, by default None
    optimized_metric : str | None, optional
        _description_, by default None
    zen_cfg : DictConfig | None, optional
        _description_, by default None

    Returns
    -------
    float | None
        _description_
    """
    if _DBX_CREDS:
        os.environ.update(_DBX_CREDS)

    metric_dict, _ = train_func(
        model=model,
        trainer=trainer,
        datamodule=datamodule,
        train=train,
        test=test,
        ckpt_path=ckpt_path,
        cfg=zen_cfg,
    )

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = None
    if optimized_metric is not None:
        metric_value = get_metric_value(
            metric_dict=metric_dict, metric_name=optimized_metric
        )
    return metric_value


# creates the final train config from the default presets
# in `builder.py` and signature of the `main` function
TrainCfg = fbuilds(main)
RunCfg = make_config(bases=(TrainCfg, BaseRunCfg))
