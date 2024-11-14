import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

import lightning as L
import torch
from hydra_zen import zen
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig

from .extra import enforce_tags, print_config_tree
from .logging import init_logger

log = init_logger(__name__)

_T = TypeVar("_T")


def setup(zen_cfg: DictConfig) -> None:
    r"""
    Applies optional utilities before the task is started.

    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing

    Args:
        zen_cfg (DictConfig): Hydra config object.
    """

    if zen_cfg.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    if zen_cfg.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.enforce_tags=True>")
        enforce_tags(zen_cfg, save_to_file=True)

    if zen_cfg.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.print_config=True>")
        print_config_tree(zen_cfg, resolve=True, save_to_file=True)


# Pre-callbacks that executed before the training function
PRE_CALLS: list[Callable] = [
    zen(lambda seed: L.seed_everything(seed, workers=True)),
    zen(setup),
]


def log_instantiation(obj: _T) -> _T:
    """
    Hook to log the instantiation of each object by `hydra_zen`
    """
    log.info(f"Instantiating\t{obj.__name__}")  # type: ignore
    return obj


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that wraps the task function in extra utilities.

    Makes multirun more resistant to failure.

    Utilities:
    - Logging the exception if occurs
    - Logging the output dir
    """

    @wraps(task_func)
    def wrap(*args: Any, **kwargs) -> dict[str, Any]:
        try:
            metric_dict = task_func(*args, **kwargs)

        except Exception:
            log.exception("Run failed")

            # when using hydra plugins like Optuna, you might want to disable raising
            # exception to avoid multirun failure
            raise

        finally:
            try:
                outdir = kwargs["zen_cfg"].paths.output_dir
                log.info(f"Output dir: {outdir}")
            except KeyError:
                pass
            # you might want to explicitly close loggers here in case of failure

        return metric_dict

    return wrap


@rank_zero_only
def log_hyperparameters(
    trainer: L.Trainer, model: L.LightningModule, cfg: DictConfig
) -> None:
    """
    Logs configuration to each logger.

    Additionally saves:
    - Number of model parameters
    """

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    log.info("Logging hyperparameters!")
    hparams = {}

    for k, v in cfg.items():
        if isinstance(k, str) and not k.startswith("_"):
            hparams[k] = v

    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)


def get_metric_value(
    metric_dict: dict[str, torch.Tensor], metric_name: str
) -> float | None:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        msg = (
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!"
        )

        raise KeyError(msg)

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value
