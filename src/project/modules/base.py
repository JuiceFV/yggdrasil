import abc
import inspect
from collections.abc import Generator
from typing import Any, Generic, TypeVar, cast, final

import lightning as L
import torch
from lightning.fabric.wrappers import _FabricOptimizer
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim.optimizer import Optimizer

from project.core.dtypes.base import TensorDataClass
from project.core.utils import lazy_property
from project.utils.logging import init_logger

log = init_logger(__name__)

T = TypeVar("T", bound=TensorDataClass)

STEP_OUT_TYPE = STEP_OUTPUT | TensorDataClass
STEP_GEN_T = TypeVar("STEP_GEN_T", bound=STEP_OUT_TYPE)

Optimizers = Optimizer | LightningOptimizer | _FabricOptimizer
OptimizersList = (
    list[Optimizer]
    | list[LightningOptimizer]
    | list[_FabricOptimizer]
    | list[Optimizers]
)


class BaseModule(L.LightningModule, abc.ABC, Generic[T, STEP_GEN_T]):
    r"""
    Base class for all lightning modules.
    :class:`~lightning.pytorch.core.LightningModule` typically implements the
    optimization step of a model, automizing all the data-preprocessing and
    loss computation. To justify the model-flow and make it uniform for all
    the users the :class:`BaseModule` pre-defines the training process around
    the :class:`~project.core.dtypes.base.TensorDataClass` dataclass. To define
    a manual :class:`~lightning.pytorch.core.LightningModule` you have to implement:

    1. :meth:`train_step_gen` method is a generator that yields the training loss.
       Usful for non-composite models consisting of a single optimizer.
    2. :meth:`training_step` method is a wrapper around the generator.
       If you want to define a composite model with multiple optimizers, you
       have to re-implement the method manually.
    3. :meth:`validation_step` and :meth:`test_step` methods are optional.
       Implement them if your datamodule implements relevant dataloaders.
    4. :meth:`configure_optimizers` method is mandatory. Implement optimizer(s)
       and scheduler(s) initialization.

    Args:
        automatic_optimization (bool, optional): Whether to enable automatic
            optimization. Defaults to True.

    Example::

        class MyModel(BaseModule):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Linear(10, 1)
                self.save_hyperparameters()

            def configure_optimizers(self):
                return {
                    "optimizers": [torch.optim.Adam(self.model.parameters(), lr=1e-3)]
                }

            def train_step_gen(self, training_batch, batch_idx):
                output = self.model(training_batch)
                loss = torch.nn.functional.mse_loss(output, training_batch)
                yield loss

        model = MyModel()
        trainer = L.Trainer(max_epochs=5)
        trainer.fit(model, train_dataloader)
    """

    def __init__(self, automatic_optimization: bool = True) -> None:
        r"""Initializes the BaseModule class.

        Args:
            automatic_optimization (bool, optional): Whether to enable automatic
                optimization. Defaults to True.
        """
        super().__init__()
        self._automatic_optimization = automatic_optimization
        self._training_step_gen: Generator[STEP_GEN_T, None, None] | None = None
        self._verified_steps = False
        self._setup_input_type()
        self.train_batches_processed_this_epoch = 0
        self.val_batches_processed_this_epoch = 0
        self.test_batches_processed_this_epoch = 0
        self.all_batches_processed = 0

    def _setup_input_type(self) -> None:
        r"""
        Sets up the input type for the training step generator.

        Raises:
            RuntimeError: If the `train_step_gen` method does not have a
                `training_batch` parameter,
            this exception is raised to indicate that the training data's
                type could not be inferred.
        """
        self._training_batch_type: type[T] | None = None
        sig = inspect.signature(self.train_step_gen)
        if "training_batch" not in sig.parameters:
            msg = "Missing training data to infer its type."
            raise RuntimeError(msg)
        param = sig.parameters["training_batch"]
        annotation = param.annotation
        if annotation == inspect.Parameter.empty:
            return
        self._training_batch_type = annotation

    @abc.abstractmethod
    def train_step_gen(
        self, training_batch: T, batch_idx: int
    ) -> Generator[STEP_GEN_T, None, None]:
        r"""
        Generator for training steps. Should be implemented in subclasses.

        Args:
            training_batch (TensorDataClass): The batch of training data.
            batch_idx (int): The index of the current batch.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.

        Returns:
            Generator[STEP_GEN_T, None, None]: A generator for a training step output.
        """
        raise NotImplementedError

    def training_step(  # type: ignore
        self,
        batch: T | dict[str, torch.Tensor],
        batch_idx: int,
    ) -> STEP_GEN_T:
        r"""
        Executes a training step.

        Args:
            batch (TensorDataClass | dict[str, torch.Tensor]): The current batch of
                training data.
            batch_idx (int): The index of the current batch.

        Raises:
            TypeError: If the batch type does not match the expected `TensorDataClass`.

        Returns:
            STEP_GEN_T: The output of the training step.
        """
        if self._training_step_gen is None:
            if self._training_batch_type and isinstance(batch, dict):
                batch = self._training_batch_type(**batch)
            if not isinstance(batch, TensorDataClass):
                msg = f"Expected {self._training_batch_type} but got {type(batch)}"
                raise TypeError(msg)
            self._training_step_gen = self.train_step_gen(cast(T, batch), batch_idx)

        output = next(self._training_step_gen)

        if not self._verified_steps:
            try:
                next(self._training_step_gen)
            except StopIteration:
                self._verified_steps = True
            if not self._verified_steps:
                msg = (
                    "The number of training steps should match the number "
                    f"of optimizers {self._num_opt_steps}"
                )
                raise RuntimeError(msg)
        self._training_step_gen = None
        return output

    def optimizers(self, use_pl_optimizer: bool = True) -> OptimizersList:  # type: ignore
        r"""
        Returns the optimizers used during training.

        Args:
            use_pl_optimizer (bool, optional): Whether to use PyTorch Lightning's
                optimizer handling. Defaults to True.

        Returns:
            list: A list of optimizers.
        """
        opt = super().optimizers(use_pl_optimizer)
        return opt if isinstance(opt, list) else [opt]

    @lazy_property
    def _num_opt_steps(self) -> int:
        r"""
        Returns the number of optimizer steps based on the number configured optimizers.

        Returns:
            int: The number of optimizers or 1 if no optimizers are defined.
        """
        optimizers = self.configure_optimizers()
        if isinstance(optimizers, list | tuple):
            return len(optimizers)
        return 1

    @final
    def on_train_epoch_end(self) -> None:
        r"""
        Called at the end of each training epoch. Resets the counter for processed
        training batches. If the current epoch matches the next stopping epoch, it
        triggers stopping the training.
        """
        log.info(
            f"Finished train epoch {self.current_epoch} "
            f"with {self.train_batches_processed_this_epoch} batches processed"
        )
        self.train_batches_processed_this_epoch = 0

    @final
    def on_validation_epoch_end(self) -> None:
        r"""
        Called at the end of each validation epoch. Resets the counter for processed
        validation batches.

        .. todo::
            Currently this method is not :func:`~typing.final` because it is overridden.
            The metrics are crucial part of the model trainig proces, so consider how to
            add them to the base module.
        """
        log.info(
            f"Finished validation epoch {self.current_epoch} "
            f"with {self.val_batches_processed_this_epoch} batches processed"
        )
        self.val_batches_processed_this_epoch = 0

    @final
    def on_test_epoch_end(self) -> None:
        r"""
        Called at the end of each test epoch. Resets the counter for processed test
        batches.
        """
        log.info(
            f"Finished test epoch {self.current_epoch} "
            f"with {self.test_batches_processed_this_epoch} batches processed"
        )
        self.test_batches_processed_this_epoch = 0

    @final
    def on_train_batch_end(self, *args: Any, **kwargs: Any) -> None:
        r"""
        Called at the end of each training batch. Increments the counters for
        processed training batches and the total number of processed batches.
        """
        self.train_batches_processed_this_epoch += 1
        self.all_batches_processed += 1

    @final
    def on_validation_batch_end(self, *args: Any, **kwargs: Any) -> None:
        r"""
        Called at the end of each validation batch. Increments the counters for
        processed validation batches and the total number of processed batches.
        """
        self.val_batches_processed_this_epoch += 1
        self.all_batches_processed += 1

    @final
    def on_test_batch_end(self, *args: Any, **kwargs: Any) -> None:
        r"""
        Called at the end of each test batch. Increments the counters for processed
        test batches and the total number of processed batches.
        """
        self.test_batches_processed_this_epoch += 1
        self.all_batches_processed += 1
