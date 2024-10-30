import warnings
from unittest.mock import MagicMock, patch

import lightning.pytorch as L
import pytest
import torch
import torchmetrics
import torchmetrics.classification
from torch import nn

from project.callbacks.metrics import MaxMetricCallback, MetricCallback
from project.core.dtypes import MetricInput
from project.core.dtypes.classification.base import BinaryPreprocessedInput


class TestMetricInitialization:
    def test_metric_callback_init_with_train_metrics(self) -> None:
        train_metrics = torchmetrics.MetricCollection(
            [torchmetrics.classification.BinaryAccuracy()], prefix="train_"
        )
        callback = MetricCallback(train_metrics=train_metrics)
        assert callback.train_metrics == train_metrics
        assert callback.val_metrics.prefix == "val_"
        assert callback.test_metrics.prefix == "test_"

    def test_metric_callback_init_with_val_metrics(self) -> None:
        val_metrics = torchmetrics.MetricCollection(
            [torchmetrics.classification.BinaryAccuracy()], prefix="val_"
        )
        callback = MetricCallback(val_metrics=val_metrics)
        assert callback.train_metrics.prefix == "train_"
        assert callback.val_metrics == val_metrics
        assert callback.test_metrics.prefix == "test_"

    def test_metric_callback_init_with_test_metrics(self) -> None:
        test_metrics = torchmetrics.MetricCollection(
            [torchmetrics.classification.BinaryAccuracy()], prefix="test_"
        )
        callback = MetricCallback(test_metrics=test_metrics)
        assert callback.train_metrics.prefix == "train_"
        assert callback.val_metrics.prefix == "val_"
        assert callback.test_metrics == test_metrics

    def test_metric_callback_init_without_any_metrics(self) -> None:
        with pytest.raises(
            ValueError, match="At least one of the metrics should be provided"
        ):
            MetricCallback()

    def test_metric_callback_init_with_all_metrics(self) -> None:
        train_metrics = torchmetrics.MetricCollection(
            [torchmetrics.classification.BinaryAccuracy()], prefix="train_"
        )
        val_metrics = torchmetrics.MetricCollection(
            [torchmetrics.classification.BinaryAccuracy()], prefix="val_"
        )
        test_metrics = torchmetrics.MetricCollection(
            [torchmetrics.classification.BinaryAccuracy()], prefix="test_"
        )
        callback = MetricCallback(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        assert callback.train_metrics == train_metrics
        assert callback.val_metrics == val_metrics
        assert callback.test_metrics == test_metrics


class TestMetricCallback:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        train_metrics = torchmetrics.MetricCollection(
            [torchmetrics.classification.BinaryAccuracy()],
            prefix="train_",
        )
        val_metrics = train_metrics.clone(prefix="val_")
        test_metrics = train_metrics.clone(prefix="test_")

        self.callback = MetricCallback(
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        self.pl_module = MagicMock(spec=L.LightningModule)

    def test_initialization(self) -> None:
        assert isinstance(self.callback.train_metrics, torchmetrics.MetricCollection)
        assert isinstance(self.callback.val_metrics, torchmetrics.MetricCollection)
        assert isinstance(self.callback.test_metrics, torchmetrics.MetricCollection)

    def test_setup_fit(self) -> None:
        self.callback.setup(trainer=MagicMock(), pl_module=self.pl_module, stage="fit")
        assert self.pl_module.train_metrics is self.callback.train_metrics
        assert self.pl_module.val_metrics is self.callback.val_metrics

    def test_setup_validate(self) -> None:
        self.callback.setup(
            trainer=MagicMock(), pl_module=self.pl_module, stage="validate"
        )
        assert self.pl_module.val_metrics is self.callback.val_metrics

    def test_setup_test(self) -> None:
        self.callback.setup(trainer=MagicMock(), pl_module=self.pl_module, stage="test")
        assert self.pl_module.test_metrics is self.callback.test_metrics

    @patch.object(MetricCallback, "_log_metrics")
    def test_on_train_batch_end(self, mock_log_metrics: MagicMock) -> None:
        batch = BinaryPreprocessedInput.from_tensors(
            target=torch.tensor([1, 0, 1]),
            features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )
        outputs = {
            "metric_input": MetricInput(
                preds=torch.tensor([0.8, 0.3, 0.6]),
                target=batch.target,  # They are randmom (placeholder)
            )
        }
        self.callback.setup(trainer=MagicMock(), pl_module=self.pl_module, stage="fit")
        self.callback.on_train_batch_end(
            trainer=MagicMock(),
            pl_module=self.pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=0,
        )
        mock_log_metrics.assert_called_once_with(
            self.pl_module, outputs, batch, self.callback.train_metrics
        )

    @patch.object(MetricCallback, "_log_metrics")
    def test_on_validation_batch_end(self, mock_log_metrics: MagicMock) -> None:
        batch = BinaryPreprocessedInput.from_tensors(
            target=torch.tensor([1, 0, 1]),
            features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )
        outputs = {
            "metric_input": MetricInput(
                preds=torch.tensor([0.8, 0.3, 0.6]),
                target=batch.target,  # They are randmom (placeholder)
            )
        }
        self.callback.setup(
            trainer=MagicMock(), pl_module=self.pl_module, stage="validate"
        )
        self.callback.on_validation_batch_end(
            trainer=MagicMock(),
            pl_module=self.pl_module,
            outputs=outputs,
            batch=batch,
            batch_idx=0,
        )
        mock_log_metrics.assert_called_once_with(
            self.pl_module, outputs, batch, self.callback.val_metrics
        )

    def test_log_metrics_invalid_output_type(self) -> None:
        batch = BinaryPreprocessedInput.from_tensors(
            target=torch.tensor([1, 0, 1]),
            features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )
        with pytest.raises(TypeError, match="Outpu expected Mapping"):
            self.callback._log_metrics(
                pl_module=self.pl_module,
                outputs=torch.tensor([0.8, 0.3, 0.6]),  # Invalid output type
                batch=batch,
                metrics=self.callback.train_metrics,
            )

    def test_log_metrics_missing_out_key(self) -> None:
        batch = BinaryPreprocessedInput.from_tensors(
            target=torch.tensor([1, 0, 1]),
            features=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        )
        with pytest.raises(KeyError, match="Expected 'metric_input' in outputs"):
            self.callback._log_metrics(
                pl_module=self.pl_module,
                outputs={"loss": torch.tensor(0.5)},  # Missing 'metric_input' key
                batch=batch,
                metrics=self.callback.train_metrics,
            )


class TestMaxMetricCallback:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        metric_names = ["accuracy", "f1_score"]
        self.callback = MaxMetricCallback(metric_names=metric_names)
        self.pl_module = MagicMock(spec=L.LightningModule)
        self.trainer = MagicMock(spec=L.Trainer)
        self.trainer.callback_metrics = {
            "accuracy": torch.tensor(0.85),
            "f1_score": torch.tensor(0.75),
        }

    def test_initialization(self) -> None:
        assert isinstance(self.callback.max_metrics, nn.ModuleDict)
        assert "max_accuracy" in self.callback.max_metrics
        assert "max_f1_score" in self.callback.max_metrics
        assert isinstance(
            self.callback.max_metrics["max_accuracy"], torchmetrics.MaxMetric
        )
        assert isinstance(
            self.callback.max_metrics["max_f1_score"], torchmetrics.MaxMetric
        )

    def test_on_validation_epoch_end_update(self) -> None:
        with (
            patch.object(
                self.callback.max_metrics["max_accuracy"],
                "update",
                wraps=self.callback.max_metrics["max_accuracy"].update,
            ) as mock_update_acc,
            patch.object(
                self.callback.max_metrics["max_f1_score"],
                "update",
                wraps=self.callback.max_metrics["max_f1_score"].update,
            ) as mock_update_f1,
        ):
            self.callback.on_validation_epoch_end(
                trainer=self.trainer, pl_module=self.pl_module
            )

            mock_update_acc.assert_called_once_with(
                self.trainer.callback_metrics["accuracy"]
            )
            mock_update_f1.assert_called_once_with(
                self.trainer.callback_metrics["f1_score"]
            )

    def test_on_validation_epoch_end_missing_metric(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with warnings.catch_warnings():
            # Ignore UserWarning from torchmetrics (compute before update)
            warnings.simplefilter("ignore", UserWarning)
            self.trainer.callback_metrics = {"non_existent_metric": torch.tensor(0.5)}
            with caplog.at_level("WARNING"):
                self.callback.on_validation_epoch_end(
                    trainer=self.trainer, pl_module=self.pl_module
                )
            assert "Metric max_accuracy not found in callback_metrics" in caplog.text
            assert "Metric max_f1_score not found in callback_metrics" in caplog.text

    @patch.object(torchmetrics.MaxMetric, "compute")
    def test_on_validation_epoch_end_logging(self, mock_compute: MagicMock) -> None:
        mock_compute.side_effect = [torch.tensor(0.85), torch.tensor(0.75)]
        self.callback.on_validation_epoch_end(
            trainer=self.trainer, pl_module=self.pl_module
        )

        expected_log_dict = {
            "max_accuracy": torch.tensor(0.85),
            "max_f1_score": torch.tensor(0.75),
        }
        self.pl_module.log_dict.assert_called_once_with(
            expected_log_dict, sync_dist=True
        )
