from omegaconf import MISSING
from torchmetrics import MetricCollection

from yggdrasil.callbacks.metrics import MaxMetricCallback, MetricCallback
from yggdrasil.configs.utils import CB_STORE, fbuilds

MetricCollectionConf = fbuilds(MetricCollection, metrics=MISSING, prefix=MISSING)
MetricCallbackConf = fbuilds(MetricCallback, train_metrics=MISSING)
MaxMetricCallbackConf = fbuilds(MaxMetricCallback, metric_names=MISSING)

metrics_conf = {
    "metrics": MetricCallbackConf,
    "max_metrics": MaxMetricCallbackConf,
}


def register_metrics_callbacks() -> None:
    CB_STORE(metrics_conf, name="metrics")
