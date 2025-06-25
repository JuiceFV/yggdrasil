from yggdrasil.configs.callbacks.base import register_base_callbacks
from yggdrasil.configs.callbacks.logging import register_logging_callbacks
from yggdrasil.configs.callbacks.metrics import register_metrics_callbacks

register_base_callbacks()
register_metrics_callbacks()
register_logging_callbacks()
