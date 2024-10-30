from project.configs.callbacks.base import register_base_callbacks
from project.configs.callbacks.logging import register_logging_callbacks
from project.configs.callbacks.metrics import register_metrics_callbacks

register_base_callbacks()
register_metrics_callbacks()
register_logging_callbacks()
