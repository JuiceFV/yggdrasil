from yggdrasil.callbacks.logging import StepLoggingCallback
from yggdrasil.configs.utils import CB_STORE, fbuilds

StepLoggingCallbackConf = fbuilds(StepLoggingCallback)

logging_conf = {
    "step_logging": StepLoggingCallbackConf,
}


def register_logging_callbacks() -> None:
    CB_STORE(logging_conf, name="logging")
