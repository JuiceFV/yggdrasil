# sink for all config groups to make them
# automatically register on import of the project

import yggdrasil.configs.callbacks
import yggdrasil.configs.datamodules
import yggdrasil.configs.debug
import yggdrasil.configs.experiments
import yggdrasil.configs.loggers
import yggdrasil.configs.models
import yggdrasil.configs.models.criterions
import yggdrasil.configs.models.optimizers
import yggdrasil.configs.models.schedulers
import yggdrasil.configs.nets
import yggdrasil.configs.trainer

from .builder import BaseRunCfg
