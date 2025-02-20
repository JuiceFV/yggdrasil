"""Dataclass generally implemented for convenient way
of type-class definition (including type checking).

NOTE: Hydra Zen only handles initialization params and classes.
I don't really trust the developers (namely data scientists)
so the following dataclass implementation should prevent a
developer from type mismatching.
"""

import dataclasses
import json
import logging
import os
from dataclasses import asdict, field, is_dataclass  # noqa: F401
from typing import TYPE_CHECKING, Any

import pydantic

USE_VANILLA_DATACLASS = bool(int(os.environ.get("USE_VANILLA_DATACLASS", False)))
ARBITRARY_TYPES_ALLOWED = bool(int(os.environ.get("ARBITRARY_TYPES_ALLOWED", True)))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

log.info(f"USE_VANILLA_DATACLASS: {USE_VANILLA_DATACLASS}")
log.info(f"ARBITRARY_TYPES_ALLOWED: {ARBITRARY_TYPES_ALLOWED}")

if TYPE_CHECKING:
    from dataclasses import dataclass
else:

    def dataclass(_cls: type[object] | None = None, **kwargs: Any):  # noqa: ANN201
        def wrap(cls):  # noqa: ANN001, ANN202
            if USE_VANILLA_DATACLASS:
                # For vanilla dataclasses, directly use the dataclasses decorator
                return dataclasses.dataclass(**kwargs)(cls)
            if ARBITRARY_TYPES_ALLOWED:
                if "config" in kwargs:
                    msg = "Config duplication occures"
                    raise KeyError(msg)
                kwargs["config"] = pydantic.ConfigDict(
                    arbitrary_types_allowed=ARBITRARY_TYPES_ALLOWED
                )

            return pydantic.dataclasses.dataclass(cls, **kwargs)

        if _cls is None:
            return wrap

        return wrap(_cls)


class DataclassJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, object) and is_dataclass(o):
            if isinstance(o, type):
                msg = "asdict() should be called on dataclass instances, not types"
                raise TypeError(msg)
            return asdict(o)
        return super().default(o)
