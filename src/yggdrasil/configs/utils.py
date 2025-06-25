# create multiple build functions for dataclasses with unified presets
# `kw_only=True` argument is required to avoid issues with order of positional arguments
#  in generated dataclasses with inheritance and `hydra_convert="object"` is required to
#  recursively instantiate objects with `hydra` to native python objects
# https://hydra.cc/docs/advanced/instantiate_objects/overview/#parameter-conversion-strategies
from collections.abc import Callable
from dataclasses import is_dataclass

from hydra_zen import ZenStore, make_custom_builds_fn
from hydra_zen.typing._implementations import DataClass_
from hydra_zen.wrapper import default_to_config
from omegaconf import DictConfig, ListConfig, OmegaConf

builds: Callable = make_custom_builds_fn(
    zen_dataclass={"kw_only": True},
    hydra_convert="object",
)


fbuilds: Callable = make_custom_builds_fn(
    populate_full_signature=True,
    zen_dataclass={"kw_only": True},
    hydra_convert="object",
)


pfbuilds: Callable = make_custom_builds_fn(
    zen_partial=True,
    populate_full_signature=True,
    zen_dataclass={"kw_only": True},
    hydra_convert="object",
)


Config_ = DataClass_ | type[DataClass_] | ListConfig | DictConfig


def destructure(x: Config_) -> Config_:
    """Disables `hydra` config type checking.

    See `https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621#discussioncomment-7938326`
    """
    # apply the default auto-config logic of `store`
    x = default_to_config(x)
    if is_dataclass(x):
        # Recursively converts:
        # dataclass -> omegaconf-dict (backed by dataclass types)
        #           -> dict -> omegaconf dict (no types)
        # Omegaconf handles structured configs but this is not reflected in typing
        return OmegaConf.create(OmegaConf.to_container(OmegaConf.create(x)))  # type: ignore
    return x


# general store for configs of all user-defined objects
ZENSTORE = ZenStore(name="zenstore", overwrite_ok=True)(to_config=destructure)

# store for hydra-specific configs
HYDRASTORE = ZenStore(name="hydrastore", overwrite_ok=True, deferred_hydra_store=False)

# substore for NN nets configs inside default storage
NET_STORE = ZENSTORE(group="model/net")

# Temporary solution for callbacks storage.
CB_STORE = ZENSTORE(group="callbacks")

# substore for experiments configs inside general store, `package="_global_"` is
# required here to be able to overwrite all config groups in the store, not just local
EXP_STORE = ZENSTORE(group="experiment", package="_global_")
