from dataclasses import is_dataclass

from hydra_zen import ZenStore, make_custom_builds_fn
from hydra_zen.wrapper import default_to_config
from omegaconf import OmegaConf

builds = make_custom_builds_fn(
    # zen_dataclass=dict(kw_only=True),
    hydra_convert="object",
)
fbuilds = make_custom_builds_fn(
    populate_full_signature=True,
    zen_dataclass=dict(kw_only=True),
    hydra_convert="object",
)
pfbuilds = make_custom_builds_fn(
    zen_partial=True,
    populate_full_signature=True,
    zen_dataclass=dict(kw_only=True),
    hydra_convert="object",
)


def destructure(x):
    """Disables `hydra` config type checking.

    See `https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/621#discussioncomment-7938326`
    """
    x = default_to_config(x)  # apply the default auto-config logic of `store`
    if is_dataclass(x):
        # Recursively converts:
        # dataclass -> omegaconf-dict (backed by dataclass types)
        #           -> dict -> omegaconf dict (no types)
        return OmegaConf.create(OmegaConf.to_container(OmegaConf.create(x)))
    return x


ZENSTORE = ZenStore(name="zenstore")(to_config=destructure)
HYDRASTORE = ZenStore(name="hydrastore", deferred_hydra_store=False)

NET_STORE = ZENSTORE(group="model/net")
