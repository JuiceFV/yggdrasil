from hydra_zen import ZenStore, zen
from hydra_zen.typing._implementations import DataClass

from yggdrasil.configs.utils import ZENSTORE
from yggdrasil.main import RunCfg, main
from yggdrasil.utils import init_logger
from yggdrasil.utils.boilerplate import PRE_CALLS, log_instantiation

log = init_logger(__name__)

# decorator to create a hydra context for the `main` function
run = zen(
    main,
    pre_call=PRE_CALLS,
    instantiation_wrapper=log_instantiation,
    unpack_kwargs=True,
)


def register_config(cfg: type[DataClass], store: ZenStore) -> None:
    store(cfg, name="runner")
    # offload all registered configs to the original hydra store
    store.add_to_hydra_store()


if __name__ == "__main__":
    register_config(RunCfg, ZENSTORE)

    run.hydra_main(
        config_name="runner",
        version_base="1.3",
        config_path=None,
    )
