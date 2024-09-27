from hydra_zen import zen
from omegaconf import DictConfig

from project.configs.utils import ZENSTORE
from project.main import RunCfg, main
from project.utils import init_logger
from project.utils.boilerplate import PRE_CALLS, log_instantiation

log = init_logger(__name__)

# decorator to create a hydra context for the `main` function
run = zen(
    main,
    pre_call=PRE_CALLS,
    instantiation_wrapper=log_instantiation,
    unpack_kwargs=True,
)


def register_config(cfg: DictConfig) -> None:
    ZENSTORE(cfg, name="runner")
    # offload all registered configs to the original hydra store
    ZENSTORE.add_to_hydra_store()


if __name__ == "__main__":
    register_config(RunCfg)

    run.hydra_main(
        config_name="runner",
        version_base="1.3",
        config_path=None,
    )
