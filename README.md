# hydra-zen-dl-template [still WIP]

Template for organizing your DL project using `pytorch-lightning` with `hydra` and `hydra_zen`. The project is installable as a python package under the name `project` (which should be manually changed to the desired name) and can be easily developed if installed in the ediatable mode. But the best experience will be achieved by adopting [`uv`](https://github.com/astral-sh/uv) tool.

The project is showcasing a simple but very configurable binary classification task with an MLP network using a synthetic randomly generated dataset.

## Discaimer

This template is still a work in progress and will be improved in the future. Please use it with caution and report any issues or suggestions.

By default, the project is configured with very aggressive linting rules. Not all of them are strictly mandatory and can be turned off in `pyproject.toml` if sensible. Rules can be turned off either completely or only for the specific files and/or directories. In some corner cases, it is better to disable the rule for the specific line of code using `# noqa` comment.

For example, catching all exceptions using `try-except` block is considered a bad practice. But sometimes, for example, while building a generic function, we want to gracefully shut down our application and are required to catch everything. In such cases, we can disable the specific rule on case by case basis:

```python
def gracefully(func: Callable) -> None:
    try:
        func()
    except:  # noqa: E722
        # gracefully shutdown the application
```

The template also provies github actions for type checking your code which are also aggressively configured. Currently, strict type checking is not mandatory but is very welcomed if it does not interfere with the development too much. The type-checking requirements will be reconfigured and might be relaxed in the future.

Lastly, testing your code with `pytest` is also available in the project using github actions. Comprehensive testing helps to properly structure code and greatly simplifies maintenance and further development when the project grows. It is highly recommended to start covering all the vital parts of the codebase with tests immediately from the start of the project since it would greatly simplify the adoption of this practice. Testing will be strictly mandatory in the future for all the projects growing out of their infancy.

## Structure

This is a simplified project structure. `data` and `models` directories contain all the code necessary for model training while `configs` directory holds the dynamic building of all necessary configs for cli using `hydra_zen`. `main.py` contains the training and evaluation function and `train.py` is the entry point for the training script managed by `hydra`.

```plaintext
.github/workflows/                         # github actions for linting, type checking and testing
src/
└──── project                              # project name - RENAME
    ├── config.py
    ├── configs                            # folder containing builds of all configurations
    │   ├── __init__.py                            # sink for all submodule's configs
    │   ├── builder.py                             # final configuration composition
    │   ├── callbacks.py
    │   ├── datamodules.py
    │   ├── debug.py
    │   ├── experiments                            # experiment configurations
    │   │   ├── __init__.py
    │   ├── loggers.py
    │   ├── models
    │   │   ├── criterions.py
    │   │   ├── optimizers.py
    │   │   └── schedulers.py
    │   ├── nets
    │   │   └── mlps.py
    │   ├── trainer.py
    │   └── utils.py                               # all config utilities
    ├── data                            
    │   ├── datamodules                    # lightning datamodue responsible for data loading
    │   │   └── example.py
    │   └── dataset
    │       └── random.py                  # dataset with randomly generated data
    ├── main.py                            # module containing main train function
    ├── models
    │   ├── binary_classification.py       # lightning module responsible for training
    │   └── nets
    │       └── mlp.py                     # simple MLP as pytorch module
    ├── train.py                           # script for training the model managed by hydra
    └── utils                              # utility functions for train loop
        ├── boilerplate.py
        ├── callbacks.py                           # lightning callbacks
        ├── extra.py
        ├── logging.py
        └── mlflow.py                              # custom mlflow logger & checkpointer
```

## Install

Create a virtual environment and activate it:

```bash
uv venv --python 3.11 # pinned to recent DBX ML runtime python version
source .venv/bin/activate
```

Install the project with `dev` dependencies:

```bash
uv sync --extra dev
```

Additional optional dependencies can be found in the `pyproject.toml` file under the `[project.optional-dependencies]` section.

Setup pre-commit hooks - **MANDATORY** for further development:

```bash
pre-commit install
```

## Usage

The project is configured using `hydra` and `hydra_zen` which provide extensive cli configuration options. To run the training script, you need to specify all required parameters in the command line (required parameters are marked with `???` in config available in the [debug mode](#debug-hydra)). For example:

```bash
python src/project/train.py datamodule.dataset.num_features=64 model.net.input_dim=64
```

More details about `hydra` cli can be found in the [official documentation](https://hydra.cc/docs/advanced/override_grammar/basic/).

Additionally, you can specify the experiment configuration created and registered in the `configs/experiments/__init__.py` which would override `model/net` config group to initialize the network with different parameters, will enable logging using `mlflow` with the desired experiment name and will provide the required `num_features` and `input_dim` parameters, etc:

```bash
python src/project/train.py experiment=example-deep
# OR
python src/project/train.py experiment=example-wide
```

This project allows configuring independently each part of the training pipeline such as: `trainer`, `loggers`, `callbacks`, `datamodule`,
and `model` including its components (`model/net`, `model/criterion`, `model/optimizer`, `model/scheduler` and `model/lr_scheduler_config`). On top of that, the project provides a number of presets useful for debugging and testing your pipeline available in the `debug` config group registered in the `configs/debug.py` file.

### Debug `hydra`

There are a lot of cli args availabe to [control](https://hydra.cc/docs/advanced/hydra-command-line-flags/) `hydra`s behavior and to [debug](https://hydra.cc/docs/tutorials/basic/running_your_app/debugging/) its configuration composition. For example, to preview the composed configuration, you can run:

```bash
python src/project/train.py --cfg job
```

and to see the priorities of the default config lists, run:

```bash
python src/project/train.py --info defaults-tree
```

## Databricks

To sync local files with `Repos` [on the Databricks](https://docs.databricks.com/en/archive/dev-tools/dbx/dbx-sync.html), create `.syncinclude` file with patterns to include. Run command:

```bash
dbx sync repo -s . -d floor_optimization
```
