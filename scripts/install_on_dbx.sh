# install uv to the specified dir
export UV_INSTALL_DIR=$HOME/.local/bin
curl -LsSf https://astral.sh/uv/install.sh | sh
# activate it
source $UV_INSTALL_DIR/env
# set uv venv path to match the dbx virtual env dir
export UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV
echo installing project to $UV_PROJECT_ENVIRONMENT
# install project and dependencies
uv sync --link-mode=copy $@
