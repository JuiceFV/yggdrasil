# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# activate it
source $HOME/.cargo/env
# set uv venv path to match the dbx virtual env dir
export UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV
echo installing project to $UV_PROJECT_ENVIRONMENT
# install project and dependencies
uv sync --link-mode=copy
