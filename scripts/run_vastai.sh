#!/usr/bin/env bash
# vastai is running in a docker container with already torch and cuda installed
# this script reuse the system installed python env + cuda + torch
# and install the rest.

# install uv prior to running this script
# curl -LsSf https://astral.sh/uv/install.sh | sh

# inside the container, pick the interpreter that has torch preinstalled
PY=$(command -v python)  # often /usr/bin/python or conda's python
export UV_PYTHON="$PY"

uv venv --system-site-packages
uv sync --extra transformers # no --extra cpu/cu129 -> we don't install torch at all
uv run python -m iudicium transformers_translator $@