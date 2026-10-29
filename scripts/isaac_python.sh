#!/usr/bin/env bash
# Run a Python script with Isaac Sim's interpreter and this repo's src/ on PYTHONPATH.
#
# Interpreter resolution order:
#   1. $ISAACSIM_PYTHON            (explicit interpreter, e.g. a venv with `pip install isaacsim`)
#   2. $ISAACSIM_PATH/python.sh    (binary/zip install, e.g. ISAACSIM_PATH=~/isaacsim)
#   3. ~/isaacsim/python.sh
#   4. python3                     (assumes isaacsim is pip-installed in the active env)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -n "${ISAACSIM_PYTHON:-}" ]]; then
  PY=("${ISAACSIM_PYTHON}")
elif [[ -n "${ISAACSIM_PATH:-}" && -x "${ISAACSIM_PATH}/python.sh" ]]; then
  PY=("${ISAACSIM_PATH}/python.sh")
elif [[ -x "${HOME}/isaacsim/python.sh" ]]; then
  PY=("${HOME}/isaacsim/python.sh")
else
  PY=(python3)
  echo "[isaac_python] ISAACSIM_PATH not set; falling back to python3 (pip-installed isaacsim)" >&2
fi
exec "${PY[@]}" "$@"
