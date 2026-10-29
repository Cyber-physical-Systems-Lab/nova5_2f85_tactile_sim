#!/usr/bin/env bash
# Run a script with Isaac Sim + Isaac Lab using conda's Python.
# Usage: bash scripts/isaac_python.sh scripts/grasp_demo.py --headless --out runs/x
set -e
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
TASK_CONDA_ROOT="${TASK_CONDA_ROOT:-${HOME}/miniconda3}"
TASK_CONDA_ENV="${TASK_CONDA_ENV:-locomanipu}"
source "${TASK_CONDA_ROOT}/etc/profile.d/conda.sh"
# Re-activating the same environment can trip an inherited deactivation hook.
if [[ "${CONDA_PREFIX:-}" != "${TASK_CONDA_ROOT}/envs/${TASK_CONDA_ENV}" ]]; then
    conda activate "${TASK_CONDA_ENV}"
fi
export ISAAC_PATH="${ISAAC_PATH:-${HOME}/isaacsim}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-${HOME}/IsaacLab}"
export CARB_APP_PATH="${ISAAC_PATH}/kit"
export EXP_PATH="${ISAAC_PATH}/apps"
export RESOURCE_NAME=IsaacSim
source "${ISAAC_PATH}/setup_python_env.sh"
# trimesh (needed by TacSL) ships in Isaac Sim's pip prebundle.
export PYTHONPATH="${TASK_ROOT}:${ISAACLAB_PATH}/source/isaaclab:${PYTHONPATH:-}:${ISAAC_PATH}/exts/omni.pip.compute/pip_prebundle"
# Conda's ICU/SQLite needs a newer C++ ABI than Ubuntu 22.04's system library.
export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6:${ISAAC_PATH}/kit/libcarb.so${LD_PRELOAD:+:${LD_PRELOAD}}"
export PYTHONNOUSERSITE=1
cd "${TASK_ROOT}"
exec "${CONDA_PREFIX}/bin/python" "$@"
