#!/usr/bin/env bash
# Prepare everything under assets/generated/ (run once after cloning, and after changes to nova5_tactile/).
#  1. clone the CPS-lab tactile gripper CAD into third_party/tactile_gripper (pinned commit), and copy
#     Daimon's DM-Tac WL sensor STEP into third_party/daimon/ if DAIMON_STEP points to it
#     (vendor file, not redistributed; without it the sensor fingers are plain boxes)
#  2. convert the gripper's Solid Edge assembly + STEP parts and the sensor STEP to USD (Isaac Sim HOOPS)
# Usage: DAIMON_STEP="$HOME/Downloads/DM-Tac-W-STEP-20250417/DM-Tac WL.STEP" bash scripts/setup_assets.sh
#  3. build the robots (NOVA5 + CPS tactile gripper, NOVA5 + Robotiq 2F-85), pegs and boards
set -e
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
GRIPPER_REPO=https://github.com/Cyber-physical-Systems-Lab/master-thesis-tactile-gripper.git
GRIPPER_COMMIT=783f026
cd "${ROOT}"
if [[ ! -d third_party/tactile_gripper ]]; then
    git clone "${GRIPPER_REPO}" third_party/tactile_gripper
fi
git -C third_party/tactile_gripper checkout -q "${GRIPPER_COMMIT}"
mkdir -p third_party/daimon
if [[ -n "${DAIMON_STEP:-}" ]]; then
    cp "${DAIMON_STEP}" "third_party/daimon/DM-Tac WL.STEP"
fi
if [[ ! -f "third_party/daimon/DM-Tac WL.STEP" ]]; then
    echo "NOTE: third_party/daimon/DM-Tac WL.STEP not found; set DAIMON_STEP to Daimon's DM-Tac WL STEP file." >&2
    echo "      Continuing with box-shaped sensor fingers." >&2
fi
bash scripts/isaac_python.sh scripts/convert_gripper_cad.py
bash scripts/isaac_python.sh scripts/build_assets.py
echo "assets ready in ${ROOT}/assets/generated"
