#!/usr/bin/env bash
# Clone Dobot's ROS 2 repo (shallow) and produce assets/urdf/nova5.urdf with absolute mesh paths.
#
# Env overrides:
#   DOBOT_REPO_URL  repo to clone (default: Dobot-Arm/DOBOT_6Axis_ROS2_V4)
#   NOVA_MODEL      file-name pattern to search for (default: nova5)
#   NOVA_URDF       skip the search and use this .urdf/.xacro file
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${DOBOT_REPO_URL:-https://github.com/Dobot-Arm/DOBOT_6Axis_ROS2_V4.git}"
MODEL="${NOVA_MODEL:-nova5}"
DEST="${REPO_ROOT}/third_party/$(basename "${URL}" .git)"

if [[ -z "${NOVA_URDF:-}" ]]; then
  if [[ ! -d "${DEST}/.git" ]]; then
    mkdir -p "$(dirname "${DEST}")"
    git clone --depth 1 "${URL}" "${DEST}"
  else
    echo "using existing clone: ${DEST}"
  fi
  mapfile -t CANDS < <(find "${DEST}" -type f \( -iname "*${MODEL}*.urdf" -o -iname "*${MODEL}*.xacro" \) | sort)
  if (( ${#CANDS[@]} == 0 )); then
    echo "No '${MODEL}' URDF/xacro found in ${DEST}." >&2
    echo "Either set DOBOT_REPO_URL to a repo that contains it, or point at a file directly:" >&2
    echo "  NOVA_URDF=/path/to/nova5.urdf bash scripts/fetch_nova5_description.sh" >&2
    exit 1
  fi
  echo "candidates:"; printf '  %s\n' "${CANDS[@]}"
  PICK=""
  for c in "${CANDS[@]}"; do [[ "${c}" == *.urdf ]] && { PICK="${c}"; break; }; done
  [[ -z "${PICK}" ]] && PICK="${CANDS[0]}"
  SEARCH="${DEST}"
else
  PICK="${NOVA_URDF}"
  SEARCH="$(dirname "${PICK}")/.."
fi

echo "using: ${PICK}"
PYTHONPATH="${REPO_ROOT}/src" python3 "${REPO_ROOT}/scripts/prepare_urdf.py" \
  --input "${PICK}" --search-root "${SEARCH}" --output "${REPO_ROOT}/assets/urdf/nova5.urdf"
