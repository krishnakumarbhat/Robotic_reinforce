#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

ENV_ID="${1:-PickCube-v1}"
RECORD_DIR="${2:-$HOME/.maniskill/demos}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

cat <<EOF
Launching ManiSkill click-and-drag teleoperation for $ENV_ID

Useful keys:
  n = motion plan to dragged pose
  g = toggle gripper open or close
  c = save the current trajectory and start another
  q = quit
  h = help

For a pilot run, collect 2 to 3 successful demos.
EOF

env -u PYTHONPATH "$PYTHON_BIN" -m mani_skill.examples.teleoperation.interactive_panda \
  -e "$ENV_ID" \
  --record-dir "$RECORD_DIR" \
  --save-video