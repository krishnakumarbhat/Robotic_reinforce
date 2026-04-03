#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

if [ "$#" -eq 0 ]; then
  set -- PickCube-v1 PushCube-v1 StackCube-v1
fi

for ENV_ID in "$@"; do
  echo "Downloading official demos for $ENV_ID"
  env -u PYTHONPATH "$PYTHON_BIN" -m mani_skill.utils.download_demo "$ENV_ID"
done