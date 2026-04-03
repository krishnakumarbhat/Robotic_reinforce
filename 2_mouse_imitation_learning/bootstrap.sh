#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  env -u PYTHONPATH python3 -m venv "$VENV_DIR"
fi

env -u PYTHONPATH "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
env -u PYTHONPATH "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu torch
env -u PYTHONPATH "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"

mkdir -p "$ROOT_DIR/data/recordings" "$ROOT_DIR/data/evaluations" "$ROOT_DIR/models" "$ROOT_DIR/logs"

cat <<EOF
Project environment created in:
  $VENV_DIR

Next useful commands:
  env -u PYTHONPATH "$PYTHON_BIN" "$ROOT_DIR/detect_two_mice.py"
  env -u PYTHONPATH "$PYTHON_BIN" "$ROOT_DIR/smoke_test.py"
  env -u PYTHONPATH "$PYTHON_BIN" "$ROOT_DIR/record_dual_mouse_demos.py"
EOF
