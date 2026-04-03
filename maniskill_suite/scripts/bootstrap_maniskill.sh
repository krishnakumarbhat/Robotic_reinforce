#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  python3 -m venv "$VENV_DIR"
fi

env -u PYTHONPATH "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
env -u PYTHONPATH "$PYTHON_BIN" -m pip install --upgrade -r requirements.txt

mkdir -p external
if [ ! -d external/ManiSkill ]; then
  git clone https://github.com/haosulab/ManiSkill external/ManiSkill
fi

cat <<EOF
Core ManiSkill install completed inside:
  $ROOT_DIR/.venv

Next manual steps for Linux plus NVIDIA:
  sudo apt-get install libvulkan1 vulkan-tools
  vulkaninfo

Optional environment variables:
  export MANISKILL_REPO="$ROOT_DIR/external/ManiSkill"
  export MANISKILL_DEMO_ROOT="\$HOME/.maniskill/demos"

Quick smoke test:
  env -u PYTHONPATH "$ROOT_DIR/.venv/bin/python" -m mani_skill.examples.demo_random_action -e PickCube-v1 --render-mode none -n 1

Notes:
  - RFCL and RLPD require extra JAX-based dependencies from the official ManiSkill baseline folders.
  - This script intentionally does not run sudo commands for you.
EOF