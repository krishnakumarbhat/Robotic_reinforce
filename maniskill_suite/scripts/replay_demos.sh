#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

ENV_ID="${1:-PickCube-v1}"
SOURCE="${2:-teleop}"
OBS_MODE="${3:-state}"
CONTROL_MODE="${4:-pd_ee_delta_pos}"
COUNT="${5:-3}"
SIM_BACKEND="${6:-physx_cpu}"
NUM_ENVS="${7:-4}"
DEMO_ROOT="${MANISKILL_DEMO_ROOT:-$HOME/.maniskill/demos}"
TRAJ_PATH="$DEMO_ROOT/$ENV_ID/$SOURCE/trajectory.h5"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

if [ "$SOURCE" = "rl" ] && [ ! -f "$TRAJ_PATH" ]; then
  CUDA_PATH="$DEMO_ROOT/$ENV_ID/$SOURCE/trajectory.none.$CONTROL_MODE.physx_cuda.h5"
  CPU_PATH="$DEMO_ROOT/$ENV_ID/$SOURCE/trajectory.none.$CONTROL_MODE.physx_cpu.h5"
  if [ -f "$CUDA_PATH" ]; then
    TRAJ_PATH="$CUDA_PATH"
  elif [ -f "$CPU_PATH" ]; then
    TRAJ_PATH="$CPU_PATH"
  fi
fi

if [ ! -f "$TRAJ_PATH" ]; then
  echo "Could not find trajectory file: $TRAJ_PATH" >&2
  exit 1
fi

STATE_FLAG="--use-first-env-state"
if [ "$SOURCE" = "rl" ]; then
  STATE_FLAG="--use-env-states"
fi

echo "Replaying $TRAJ_PATH to obs_mode=$OBS_MODE with control_mode=$CONTROL_MODE"

env -u PYTHONPATH "$PYTHON_BIN" -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$TRAJ_PATH" \
  $STATE_FLAG \
  -c "$CONTROL_MODE" \
  -o "$OBS_MODE" \
  --save-traj \
  --count "$COUNT" \
  --num-envs "$NUM_ENVS" \
  -b "$SIM_BACKEND"