# Two Mouse Dual-Arm Imitation Learning

This folder is a self-contained ManiSkill project for collecting dual-arm demonstrations with two mice and training a simple behavior cloning policy from those demonstrations.

What it does:

- detects candidate mouse devices on Linux
- checks whether raw per-device event access is available
- runs a two-arm ManiSkill task with one mouse mapped to each Panda end-effector
- records demonstrations into a simple HDF5 plus JSON dataset format
- trains a small behavior cloning model on state observations
- evaluates the learned model back in ManiSkill

The default task is `TwoRobotPickCube-v1`, which is a built-in ManiSkill collaborative two-Panda task.

## Folder Layout

```text
2_mouse_imitation_learning/
├── README.md
├── requirements.txt
├── bootstrap.sh
├── detect_two_mice.py
├── generate_sanity_dataset.py
├── record_dual_mouse_demos.py
├── train_behavior_cloning.py
├── evaluate_behavior_cloning.py
├── smoke_test.py
└── two_mouse_il/
    ├── __init__.py
    ├── dataset.py
    ├── env_setup.py
    ├── input_devices.py
    ├── policy.py
    └── teleop.py
```

## Setup

```bash
cd 2_mouse_imitation_learning
bash bootstrap.sh
```

This creates a local `.venv` and installs a CPU-only Torch build plus the ManiSkill and Linux input dependencies used by this project.

## Detect Mice

```bash
env -u PYTHONPATH ./.venv/bin/python detect_two_mice.py
```

To also include the laptop touchpad as a second pointer candidate:

```bash
env -u PYTHONPATH ./.venv/bin/python detect_two_mice.py --include-touchpad
```

To test live device opening:

```bash
env -u PYTHONPATH ./.venv/bin/python detect_two_mice.py --open
```

Important Linux note:

- raw per-device mouse tracking uses `/dev/input/event*`
- your user must be able to read those devices
- if you get `PermissionError`, add your user to the `input` group or install a suitable udev rule

## Smoke Test

```bash
env -u PYTHONPATH ./.venv/bin/python smoke_test.py
```

This checks both:

- pointer-device discovery
- ManiSkill environment startup for the two-arm task

## Generate A Sanity Dataset

If Linux input permissions prevent live dual-mouse recording, you can still validate the full training pipeline with a small scripted dataset:

```bash
env -u PYTHONPATH ./.venv/bin/python generate_sanity_dataset.py \
  --dataset data/recordings/two_robot_pick_cube_sanity.h5 \
  --episodes 4 \
  --steps 30
```

This is not a useful imitation dataset for learning task success. It is only for verifying that dataset writing, training, checkpointing, and evaluation all work end to end.

## Record Demonstrations

```bash
env -u PYTHONPATH ./.venv/bin/python record_dual_mouse_demos.py
```

Default controls:

- mouse movement: XY motion for that arm
- mouse wheel: Z motion for that arm
- left click: toggle that arm gripper open or closed
- left mouse middle click: save current episode and reset
- right mouse middle click: save current episode and quit
- `Ctrl+C`: quit safely

The left mouse controls the left Panda arm. The right mouse controls the right Panda arm.

## Train Behavior Cloning

```bash
env -u PYTHONPATH ./.venv/bin/python train_behavior_cloning.py \
  --dataset data/recordings/two_robot_pick_cube.h5 \
  --output models/two_robot_pick_cube_bc.pt
```

## Evaluate Behavior Cloning

```bash
env -u PYTHONPATH ./.venv/bin/python evaluate_behavior_cloning.py \
  --checkpoint models/two_robot_pick_cube_bc.pt \
  --episodes 5
```

## Current Machine Caveat

This project can detect candidate pointer devices without special privileges, but live per-device event capture still depends on Linux input permissions. If you can see devices listed but cannot open them, the code is correct and the blocker is OS permissions, not the teleop logic.

## Verified In This Workspace

This project was validated locally with the following outcomes:

- `detect_two_mice.py --include-touchpad` listed three pointer candidates, including two mouse-class devices
- `detect_two_mice.py --include-touchpad --open` failed cleanly with `Permission denied` on `/dev/input/event*`
- `smoke_test.py --include-touchpad --steps 3` booted `TwoRobotPickCube-v1`, found both Panda agents, and stepped the environment successfully
- `record_dual_mouse_demos.py --include-touchpad` failed cleanly with the same permission warning instead of crashing
- `generate_sanity_dataset.py`, `train_behavior_cloning.py`, and `evaluate_behavior_cloning.py` all completed successfully inside this folder's `.venv`

See `VALIDATION.md` for the concrete outputs that were observed.
