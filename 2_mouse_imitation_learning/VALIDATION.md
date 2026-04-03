# Validation

This file records the concrete end-to-end checks run for the `2_mouse_imitation_learning` project in this workspace.

## Environment

- Python: `3.10.12`
- Interpreter: local `.venv` inside this folder
- Torch path: local `.venv` torch install
- Torch CUDA availability: `False`
- ManiSkill version: `3.0.0b22`

The project is intentionally validated on CPU, which is sufficient for device detection, dataset writing, training, and small evaluation runs.

## Mouse Detection

Command:

```bash
env -u PYTHONPATH ./.venv/bin/python detect_two_mice.py --include-touchpad
```

Observed result:

- `/dev/input/event8` `PixArt Dell MS116 USB Optical Mouse`
- `/dev/input/event10` `SYNA7DB5:00 06CB:CEA8 Mouse`
- `/dev/input/event15` `SYNA7DB5:00 06CB:CEA8 Touchpad`

This means the machine does expose two mouse-class devices plus a touchpad candidate.

## Raw Device Access

Command:

```bash
env -u PYTHONPATH ./.venv/bin/python detect_two_mice.py --include-touchpad --open
```

Observed result:

- device enumeration succeeded
- live open failed with `Permission denied opening /dev/input/event8`

Interpretation:

The project can detect the devices, but live per-device teleoperation is currently blocked by Linux input permissions for this user account.

## ManiSkill Smoke Test

Command:

```bash
env -u PYTHONPATH ./.venv/bin/python smoke_test.py --include-touchpad --steps 3
```

Observed result:

- environment booted successfully
- `env_id=TwoRobotPickCube-v1`
- `agent_keys=['panda_wristcam-0', 'panda_wristcam-1']`
- `action_dim=8`
- three zero-action steps executed successfully

## Recorder Entrypoint

Command:

```bash
env -u PYTHONPATH ./.venv/bin/python record_dual_mouse_demos.py --include-touchpad
```

Observed result:

- recorder started correctly
- recorder failed cleanly with the same `/dev/input/event*` permission warning
- no crash or traceback occurred

## Training Pipeline Sanity Check

Because live recording is currently blocked by OS permissions, a scripted dataset was generated to validate the training and evaluation pipeline.

### Dataset Generation

Command:

```bash
env -u PYTHONPATH ./.venv/bin/python generate_sanity_dataset.py \
  --dataset data/recordings/two_robot_pick_cube_sanity.h5 \
  --episodes 4 \
  --steps 20
```

Observed result:

- 4 episodes saved successfully
- dataset and metadata JSON were written under `data/recordings/`

### Behavior Cloning Training

Command:

```bash
env -u PYTHONPATH ./.venv/bin/python train_behavior_cloning.py \
  --dataset data/recordings/two_robot_pick_cube_sanity.h5 \
  --output models/two_robot_pick_cube_sanity_bc.pt \
  --epochs 5 \
  --batch-size 64
```

Observed result:

- training completed successfully
- validation loss decreased from about `0.2435` to about `0.1530`
- checkpoint saved under `models/`

### Behavior Cloning Evaluation

Command:

```bash
env -u PYTHONPATH ./.venv/bin/python evaluate_behavior_cloning.py \
  --checkpoint models/two_robot_pick_cube_sanity_bc.pt \
  --episodes 2
```

Observed result:

- evaluation completed successfully
- average return was about `0.9878`
- `success_once=0.0`
- `success_at_end=0.0`

This is expected because the scripted sanity dataset is only for pipeline validation, not for learning useful task behavior.

## Final Status

- Project code: complete and runnable inside `2_mouse_imitation_learning/`
- Two-mouse detection: confirmed
- Two-mouse live opening: blocked by OS permissions on this machine
- ManiSkill two-arm environment: confirmed
- Dataset writer: confirmed
- BC trainer: confirmed
- BC evaluator: confirmed

To use live dual-mouse teleoperation, grant this user read access to `/dev/input/event*` and rerun `record_dual_mouse_demos.py`.