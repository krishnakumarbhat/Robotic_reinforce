# lerobot-configured — arm_01 setup guide

A fork of [huggingface/lerobot](https://github.com/huggingface/lerobot) configured for the **arm_01** prototype:

| Role     | LeRobot type      | Calibration id       | Hardware                                                  | Port                                 |
| -------- | ----------------- | -------------------- | --------------------------------------------------------- | ------------------------------------ |
| Follower | `damiao_follower` | `damiao_follower_01` | 6x Damiao (DM4340 x3, DM4310 x3) on CAN, via CANalyst-II  | `vcan0` (bridged, see step 5)        |
| Leader   | `sts_b601_leader` | `sts_b601_leader_01` | 6x Feetech STS3215 on USB serial                          | `/dev/ttyACM0`                       |

Both arms are 6-DOF with no gripper and share the joint names `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll`, so leader actions map straight onto the follower.

The steps below take a fresh Linux machine from clone to teleoperation.

## 1. Clone the repo

```bash
mkdir -p ~/dev && cd ~/dev
git clone -b custom_design_01 git@github.com:LeevAI-Devs/lerobot-configured.git
cd lerobot-configured
```

Keep it at `~/dev/lerobot-configured`: the one-shot launcher in step 6 looks there by default.

## 2. Install uv and the dependencies

[uv](https://docs.astral.sh/uv/) manages Python (3.12+) and the virtualenv for you.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env        # or open a new terminal

uv sync --locked --extra feetech --extra damiao
```

`feetech` is the leader's servo SDK and `damiao` pulls in `python-can` for the follower. Use `--extra all` if you also want training and simulation dependencies.

## 3. Install the calibration data

LeRobot reads calibration only from `~/.cache/huggingface/lerobot/calibration` (or `$HF_LEROBOT_CALIBRATION`). The arm_01 calibration is versioned in [`calibration/`](./calibration). Copy it into place:

```bash
uv run python scripts/install_calibration.py
```

```
calibration/
├── robots/damiao_follower/damiao_follower_01.json
└── teleoperators/sts_b601_leader/sts_b601_leader_01.json
```

The script skips identical files and never silently overwrites a different one. Its other flags:

```bash
uv run python scripts/install_calibration.py --dry-run  # preview only
uv run python scripts/install_calibration.py --force    # overwrite; old file kept as *.json.bak-<timestamp>
uv run python scripts/install_calibration.py --export   # after re-calibrating: cache -> repo, then commit
```

The calibration id (`--robot.id` / `--teleop.id`) selects the file, so always pass `damiao_follower_01` and `sts_b601_leader_01`. With any other id, LeRobot starts a fresh calibration.

## 4. Hardware permissions (one-time)

```bash
# Leader arm serial port (/dev/ttyACM*)
sudo usermod -aG dialout,plugdev $USER

# CANalyst-II USB-CAN adapter: let plugdev open it through libusb
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="04d8", ATTR{idProduct}=="0053", MODE="0660", GROUP="plugdev"' \
  | sudo tee /etc/udev/rules.d/99-canalystii.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Log out and back in so the new groups take effect.

To confirm the leader's port, run `uv run lerobot-find-port` and unplug the leader's USB cable when prompted. It is usually `/dev/ttyACM0`.

## 5. Bring up the CAN bridge

The follower is driven through a Waveshare CANalyst-II, which python-can can't reach through socketcan directly. A small bridge in the [`damiao_cli`](https://github.com/LeevAI-Devs/damiao_cli) repo forwards traffic between the adapter and a virtual CAN interface, `vcan0`, and LeRobot talks to `vcan0`.

```bash
# One-time: clone the bridge next to this repo
mkdir -p ~/dev/etc && git clone git@github.com:LeevAI-Devs/damiao_cli.git ~/dev/etc/damiao_cli

# Every boot: create vcan0
sudo modprobe vcan
ip link show vcan0 &>/dev/null || sudo ip link add dev vcan0 type vcan
sudo ip link set up vcan0

# Run the bridge (leave it running in its own terminal)
cd ~/dev/etc/damiao_cli
uv run socketcan_virtual_converter.py     # prints "Bridge running. vcan0 <-> Waveshare channel 1"
```

Use `candump vcan0` (from `can-utils`) to watch traffic.

## 6. Teleoperate

Power both arms and put them in roughly the same pose. Then, with the bridge running, in a second terminal:

```bash
cd ~/dev/lerobot-configured
uv run lerobot-teleoperate \
  --robot.type=damiao_follower --robot.port=vcan0 --robot.id=damiao_follower_01 \
  --teleop.type=sts_b601_leader --teleop.port=/dev/ttyACM0 --teleop.id=sts_b601_leader_01 \
  --fps=20
```

Press `Ctrl+C` to stop; torque is disabled on disconnect. Add `--display_data=true` to stream joint values to the Rerun viewer (it is skipped automatically when there is no display).

**One-shot alternative:** `damiao_cli/scripts/teleop.sh` does step 5 and this step in one go. It sets up `vcan0`, starts the bridge, waits for it, runs the exact command above at 20 fps, and tears everything down on `Ctrl+C`:

```bash
~/dev/etc/damiao_cli/scripts/teleop.sh                               # asks for sudo once, for vcan0
TELEOP_PORT=/dev/ttyACM1 ~/dev/etc/damiao_cli/scripts/teleop.sh      # leader on a different port
```

## Troubleshooting

- **Bridge fails with "CANalyst-II not found":** check the adapter's USB cable and step 4's udev rule. Restart the motor bus power and retry.
- **Teleoperation exits right away:** make sure both arms are powered and connected and start from similar poses. Check `candump vcan0` for motor replies.
- **Asked to calibrate:** the id didn't match an installed file. Re-run step 3 and check the ids.
- **Follower tracking is weak or buzzes:** gains, joint limits, signs and offsets live in `src/lerobot/robots/damiao_follower/config_damiao_follower.py`, and every value can be overridden on the command line (e.g. `--robot.range_margin_deg=5`).

## Re-calibrating

```bash
uv run lerobot-calibrate --robot.type=damiao_follower --robot.port=vcan0 --robot.id=damiao_follower_01
uv run lerobot-calibrate --teleop.type=sts_b601_leader --teleop.port=/dev/ttyACM0 --teleop.id=sts_b601_leader_01
uv run python scripts/install_calibration.py --export    # copy the new files into calibration/, then commit
```

After re-calibrating either arm, re-measure the follower's `joint_offsets` (bridge running) and paste them into `config_damiao_follower.py`:

```bash
uv run python -m lerobot.robots.damiao_follower.measure_joint_alignment \
  --robot.type=damiao_follower --robot.port=vcan0 --robot.id=damiao_follower_01 \
  --teleop.type=sts_b601_leader --teleop.port=/dev/ttyACM0 --teleop.id=sts_b601_leader_01 \
  --measure_signs=false
```

## More

- [`AGENT_GUIDE.md`](./AGENT_GUIDE.md): recording datasets, choosing and training policies, evaluation.
- [LeRobot documentation](https://huggingface.co/docs/lerobot) for everything upstream.
- License: Apache 2.0, see [`LICENSE`](./LICENSE).
