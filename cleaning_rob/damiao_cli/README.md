# damiao_cli

Tools for driving the Damiao DM-series motors (DM-J4310 / DM-J4340) of the **arm_01** follower arm over a Waveshare CANalyst-II USB-CAN adapter. This repo contains:

- the **CAN bridge** (`socketcan_virtual_converter.py`), which forwards traffic between the CANalyst-II and a virtual CAN interface (`vcan0`) so LeRobot can talk to the follower
- the **one-shot teleop launcher** (`scripts/teleop.sh`)
- a standalone motor CLI and SDK (`run.py`, `motor.py`, `params.yaml`) for bench-testing individual motors

## Teleoperation quick start

These steps take a fresh Linux machine from nothing to leader → follower teleoperation.

> [!IMPORTANT]
> **Use the directory layout below exactly.** The scripts in both repos use hardcoded paths and do not search for each other:
>
> ```
> ~/dev/
> ├── lerobot-configured/        # LeRobot fork (branch custom_design_01)
> └── etc/
>     └── damiao_cli/            # this repo
>         ├── socketcan_virtual_converter.py
>         └── scripts/
>             └── teleop.sh
> ```
>
> - `teleop.sh` runs `lerobot-teleoperate` from **`~/dev/lerobot-configured`**. If that directory is missing, the bridge starts and then teleop fails with `LeRobot dir not found`.
> - `teleop.sh` finds the bridge through its own location (`scripts/..`), so **don't move or copy it out of `scripts/`**. Symlinks are fine.
> - The lerobot-configured README and its docs assume the bridge is at **`~/dev/etc/damiao_cli`**.
>
> If you can't use this layout, set `LEROBOT_DIR=/path/to/lerobot-configured` every time you run `teleop.sh`, and remember that the lerobot-configured docs won't match your paths.

### 1. Clone this repo

```bash
mkdir -p ~/dev/etc
git clone git@github.com:LeevAI-Devs/damiao_cli.git ~/dev/etc/damiao_cli
```

### 2. Install uv (if you don't have it)

```bash
command -v uv || { curl -LsSf https://astral.sh/uv/install.sh | sh; source ~/.local/bin/env; }
```

uv needs to be at `~/.local/bin/uv`, where the installer puts it. When `teleop.sh` is started with `sudo`, it finds uv through that path.

Then install this repo's dependencies (Python 3.10 is downloaded automatically if needed):

```bash
cd ~/dev/etc/damiao_cli
uv sync --locked
```

### 3. Get and set up lerobot-configured

```bash
git clone -b custom_design_01 git@github.com:LeevAI-Devs/lerobot-configured.git ~/dev/lerobot-configured
cd ~/dev/lerobot-configured
uv sync --locked --extra feetech --extra damiao     # leader servo SDK + python-can for the follower
uv run python scripts/install_calibration.py         # copy arm_01 calibration into ~/.cache/huggingface/lerobot
```

The arm_01 code lives on the **`custom_design_01`** branch. Don't skip the calibration step: without it, LeRobot can't find `damiao_follower_01` / `sts_b601_leader_01` and starts an interactive calibration instead of teleoperating. See the [lerobot-configured README](https://github.com/LeevAI-Devs/lerobot-configured) for details.

### 4. One-time system setup

```bash
# Serial access for the leader arm, and USB access to the CANalyst-II
sudo usermod -aG dialout,plugdev $USER
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="04d8", ATTR{idProduct}=="0053", MODE="0660", GROUP="plugdev"' \
  | sudo tee /etc/udev/rules.d/99-canalystii.rules
sudo udevadm control --reload-rules && sudo udevadm trigger

# Optional: candump, for watching CAN traffic
sudo apt install can-utils
```

Log out and back in so the new groups take effect. Check that the `vcan` kernel module exists with `sudo modprobe vcan`. `teleop.sh` loads it for you on every run.

### 5. Run teleop

1. Plug in the CANalyst-II and the leader's USB cable.
2. Power both arms.
3. Put the arms in roughly the same pose.
4. Run:

```bash
~/dev/etc/damiao_cli/scripts/teleop.sh
```

The script:

1. Asks for your sudo password once, to set up `vcan0`.
2. Starts the bridge and waits until it reports `Bridge running.`
3. Runs `lerobot-teleoperate` at 20 fps.

Press `Ctrl+C` to stop teleop and the bridge together.

The script accepts these environment overrides:

| Variable         | Default                     | Purpose                                                       |
| ---------------- | --------------------------- | ------------------------------------------------------------- |
| `TELEOP_PORT`    | `/dev/ttyACM0`              | Leader arm serial port (find it with `uv run lerobot-find-port` in lerobot-configured) |
| `LEROBOT_DIR`    | `~/dev/lerobot-configured`  | Location of the LeRobot fork                                  |
| `BRIDGE_TIMEOUT` | `30`                        | Seconds to wait for the bridge to come up                     |

```bash
TELEOP_PORT=/dev/ttyACM1 ~/dev/etc/damiao_cli/scripts/teleop.sh
```

To run it with no password prompt at all, add the sudoers drop-in described in the header of `scripts/teleop.sh`.

### Troubleshooting

- **"The bridge could not be started or died suddenly":** check the CANalyst-II USB cable and the udev rule from step 4. Power-cycle the motor bus and retry.
- **"Teleoperation failed":** make sure both arms are powered and connected and start from similar poses. The bridge keeps running so you can inspect traffic with `candump vcan0`.
- **"LeRobot dir not found":** lerobot-configured isn't at `~/dev/lerobot-configured`. See the directory layout above.
- **Asked to calibrate:** re-run `scripts/install_calibration.py` in lerobot-configured (step 3).

## Motor CLI and SDK

The rest of the repo is for driving motors directly, without LeRobot. The CLI talks to `vcan0` by default, so the bridge has to be running first. Either run `uv run socketcan_virtual_converter.py` after creating `vcan0` (see the vcan setup in `scripts/teleop.sh`), or use `--channel` / `--interface`.

```bash
uv run run.py help                    # list all commands
uv run run.py scan                    # read positions of motors 0x01–0x07
uv run run.py query 0x03              # read state
uv run run.py move_to_pos 0x03 -45.0  # absolute move (degrees)
uv run run.py move_by_offset 0x03 20  # relative move (degrees)
```

| Path                       | What it is                                                                      |
| -------------------------- | ------------------------------------------------------------------------------- |
| `motor.py`                 | `MotorBus` SDK: MIT control, register protocol, motion profiles, settle          |
| `run.py`                   | CLI over `MotorBus`                                                             |
| `params.yaml`              | Per-model (DM4310 / DM4340) gains and tuning, loaded by default from the repo root |
| `scripts/sequence.py`      | Fixed multi-motor move sequence demo                                            |
| `scripts/bilateral.py`     | Bilateral spring/damper coupling between two motors                             |
| `scripts/settle_sim.py`    | Hardware-free simulation for re-tuning the end-of-move settle                   |
| `SDK_REFERENCE.md`         | Detailed SDK and protocol reference                                             |
| `bilateral_breakdown.pdf`  | Write-up of the bilateral coupling control law                                  |
