#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Measure `joint_signs` and `joint_offsets` for `damiao_follower` against its leader.

Both arms stay limp. Each arm is posed and sampled on its own, so you only ever hold one arm:

1. Reference pose: the same physical pose on both arms. Gives offset = F_ref - sign * L_ref.
2. Displaced pose: every joint moved away from the reference, in the same physical direction on
   both arms. Gives the sign of each joint from the direction each arm's reading moved. Skipped
   with `--measure_signs=false`, which trusts the signs already in the follower config.

Both arms must already be calibrated, and the offsets must be re-measured whenever either arm is
re-calibrated. The signs only change if an arm is rebuilt.

Example, taking the config's signs as given and measuring offsets alone:

```shell
uv run python -m lerobot.robots.damiao_follower.measure_joint_alignment \
    --robot.type=damiao_follower \
    --robot.id=damiao_follower_01 \
    --teleop.type=sts_b601_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=sts_b601_leader_01 \
    --measure_signs=false
```
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

import draccus

from lerobot.robots import RobotConfig
from lerobot.teleoperators import (
    TeleoperatorConfig,
    make_teleoperator_from_config,
    sts_b601_leader,  # noqa: F401
)
from lerobot.utils.utils import init_logging

from .config_damiao_follower import DamiaoFollowerConfig
from .damiao_follower import DamiaoFollower


@dataclass
class MeasureJointAlignmentConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Readings averaged per pose, to smooth out encoder jitter.
    num_samples: int = 20
    # Readings discarded before averaging. A reply that misses the bus's receive window is returned
    # by the next read instead, so the first readings after ENTER can predate the pose.
    num_warmup: int = 5
    # Pause between readings; longer than the CAN round trip, so each read picks up a fresh reply.
    read_period_s: float = 0.03
    # Whether to measure each joint's direction. False takes `--robot.joint_signs` as given and
    # asks only for the reference pose.
    measure_signs: bool = True
    # Smallest movement between the two poses that still gives a trustworthy sign.
    min_travel_deg: float = 15.0


def _mean_positions(
    read: Callable[[], dict[str, float]], num_samples: int, num_warmup: int, period_s: float
) -> dict[str, float]:
    for _ in range(num_warmup):
        read()
        time.sleep(period_s)
    samples = []
    for _ in range(num_samples):
        samples.append(read())
        time.sleep(period_s)
    return {motor: sum(s[motor] for s in samples) / num_samples for motor in samples[0]}


def _fmt_dict(values: dict[str, float]) -> str:
    return "{" + ", ".join(f"{motor}: {val:.2f}" for motor, val in values.items()) + "}"


@draccus.wrap()
def measure_joint_alignment(cfg: MeasureJointAlignmentConfig) -> None:
    init_logging()
    if not isinstance(cfg.robot, DamiaoFollowerConfig):
        raise ValueError(f"--robot.type must be damiao_follower, got {cfg.robot.type}")

    robot = DamiaoFollower(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    if not robot.is_calibrated:
        raise RuntimeError(
            f"No calibration covering {list(robot.bus.motors)} for follower id '{robot.id}'. "
            "Run lerobot-calibrate on the follower first."
        )

    motors = list(robot.bus.motors)
    try:
        teleop.connect(calibrate=False)
        if not teleop.is_calibrated:
            raise RuntimeError(
                f"Leader id '{teleop.id}' is not calibrated. Run lerobot-calibrate on it first."
            )

        # Connecting runs a handshake that enables every motor; go limp straight away.
        robot.bus.connect()
        robot.bus.disable_torque()

        def read_leader() -> dict[str, float]:
            action = teleop.get_action()
            return {motor: action[f"{motor}.pos"] for motor in motors}

        def read_follower() -> dict[str, float]:
            return robot.bus.sync_read("Present_Position", motors)

        def sample(prompt: str, read: Callable[[], dict[str, float]]) -> dict[str, float]:
            input(f"\n{prompt}\nHold it steady and press ENTER...")
            return _mean_positions(read, cfg.num_samples, cfg.num_warmup, cfg.read_period_s)

        steps = 2 if cfg.measure_signs else 1
        print(f"\nStep 1/{steps}: reference pose. Use a pose you can repeat exactly on both arms, e.g.")
        print("the pose the follower was zeroed in during lerobot-calibrate.")
        f_ref = sample("Pose the FOLLOWER in the reference pose.", read_follower)
        l_ref = sample("Pose the LEADER in the same reference pose.", read_leader)

        f_moved = l_moved = None
        if cfg.measure_signs:
            print(f"\nStep 2/2: displaced pose. Move EVERY joint at least {cfg.min_travel_deg:g} deg away")
            print("from the reference, the same physical direction on both arms. Amounts need not match.")
            f_moved = sample("Move the FOLLOWER joints.", read_follower)
            l_moved = sample("Move the LEADER joints the same way.", read_leader)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=True)
        if teleop.is_connected:
            teleop.disconnect()

    signs: dict[str, float] = {}
    offsets: dict[str, float] = {}
    undetermined: list[str] = []
    disagreements: list[str] = []
    moved_cols = f"{'L_MOVE':>9} {'F_MOVE':>9} " if cfg.measure_signs else ""
    print(f"\n{'JOINT':<15} {'L_REF':>9} {'F_REF':>9} {moved_cols}{'SIGN':>5} {'OFFSET':>9}")
    for motor in motors:
        moved_vals = ""
        if cfg.measure_signs:
            d_leader, d_follower = l_moved[motor] - l_ref[motor], f_moved[motor] - f_ref[motor]
            moved_vals = f"{d_leader:>9.2f} {d_follower:>9.2f} "
            if min(abs(d_leader), abs(d_follower)) < cfg.min_travel_deg:
                undetermined.append(motor)
                print(f"{motor:<15} {l_ref[motor]:>9.2f} {f_ref[motor]:>9.2f} {moved_vals}{'?':>5} {'?':>9}")
                continue
            signs[motor] = 1.0 if d_leader * d_follower > 0 else -1.0
            if signs[motor] != cfg.robot.joint_signs[motor]:
                disagreements.append(motor)
        else:
            signs[motor] = cfg.robot.joint_signs[motor]
        offsets[motor] = f_ref[motor] - signs[motor] * l_ref[motor]
        print(
            f"{motor:<15} {l_ref[motor]:>9.2f} {f_ref[motor]:>9.2f} {moved_vals}"
            f"{signs[motor]:>+5.0f} {offsets[motor]:>9.2f}"
        )

    if undetermined:
        raise SystemExit(
            f"\nMoved less than {cfg.min_travel_deg:g} deg on at least one arm: {undetermined}. "
            "Re-run and move these joints further in step 2."
        )

    if disagreements:
        print(f"\n/!\\ Measured sign differs from --robot.joint_signs for: {disagreements}")

    print("\nAdd to the follower's command line (or paste into config_damiao_follower.py defaults):")
    if cfg.measure_signs:
        print(f"    --robot.joint_signs='{_fmt_dict(signs)}' \\")
    print(f"    --robot.joint_offsets='{_fmt_dict(offsets)}'")


if __name__ == "__main__":
    measure_joint_alignment()
