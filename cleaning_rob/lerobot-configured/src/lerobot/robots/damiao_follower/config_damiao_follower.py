#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@dataclass
class DamiaoFollowerConfigBase:
    """Configuration for a 6-DOF follower arm (no gripper) built from Damiao motors on a CAN bus.

    All motors listed in `motor_config` are enabled and commanded every cycle, but only the
    joints in `active_joints` follow the teleoperator; the rest are parked at their calibrated
    zero so they stay rigid. Motor names follow the reBot B601 arm joints
    (`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll`), which the
    matching `sts_b601_leader` also uses, so action keys map across key-for-key with no processor
    step. The chain ends at `wrist_roll`.

    `motor_config` order is chain order, base to tip, and the send ids run 0x01-0x06 to match.
    Ids need not be sequential in general, but keeping them aligned with the chain makes `candump`
    traces readable. Replies are matched by `recv_id`, or failing that by the motor-id nibble in
    the payload, so a motor answering on another feedback id (e.g. 0x000) is still recognized.
    That fallback needs the send ids' low nibbles to be unique.
    """

    # CAN channel, e.g. "vcan0" (bridged adapter), "can0" (native socketcan)
    port: str = "vcan0"

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # Classic CAN 2.0 by default; most USB-CAN adapters and bridges don't carry CAN FD frames
    use_can_fd: bool = False
    can_bitrate: int = 1_000_000
    can_data_bitrate: int | None = None  # only used when use_can_fd is True

    # How long to wait for motor replies, in seconds. A software bridge such as the vcan0 <->
    # CANalyst-II converter adds ~20 ms per round trip, far beyond the bus's built-in 1-10 ms windows,
    # so replies would otherwise land a cycle late. Waits end once every reply is in, so this only
    # costs its full length when a motor is silent. None = the bus's built-in windows.
    can_response_timeout_s: float | None = 0.05

    # Whether to disable torque when disconnecting
    disable_torque_on_disconnect: bool = True

    # When True, expose `.vel` and `.torque` per motor in observation features.
    use_velocity_and_torque: bool = False

    # Safety limit for relative target positions, in degrees per control tick.
    # Set to a positive scalar for all motors, or a dict mapping motor names to limits.
    # /!\ In MIT mode this is NOT just a speed limit: torque is kp * (goal - present), so capping
    # how far the goal may sit from the present position also caps the torque at kp * limit. At
    # kp=25 and 3 deg that is ~1.3 Nm, too little to move the arm -- it then never moves, the error
    # never grows, and the joint deadlocks (seen 2026-09-18: only the unloaded wrist_roll moved).
    # Keep None unless the limit is large enough that kp * limit still exceeds the load.
    # /!\ It also costs an extra CAN read-back round trip per tick (~20 ms through the vcan0
    # bridge), which roughly halves the control rate.
    max_relative_target: float | dict[str, float] | None = None

    # Camera configurations
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Maps motor names to (send_can_id, recv_can_id, motor_type)
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "shoulder_pan": (0x01, 0x11, "dm4340"),
            "shoulder_lift": (0x02, 0x12, "dm4340"),
            "elbow_flex": (0x03, 0x13, "dm4340"),
            "wrist_flex": (0x04, 0x14, "dm4310"),
            "wrist_yaw": (0x05, 0x15, "dm4310"),
            "wrist_roll": (0x06, 0x16, "dm4310"),
        }
    )

    # Joints driven by the teleoperator. Motors not listed here are still powered and are
    # parked at their calibrated zero. None = every motor in `motor_config` is active.
    active_joints: list[str] | None = None

    # MIT control gains per motor name. These are the only source of kp/kd on the control path:
    # send_action reads them for every motor on every tick and encodes them into the MIT frame,
    # so an edit here takes effect on the next run with nothing to re-flash or re-calibrate.
    # Hard limits: kp is clamped to MIT_KP_RANGE (0-500), kd to MIT_KD_RANGE (0-5) by
    # _encode_mit_packet -- out-of-range values are silently clamped, not rejected.
    # Tuning: raise kp in ~1.5x steps until tracking is crisp; on buzz or oscillation, back kp
    # off one step and raise kd by ~0.25. The DM4340 joints carry the arm's weight and will want
    # far more kp than the DM4310 wrist. The values below are `rebot_b601_follower`'s arm gains,
    # which drive the same motor class; kp=25/5 was too weak to lift this arm.
    position_kp: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 30.0,
            "shoulder_lift": 25.0,
            "elbow_flex": 25.0,
            "wrist_flex": 8.0,
            "wrist_yaw": 6.5,
            "wrist_roll": 6.5,
        }
    )
    position_kd: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 5.0,
            "shoulder_lift": 5.0,
            "elbow_flex": 5.0,
            "wrist_flex": 1.8,
            "wrist_yaw": 1.5,
            "wrist_roll": 1.5,
        }
    )

    # Per-joint (min, max) clip in degrees, in the follower motor frame. Entries here OVERRIDE
    # the ranges recorded during calibration -- use this to tighten a joint, not to describe the
    # arm. Joints absent from this dict are clipped to their calibrated range.
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Fallback clip used for a joint with neither a calibrated range nor a `joint_limits` entry.
    default_joint_limit_deg: float = 90.0

    # Safety margin trimmed off each end of a recorded range, so commands never reach a hard stop.
    range_margin_deg: float = 3.0

    # +1.0 / -1.0 per joint, for axes whose positive direction is mirrored w.r.t. the leader:
    # motor_deg = sign * leader_deg + offset. Below are the directions the user compared by hand on
    # 2026-09-17; `measure_joint_alignment` re-measures them if the arms are rebuilt.
    joint_signs: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": -1.0,
            "shoulder_lift": -1.0,
            "elbow_flex": 1.0,
            "wrist_flex": 1.0,
            "wrist_yaw": -1.0,
            "wrist_roll": 1.0,
        }
    )

    # Follower motor angle (deg, from its calibrated zero) that corresponds to the leader
    # reading 0.0 deg on that joint: offset = F_ref - sign * L_ref, measured with both arms in
    # the same reference pose. Re-measure whenever either arm is re-calibrated.
    # Measured with `measure_joint_alignment` on 2026-09-18, against follower calibration
    # damiao_follower_01 and leader calibration sts_b601_leader_01.
    joint_offsets: dict[str, float] = field(
        default_factory=lambda: {
            "shoulder_pan": 20.10,
            "shoulder_lift": -104.85,
            "elbow_flex": -97.98,
            "wrist_flex": 18.86,
            "wrist_yaw": 9.04,
            "wrist_roll": -34.72,
        }
    )


@RobotConfig.register_subclass("damiao_follower")
@dataclass
class DamiaoFollowerConfig(RobotConfig, DamiaoFollowerConfigBase):
    pass
