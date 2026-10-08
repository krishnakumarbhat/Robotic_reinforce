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

import logging
import math
import time
from functools import cached_property
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_damiao_follower import DamiaoFollowerConfig

logger = logging.getLogger(__name__)


class DamiaoFollower(Robot):
    """
    6-DOF follower arm (no gripper) using Damiao motors in MIT control mode over a CAN bus.

    Motor names follow the reBot B601 arm joints and match the paired leader's, so leader
    actions map onto this follower by key. Action keys for motors this robot doesn't have are
    silently dropped, and per-joint signs and offsets map the leader's degree frame onto the
    follower's.

    Every motor in `motor_config` is enabled and commanded each cycle, but only the joints in
    `config.active_joints` track the leader; the rest are held at their calibrated zero so the
    arm stays rigid where it isn't being teleoperated.
    """

    config_class = DamiaoFollowerConfig
    name = "damiao_follower"

    def __init__(self, config: DamiaoFollowerConfig):
        super().__init__(config)
        self.config = config

        for attr in ("position_kp", "position_kd", "joint_signs", "joint_offsets"):
            missing = set(config.motor_config) - set(getattr(config, attr))
            if missing:
                raise ValueError(f"config.{attr} is missing entries for motors: {sorted(missing)}")

        bad_signs = sorted(m for m, v in config.joint_signs.items() if v not in (1.0, -1.0))
        if bad_signs:
            raise ValueError(f"config.joint_signs must be +1.0 or -1.0. Invalid for motors: {bad_signs}")

        unknown_limits = set(config.joint_limits) - set(config.motor_config)
        if unknown_limits:
            raise ValueError(f"config.joint_limits names unknown motors: {sorted(unknown_limits)}")

        if config.active_joints is None:
            self._active = set(config.motor_config)
        else:
            unknown_active = set(config.active_joints) - set(config.motor_config)
            if unknown_active:
                raise ValueError(f"config.active_joints names unknown motors: {sorted(unknown_active)}")
            if not config.active_joints:
                raise ValueError("config.active_joints must name at least one motor.")
            self._active = set(config.active_joints)

        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Always use degrees for Damiao motors
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
            response_timeout_s=self.config.can_response_timeout_s,
        )

        self._limits: dict[str, tuple[float, float]] = {}
        self._refresh_limits()

        self.cameras = make_cameras_from_configs(config.cameras)

    def _refresh_limits(self) -> None:
        """Recompute the effective motor-frame clip for every motor.

        The calibrated range recorded by `calibrate()` is the base; a `config.joint_limits` entry
        intersects with it, so an explicit override can only ever tighten the safe range.
        """
        fallback = self.config.default_joint_limit_deg
        self._limits = {}
        for motor_name in self.config.motor_config:
            cal = self.calibration.get(motor_name)
            if cal is not None:
                lo, hi = float(cal.range_min), float(cal.range_max)
            else:
                lo, hi = -fallback, fallback

            override = self.config.joint_limits.get(motor_name)
            if override is not None:
                lo, hi = max(lo, float(override[0])), min(hi, float(override[1]))

            if hi < lo:
                raise ValueError(
                    f"Empty joint limit for '{motor_name}': [{lo}, {hi}]. The config.joint_limits "
                    "override does not overlap the calibrated range."
                )
            self._limits[motor_name] = (lo, hi)

    @property
    def _motors_ft(self) -> dict[str, type]:
        """Motor features for observation and action spaces."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            if self.config.use_velocity_and_torque:
                features[f"{motor}.vel"] = float
                features[f"{motor}.torque"] = float
        return features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        """Camera features for observation space."""
        features: dict[str, tuple] = {}
        for cam in self.cameras:
            cfg = self.config.cameras[cam]
            if getattr(cfg, "use_rgb", True):
                features[cam] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{cam}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Combined observation features from motors and cameras."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the robot and optionally calibrate.

        We assume that at connection time, the arm is in a safe rest position,
        and torque can be safely disabled to run calibration if needed.
        """
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()

        self.bus.enable_torque()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """True only when the calibration on file covers exactly this robot's motors.

        `DamiaoMotorsBus.is_calibrated` is merely `bool(self.calibration)`, so a file written for
        a different motor roster would load and leave the missing joints on the fallback limit.
        Comparing the key sets makes a stale file trigger a fresh calibration instead.
        """
        return bool(self.calibration) and set(self.calibration) == set(self.config.motor_config)

    def calibrate(self) -> None:
        """
        Run calibration procedure.

        Two steps. First the zero pose: every motor is zeroed at a physical reference pose that
        you can reproduce on the leader too, since `config.joint_offsets` is measured as the
        leader's reading in that same pose. Then the safe travel of each active joint is recorded
        by hand and stored, shrunk by `config.range_margin_deg` at both ends, as the motor-frame
        clip applied on every action.
        """
        if self.is_calibrated:
            # A calibration file for exactly these motors exists; ask whether to reuse it
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                self._refresh_limits()
                return

        logger.info(f"\nRunning calibration for {self}")
        self.bus.disable_torque()

        input(
            "\nCalibration: Set Zero Position\n"
            "Pose the follower in the reference pose you will also reproduce on the leader\n"
            "(the pose whose leader readings become config.joint_offsets).\n"
            "Press ENTER when ready..."
        )

        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        active = sorted(self._active)
        margin = self.config.range_margin_deg
        fallback = self.config.default_joint_limit_deg

        answer = input(
            "\nCalibration: Joint Ranges\n"
            f"Press ENTER to record the travel of {active} by hand,\n"
            f"or type 's' and press ENTER to skip and use +/-{fallback:g} deg. "
        )

        if answer.strip().lower() == "s":
            logger.info(f"Skipping range recording; using +/-{fallback:g} deg for active joints.")
            ranges = dict.fromkeys(active, (-fallback, fallback))
        else:
            mins, maxes = self.bus.record_ranges_of_motion(active)
            ranges = {}
            for motor in active:
                lo, hi = mins[motor] + margin, maxes[motor] - margin
                if hi <= lo:
                    raise ValueError(
                        f"Recorded range for '{motor}' ([{mins[motor]:.1f}, {maxes[motor]:.1f}] deg) is "
                        f"narrower than twice the {margin:g} deg safety margin. Sweep it further, or "
                        "lower config.range_margin_deg."
                    )
                ranges[motor] = (lo, hi)

        # Held joints only ever get commanded to their zero; give them a tight window around it.
        for motor in self.bus.motors:
            if motor not in self._active:
                ranges[motor] = (-margin, margin)

        self.calibration = {}
        for motor_name, motor in self.bus.motors.items():
            lo, hi = ranges[motor_name]
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                # MotorCalibration stores ints; round inward so the safe range never grows.
                range_min=math.ceil(lo),
                range_max=math.floor(hi),
            )
            logger.info(
                f"{motor_name}: range [{self.calibration[motor_name].range_min}, "
                f"{self.calibration[motor_name].range_max}] deg"
                + ("" if motor_name in self._active else " (held)")
            )

        self.bus.write_calibration(self.calibration)
        self._refresh_limits()
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """Configure motors with appropriate settings."""
        with self.bus.torque_disabled():
            self.bus.configure_motors()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Get current observation from robot including position, velocity, and torque.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle instead of
        3 separate reads. Positions are reported in the leader-aligned frame (joint_offsets
        removed, then joint_signs applied), consistent with the action space. Velocity and
        torque only take the sign -- an offset on a derivative is meaningless.
        """
        start = time.perf_counter()

        obs_dict: dict[str, Any] = {}

        states = self.bus.sync_read_all_states()

        for motor in self.bus.motors:
            state = states.get(motor, {})
            sign = self.config.joint_signs[motor]
            offset = self.config.joint_offsets[motor]
            obs_dict[f"{motor}.pos"] = sign * (state.get("position", 0.0) - offset)
            if self.config.use_velocity_and_torque:
                obs_dict[f"{motor}.vel"] = sign * state.get("velocity", 0.0)
                obs_dict[f"{motor}.torque"] = sign * state.get("torque", 0.0)

        for cam_key, cam in self.cameras.items():
            if getattr(cam, "use_rgb", True):
                start = time.perf_counter()
                obs_dict[cam_key] = cam.read_latest()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

            if getattr(cam, "use_depth", False):
                start = time.perf_counter()
                obs_dict[f"{cam_key}_depth"] = cam.read_latest_depth()
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read {cam_key} depth: {dt_ms:.1f}ms")

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} get_observation took: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send action command to robot.

        Every motor is commanded on every call. An active joint tracks its `<name>.pos` key,
        mapped into the motor frame as `sign * value + offset`; an inactive joint -- or an
        active one the teleoperator didn't send -- is held at its calibrated zero. Keys for
        motors this robot doesn't have are ignored.

        Args:
            action: Dictionary with motor positions in degrees, leader-aligned frame
                (e.g., "shoulder_pan.pos").

        Returns:
            The action actually sent (potentially clipped), leader-aligned frame.
        """
        leader_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Build a motor-frame target for every motor, then clip to the effective safe range.
        goal_pos: dict[str, float] = {}
        for motor_name in self.bus.motors:
            offset = self.config.joint_offsets[motor_name]
            if motor_name in self._active and motor_name in leader_pos:
                position = self.config.joint_signs[motor_name] * leader_pos[motor_name] + offset
            else:
                # Held: park at the calibrated zero.
                position = offset

            min_limit, max_limit = self._limits[motor_name]
            clipped_position = max(min_limit, min(max_limit, position))
            if clipped_position != position:
                logger.debug(f"Clipped {motor_name} from {position:.2f}° to {clipped_position:.2f}°")
            goal_pos[motor_name] = clipped_position

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Use batch MIT control (sends all commands, then collects responses)
        commands = {}
        for motor_name, position_degrees in goal_pos.items():
            kp = self.config.position_kp[motor_name]
            kd = self.config.position_kd[motor_name]
            commands[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)

        self.bus._mit_control_batch(commands)

        return {
            f"{motor}.pos": self.config.joint_signs[motor] * (val - self.config.joint_offsets[motor])
            for motor, val in goal_pos.items()
        }

    @check_if_not_connected
    def disconnect(self):
        """Disconnect from robot."""
        self.bus.disconnect(self.config.disable_torque_on_disconnect)

        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
