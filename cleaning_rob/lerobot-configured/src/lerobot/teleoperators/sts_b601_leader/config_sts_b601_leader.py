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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class STSB601LeaderConfig:
    """Configuration for a 6-DOF leader arm (no gripper) built from Feetech STS3215 servos.

    Joint names follow the reBot B601 arm joints, so this leader pairs key-for-key with
    `damiao_follower` with no processor step in between. The chain ends at `wrist_roll`.
    """

    # Port to connect to the arm, e.g. "/dev/ttyACM0"
    port: str

    # Whether to use degrees for angles. The follower maps leader degrees onto its own motor
    # degrees with a sign and offset only, so leave this on when driving `damiao_follower`.
    use_degrees: bool = True

    # Number of extra attempts when a `sync_read` of the motors fails. Feetech buses can occasionally
    # return a corrupted status packet ("Incorrect status packet!"), especially when several joints move
    # at once, which otherwise aborts the teleoperation loop. Retries are immediate (no sleep) and only
    # happen on failure, so the steady-state read cost is unchanged.
    num_read_retries: int = 2

    # Joints that turn freely through a full revolution. These are skipped during the range-of-
    # motion sweep and given the encoder's whole 0-4095 span instead, since there are no stops to
    # find. Empty the list to sweep every joint.
    full_turn_motors: list[str] = field(default_factory=lambda: ["wrist_roll"])


@TeleoperatorConfig.register_subclass("sts_b601_leader")
@dataclass
class STSB601LeaderTeleopConfig(TeleoperatorConfig, STSB601LeaderConfig):
    pass
