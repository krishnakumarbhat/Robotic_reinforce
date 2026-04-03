#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from two_mouse_il.teleop import TeleopConfig, run_dual_mouse_teleop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record dual-mouse, dual-arm ManiSkill demonstrations.")
    parser.add_argument("--env-id", default="TwoRobotPickCube-v1", help="ManiSkill environment id.")
    parser.add_argument("--dataset", default="data/recordings/two_robot_pick_cube.h5", help="Output dataset H5 path.")
    parser.add_argument("--metadata", default=None, help="Optional metadata JSON path. Defaults next to the H5 file.")
    parser.add_argument("--seed", type=int, default=0, help="Initial environment seed.")
    parser.add_argument("--frequency", type=float, default=20.0, help="Control loop frequency in Hz.")
    parser.add_argument("--include-touchpad", action="store_true", help="Allow a touchpad to be used as a pointer device.")
    parser.add_argument("--left-device", default=None, help="Explicit event device path for the left-arm mouse.")
    parser.add_argument("--right-device", default=None, help="Explicit event device path for the right-arm mouse.")
    parser.add_argument("--grab", action="store_true", help="Grab both devices while collecting demonstrations.")
    parser.add_argument("--xy-scale", type=float, default=0.01, help="Normalized action scale applied to mouse X/Y deltas.")
    parser.add_argument("--z-scale", type=float, default=0.25, help="Normalized action scale applied to mouse wheel deltas.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)
    metadata_path = Path(args.metadata) if args.metadata else dataset_path.with_suffix(".json")
    config = TeleopConfig(
        env_id=args.env_id,
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        seed=args.seed,
        hz=args.frequency,
        include_touchpad=args.include_touchpad,
        left_event_path=args.left_device,
        right_event_path=args.right_device,
        grab_devices=args.grab,
        xy_action_scale=args.xy_scale,
        z_action_scale=args.z_scale,
    )
    return run_dual_mouse_teleop(config)


if __name__ == "__main__":
    raise SystemExit(main())
