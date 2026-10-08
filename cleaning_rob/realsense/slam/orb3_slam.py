#!/usr/bin/env python3
"""Utility helpers to run ORB-SLAM3 with an Intel RealSense camera.

This script does **not** build ORB-SLAM3 for you, but it makes the typical
runtime workflow easier once the ROS wrapper and the ORB-SLAM3 binaries are
already compiled. It performs a set of sanity checks and launches the required
ROS nodes in dedicated terminals (or in the current process, if requested).

Example (default paths, opens gnome-terminal windows):

    python3 orb3_slam.py \
        --orbslam-root "$HOME/ORB_SLAM3" \
        --vocabulary Vocabulary/ORBvoc.txt \
        --settings Examples/ROS/ORB_SLAM3/Asus.yaml

To reuse an existing terminal session (no new windows), append
``--no-new-terminal``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional


def _expand(path: str, base: Optional[Path] = None) -> Path:
    """Resolve *path* relative to *base* (if provided) and expand ~."""

    if base is not None:
        candidate = (base / path).expanduser()
    else:
        candidate = Path(path).expanduser()
    return candidate.resolve()


def _ensure_exists(path: Path, description: str) -> None:
    """Abort with a readable message if *path* does not exist."""

    if not path.exists():
        sys.exit(f"Expected {description} at '{path}', but it does not exist. "
                 "Double-check the command-line arguments.")


def _check_command_available(command: str) -> None:
    """Ensure that *command* is present in PATH."""

    if shutil.which(command) is None:
        sys.exit(
            f"'{command}' was not found in PATH. Make sure the ROS environment "
            "is sourced (e.g. `source /opt/ros/noetic/setup.bash`) and the "
            "ORB-SLAM3 workspace is built."
        )


def _format_command(command: Iterable[str]) -> str:
    """Return a shell-escaped representation of *command* for logging."""

    return " ".join(shlex.quote(item) for item in command)


def launch_process(
    command: List[str],
    *,
    cwd: Optional[Path] = None,
    new_terminal: bool = True,
    title: Optional[str] = None,
) -> subprocess.Popen:
    """Start *command* either directly or in a new gnome-terminal window."""

    if new_terminal:
        term_cmd = [
            "gnome-terminal",
            "--",
            "bash",
            "-lc",
            f"{' '.join(shlex.quote(c) for c in command)}; exec bash",
        ]
        if title:
            term_cmd.insert(1, f"--title={title}")
        print(f"Launching in new terminal: {_format_command(command)}")
        return subprocess.Popen(term_cmd, cwd=str(cwd) if cwd else None)

    print(f"Launching process: {_format_command(command)}")
    return subprocess.Popen(command, cwd=str(cwd) if cwd else None)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch ORB-SLAM3 with a RealSense camera via ROS"
    )

    parser.add_argument(
        "--orbslam-root",
        default="~/ORB_SLAM3",
        help="Root directory of the ORB-SLAM3 repository (default: ~/ORB_SLAM3)",
    )

    parser.add_argument(
        "--vocabulary",
        default="Vocabulary/ORBvoc.txt",
        help="Path to ORB-SLAM3 vocabulary file (relative to --orbslam-root unless absolute)",
    )

    parser.add_argument(
        "--settings",
        default="Examples/ROS/ORB_SLAM3/Asus.yaml",
        help="Camera settings YAML for ORB-SLAM3 (relative to --orbslam-root unless absolute)",
    )

    parser.add_argument(
        "--ros-camera-launch",
        nargs=3,
        metavar=("PKG", "FILE", "ARG"),
        default=("realsense2_camera", "rs_camera.launch", "align_depth:=true"),
        help=(
            "Triple specifying the ROS launch package, file, and optional argument "
            "for the RealSense camera."
        ),
    )

    parser.add_argument(
        "--orbslam-executable",
        default="Stereo_Inertial",
        help="ORB-SLAM3 ROS executable to run with rosrun (e.g. Stereo, Stereo_Inertial)",
    )

    parser.add_argument(
        "--use-imu",
        action="store_true",
        help="Append 'true' to the rosrun command (required for *_Inertial modes)",
    )

    parser.add_argument(
        "--wait-camera",
        type=float,
        default=5.0,
        help="Seconds to wait after launching the camera before starting ORB-SLAM3",
    )

    parser.add_argument(
        "--no-new-terminal",
        action="store_true",
        help="Run commands in the current terminal instead of spawning gnome-terminal",
    )

    parser.add_argument(
        "--rosrun",
        default="rosrun",
        help="Override the rosrun command (default: rosrun).",
    )

    parser.add_argument(
        "--roslaunch",
        default="roslaunch",
        help="Override the roslaunch command (default: roslaunch).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    orbslam_root = _expand(args.orbslam_root)
    vocabulary = _expand(args.vocabulary, base=orbslam_root)
    settings = _expand(args.settings, base=orbslam_root)

    _ensure_exists(orbslam_root, "ORB-SLAM3 root directory")
    _ensure_exists(vocabulary, "vocabulary file")
    _ensure_exists(settings, "camera settings YAML")

    _check_command_available(args.roslaunch)
    _check_command_available(args.rosrun)

    camera_pkg, camera_file, camera_arg = args.ros_camera_launch
    camera_command = [args.roslaunch, camera_pkg, camera_file, camera_arg]

    orbslam_command = [
        args.rosrun,
        "ORB_SLAM3",
        args.orbslam_executable,
        str(vocabulary),
        str(settings),
    ]

    if args.use_imu or args.orbslam_executable.endswith("_Inertial"):
        orbslam_command.append("true")

    # Launch RealSense camera node
    camera_proc = launch_process(
        camera_command,
        new_terminal=not args.no_new_terminal,
        title="RealSense Camera",
    )

    try:
        print(f"Waiting {args.wait_camera:.1f}s for the camera to initialize...")
        time.sleep(max(args.wait_camera, 0.0))

        # Launch ORB-SLAM3 node
        orbslam_proc = launch_process(
            orbslam_command,
            cwd=orbslam_root,
            new_terminal=not args.no_new_terminal,
            title="ORB-SLAM3",
        )

        # Keep forwarding signals so Ctrl+C stops child processes too
        def forward_signal(signum, _frame):
            for proc in (camera_proc, orbslam_proc):
                if proc.poll() is None:
                    try:
                        proc.send_signal(signum)
                    except ProcessLookupError:
                        pass

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, forward_signal)

        print("Both processes launched. Use Ctrl+C to stop them.")
        orbslam_return = orbslam_proc.wait()
        print(f"ORB-SLAM3 exited with return code {orbslam_return}.")

    except KeyboardInterrupt:
        print("Interrupted by user. Terminating child processes...")
    finally:
        for proc in (camera_proc,):
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()