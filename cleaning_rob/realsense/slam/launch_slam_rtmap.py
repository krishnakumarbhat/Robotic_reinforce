#!/usr/bin/env python3
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import List, Tuple

CAMERA_COMMAND: List[str] = [
    "ros2",
    "launch",
    "realsense2_camera",
    "rs_launch.py",
    "camera_name:=camera",
    "align_depth.enable:=true",
    "enable_gyro:=true",
    "enable_accel:=true",
]

RTABMAP_COMMAND: List[str] = [
    "ros2",
    "launch",
    "rtabmap_launch",
    "rtabmap.launch.py",
    "rgb_topic:=/camera/camera/color/image_raw",
    "depth_topic:=/camera/camera/aligned_depth_to_color/image_raw",
    "camera_info_topic:=/camera/camera/color/camera_info",
    "imu_topic:=/camera/camera/imu",
    "approx_sync:=true",
    "use_sim_time:=false",
    "qos_image:=2",  # Use SensorData QoS for image topics
    "qos_imu:=2",    # Use SensorData QoS for IMU
]

# Command to publish the static transform between the robot's base and the camera
STATIC_TRANSFORM_COMMAND: List[str] = [
    "ros2",
    "run",
    "tf2_ros",
    "static_transform_publisher",
    # "x y z yaw pitch roll frame_id child_frame_id"
    "0", "0", "0", "-1.57079632679", "0", "-1.57079632679",
    "base_link",
    "camera_link",
]

TERMINAL_ENV_VAR = "ROS_LAUNCH_TERMINAL"
DEFAULT_TERMINAL = "gnome-terminal"
CAMERA_WARMUP_SECONDS = int(os.getenv("CAMERA_WARMUP_SECONDS", "5"))


def ensure_terminal_available(terminal: str) -> None:
    if shutil.which(terminal) is None:
        raise FileNotFoundError(f"Terminal emulator '{terminal}' was not found on PATH.")


def build_terminal_invocation(terminal: str, command: List[str]) -> List[str]:
    joined = shlex.join(command)
    if terminal == "gnome-terminal":
        return [terminal, "--", "bash", "-c", f"{joined}; exec bash"]
    if terminal == "konsole":
        return [terminal, "--hold", "-e", joined]
    if terminal == "xterm":
        return [terminal, "-hold", "-e", joined]
    raise ValueError(
        f"Unsupported terminal emulator '{terminal}'. Update the script or set {TERMINAL_ENV_VAR}."
    )


def launch_in_terminal(label: str, terminal: str, command: List[str]) -> subprocess.Popen:
    print(f"Starting {label} using {terminal}...")
    invocation = build_terminal_invocation(terminal, command)
    return subprocess.Popen(invocation)


def monitor_processes(processes: List[Tuple[str, subprocess.Popen]]) -> None:
    try:
        while any(proc.poll() is None for _, proc in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        raise


def main() -> None:
    """Launch the RealSense camera and RTAB-Map nodes in dedicated terminals."""

    terminal = os.environ.get(TERMINAL_ENV_VAR, DEFAULT_TERMINAL)

    try:
        ensure_terminal_available(terminal)
    except FileNotFoundError as err:
        print(f"\nERROR: {err}")
        print(
            "Install the requested terminal emulator or set "
            f"{TERMINAL_ENV_VAR} to one of: gnome-terminal, konsole, xterm."
        )
        sys.exit(1)

    processes: List[Tuple[str, subprocess.Popen]] = []

    try:
        processes.append(("RealSense camera", launch_in_terminal("RealSense camera", terminal, CAMERA_COMMAND)))

        print(f"Waiting {CAMERA_WARMUP_SECONDS} seconds for the camera to initialise...")
        time.sleep(CAMERA_WARMUP_SECONDS)

        processes.append(("Static Transform", launch_in_terminal("Static Transform", terminal, STATIC_TRANSFORM_COMMAND)))
        processes.append(("RTAB-Map", launch_in_terminal("RTAB-Map", terminal, RTABMAP_COMMAND)))

        print("\nLaunch script finished. Check the new terminal windows.")
        print(
            "Press Ctrl+C in this window to stop monitoring; the launched "
            "terminals will remain open."
        )

        monitor_processes(processes)

    except KeyboardInterrupt:
        print("\nScript interrupted by user. The launched terminals will remain open.")
    except (FileNotFoundError, ValueError) as err:
        print(f"\nERROR: {err}")
    finally:
        # Do not terminate the child terminals; simply report their status.
        for label, process in processes:
            if process.poll() is None:
                continue
            print(f"Process '{label}' exited with code {process.returncode}.")


if __name__ == "__main__":
    main()