# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Source ROS2 environment first
source /opt/ros/<distro>/setup.bash

# Build project-specific packages (uses shell alias)
build

# Build all packages including vendored deps (uses shell alias)
build-all

# Build a single package and its dependencies
colcon build --packages-up-to <package_name>

# Source the workspace after building
source install/setup.bash

# Run all tests
colcon test && colcon test-result

# Run tests for a specific package
colcon test --packages-select <package_name>

# Run a single test
colcon test --packages-select <package_name> --ctest-args -R <test_name>
```

## Formatting

```bash
# C++ (clang-format, auto-applied by pre-commit in submodules)
clang-format -i src/<package>/src/**/*.cpp src/<package>/include/**/*.hpp

# Python (ruff, auto-applied by pre-commit in submodules)
ruff check --fix . && ruff format .
```

## Running the Robot

```bash
# Simulation bringup (robot_state_publisher, controllers, move_group, RViz)
ros2 launch clean_bot headless_bringup.launch.py is_sim:=true

# Gazebo simulation
ros2 launch clean_bot gz.launch.py

# MTC pick-and-place task (requires bringup running first)
ros2 launch mtc mtc_node_launch.launch.py

# Automated MTC demo (builds, launches bringup + MTC in sequence)
./scripts/run_mtc_setup.sh
```

## Architecture

This is a ROS2 colcon workspace for a 6-DOF robotic arm with a gripper, built around MoveIt2 for motion planning.

### Project Packages (under `src/`)

- **clean_bot** — Submodule (`LeevAI-Devs/clean_bot`, branch `main`). Robot description (URDF/XACRO), Gazebo worlds, launch files. The main entry point is `headless_bringup.launch.py` which orchestrates the full system startup: robot_state_publisher -> ros2_control -> controller spawners -> move_group -> RViz. Launch params: `is_sim` (simulation mode), `is_calib` (calibration mode), `is_deb` (GDB debug on port 3000).
- **clean_bot_controller** — Calibration system. The `calibration_node` subscribes to `/joint_states`, converts radians to servo ticks, and writes calibration YAML. Also contains the controller config (`clean_bot_controller.yaml`) defining the arm (JointTrajectoryController) and gripper (GripperActionController) at 50Hz update rate.
- **clean_bot_moveit_config** — MoveIt configuration: SRDF (joint groups, named poses, disabled collisions), joint limits, OMPL planning settings, KDL kinematics solver config, and MoveIt controller mappings.
- **clean_bot_moveit_cpp** — C++ motion planning interface using MoveGroupInterface. Two executables: `clean_bot_moveit_cpp` (collision object management + plan-and-execute) and `zero_pose_interactive` (interactive control).
- **mtc** — MoveIt Task Constructor node implementing pick-and-place. The current task loads a cup mesh (`cup.stl`) as the target object. Task pipeline: open hand -> move to pick -> approach object -> generate grasp pose with IK -> allow collisions -> close hand -> attach object. Uses OMPL, JointInterpolation, and CartesianPath planners.
- **feetech_ros2_driver** — ros2_control hardware interface plugin for Feetech servo motors. Handles serial communication and motor calibration. Contains a `feetech_driver/` subdirectory with the low-level serial communication library.

### Vendored Libraries (under `src/`)

`moveit2`, `moveit_msgs`, `moveit_resources`, `moveit_task_constructor` are full source checkouts built alongside project packages. `panda_bringup` exists but is unused (no package.xml).

### Robot Configuration

- **Arm group** (6 joints): shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_yaw_joint, wrist_pitch_joint, wrist_roll_joint
- **Gripper group**: gripper_jaw_joint (open: 0.95 rad, grasp: -0.18 rad, closed: -0.69 rad, neutral: 0.0 rad)
- **Named arm poses**: `zero` (all 0.0), `slouch` (predefined non-zero config)
- **End effector**: gripper, parent link `gripper_anchor_link`, IK frame `gripper_center_link`
- **Controllers**: arm_controller (position command, FollowJointTrajectory), gripper_controller (GripperCommand action)
- **Planning**: OMPL via MoveIt2, KDL IK solver (0.5s timeout, 5 attempts)

### Submodule Workflow

Only `src/clean_bot` is a registered git submodule. It must be committed separately, then the parent repo updated to track the new submodule commit. Always clone with `--recurse-submodules`.

## Code Style

- **C++**: Google style, 120-char line limit, Allman braces, left-aligned pointers (`int* ptr`). `CamelCase` classes, `snake_case` functions/variables, `kCamelCase` constants. No `using namespace`. C++17 for MoveIt packages (mtc, clean_bot_moveit_cpp), C++20 for feetech_ros2_driver, no explicit standard for clean_bot_controller.
- **Python**: ruff for linting/formatting, PEP 257 docstrings, `snake_case` functions, `CamelCase` classes.
- **Logging**: Use ROS2 logging macros (`RCLCPP_INFO`, `RCLCPP_ERROR`, etc.).
