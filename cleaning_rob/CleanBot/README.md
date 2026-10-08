# CleanBot Workspace

> ⚠️ **Important:** This workspace uses a **single git submodule** for `clean_bot`.  
> All other dependencies are **not submodules** and must be cloned manually.

This repository is intended to be used as a **ROS 2 Jazzy workspace**.

---

## Repository Structure Overview

- `clean_bot`  
  - Tracked as a **git submodule**
  - Version-locked to this workspace

- MoveIt and related packages  
  - **Not** tracked as submodules
  - Must be cloned manually
  - Branch selection is ROS-distro–specific

This mixed approach is intentional:
- `clean_bot` is tightly coupled to this workspace
- MoveIt packages require frequent local modification, debugging, and branch control

---

## Prerequisites

- Ubuntu compatible with **ROS 2 Jazzy**
- ROS 2 **Jazzy** installed and sourced
- `git`
- `colcon`
- `rosdep`

---

## Installation Procedure

### 1. Clone the Workspace (with submodule)

```bash
git clone --recurse-submodules git@github.com:LeevAI-Devs/CleanBot.git
cd CleanBot
````

If you already cloned without submodules:

```bash
git submodule update --init
```

---

### 2. Clone Required Non-Submodule Dependencies

All non-submodule dependencies must be cloned **manually** into the workspace `src/` directory.

```bash
cd src
```

> ⚠️ The repositories and branches listed below are **specific to ROS 2 Jazzy**.
> Other ROS distributions may require different branches.

#### MoveIt 2 (Jazzy)

```bash
git clone -b jazzy https://github.com/moveit/moveit2
```

#### MoveIt Messages

```bash
git clone -b ros2 https://github.com/moveit/moveit_msgs
```

#### MoveIt Resources

```bash
git clone https://github.com/moveit/moveit_resources
```

#### MoveIt Task Constructor (Jazzy)

```bash
git clone -b jazzy https://github.com/moveit/moveit_task_constructor
```

---

### 3. Install System Dependencies Using rosdep

From the **workspace root**:

```bash
cd ..
rosdep install --from-paths src --ignore-src -r -y
```

⚠️ If a required package is **missing from `src/`**, `rosdep` may install a binary fallback.
This can result in version mismatches or unexpected behavior.

---

### 4. Build the Workspace

```bash
colcon build --symlink-install
```

Source the workspace:

```bash
source install/setup.bash
```

---

## Working With This Workspace

* `clean_bot`

  * Managed as a git submodule
  * Changes must be committed and pushed from within the submodule
  * The parent repo tracks the submodule commit reference

* MoveIt-related packages

  * Normal git repositories
  * Free to modify, rebase, or replace per ROS distro
  * Not tracked by the parent repository

---

## ROS Distribution Compatibility

* ✅ Validated for **ROS 2 Jazzy**
* ⚠️ Other ROS distributions may require:

  * Different MoveIt branches
  * Dependency adjustments
  * Code changes or patches
