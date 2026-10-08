# Clean Bot Simulation with ROS 2 Jazzy and Gazebo Harmonic

Use this as a starting point of development, it includes a basic ROS2 package structure with an example chassis frame that can be extended upon.  
**Note:** Releases that I am using are **ROS 2 Jazzy (LTS)** and **Gazebo Harmonic (LTS)**.

## 🛠️ Setup Instructions

Follow these steps to get started:

1. **Source ROS 2 Installation**

   ```bash
   source /opt/ros/jazzy/setup.bash
   ```

2. **Create a Workspace**

   Create a directory with a name of your choice (e.g., `clean_bot_ws`) and navigate into it:

   ```bash
   mkdir -p ~/clean_bot_ws/src
   cd ~/clean_bot_ws/src
   ```

3. **Clone the Repository**

   Clone this repository **inside the `src/` folder** (this is important):

   ```bash
   git clone <repository-url>
   ```

4. **Build the Workspace**

   Move to the root of the workspace and build it:

   ```bash
   cd ..
   colcon build --symlink-install
   ```

5. **Source the Workspace**

   ```bash
   source ./install/setup.bash
   ```

## 🚀 Running the Simulation

1. **Run the Robot State Publisher**

   ```bash
   ros2 launch clean_bot rsp.launch.py
   ```

2. **Open RViz in a New Terminal**

   Open a new terminal and source both ROS 2 and your workspace:

   ```bash
   source /opt/ros/jazzy/setup.bash
   source ~/clean_bot_ws/install/setup.bash
   rviz2
   ```

3. **Configure RViz**

   - Set the **Fixed Frame** to `base_link`
   - Click **Add** and include the following:
     - **TF** viewer
     - **RobotModel** viewer
   - Set the **Description Topic** of the Robot Model to `/robot_description`

## 📌 Notes

- Always source both ROS 2 and your workspace in every new terminal.
- Cloning the repository inside the `src/` folder is required for `colcon` to detect and build it.
