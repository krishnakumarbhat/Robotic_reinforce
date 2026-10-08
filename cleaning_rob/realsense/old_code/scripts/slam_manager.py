# slam_manager.py
import subprocess
import os
from launch.actions import ExecuteProcess
import time

class SlamManager:
    """
    Manages the lifecycle of the SLAM process.
    Acts as a Facade, simplifying the complex process of launching,
    monitoring, and exporting into a single `run()` method.
    """
    def __init__(self):
        self.db_path = os.path.expanduser("~/.ros/rtabmap.db")
        self.ply_path = os.path.expanduser("~/slam_map.ply")
        self.processes = []

    def _check_existing_db(self):
        """Checks for and offers to remove a pre-existing map database."""
        if os.path.exists(self.db_path):
            print(f"WARNING: An existing map database was found at {self.db_path}.")
            choice = input("Delete it and start a new map? (y/n): ").lower()
            if choice == 'y':
                os.remove(self.db_path)
                print("Previous map database deleted.")
                return True
            else:
                print("Aborting. Please move the old database file and try again.")
                return False
        return True

    def _launch_processes(self):
        """Launches Camera, RTAB-Map, and RViz in separate terminals."""
        # Use the more robust, centralized launch file from my_slam_session
        # This single command launches the camera, IMU filter, RTAB-Map, and the visualizer.
        # Note: The package name 'my_slam_project' in the original launch file might need to be
        # adjusted to 'slam_robot_orchestrator' to match the actual package name.
        # For this example, we assume the launch file is correctly installed.
        # We will use the 'rslaunch.py' as it is the most complete one.
        slam_launch_cmd = "ros2 launch slam_robot_orchestrator rslaunch.py"
        
        print("Opening a terminal for the complete SLAM system...")
        print(f"Executing: {slam_launch_cmd}")
        
        # Using Popen to run the launch command in a new terminal.
        # This is still a simple way to give the user a separate window for ROS logs.
        slam_process = subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f"{slam_launch_cmd}; exec bash"])
        self.processes.append(slam_process)

    def _shutdown_processes(self):
        """Terminates all launched subprocesses."""
        print("\n--- Shutting down SLAM processes ---")
        for p in self.processes:
            p.terminate()
        time.sleep(5)  # Allow time for graceful shutdown
        print("All processes terminated.")

    def _export_point_cloud(self):
        """Exports the final map from the database to a .ply file."""
        if os.path.exists(self.db_path):
            print(f"\n--- Exporting map from {self.db_path} to {self.ply_path} ---")
            export_cmd = f"rtabmap-export -ply {self.db_path}"
            try:
                subprocess.run(export_cmd, shell=True, check=True, capture_output=True, text=True)
                print(f"✅ Success! Point cloud saved to {self.ply_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Failed to export the point cloud. Error: {e.stderr}")
        else:
            print("❌ ERROR: No map database was created. Cannot export.")

    def run(self):
        """The main public method to execute the entire SLAM lifecycle."""
        print("\n--- Starting SLAM Process ---")
        if not self._check_existing_db():
            return
        
        try:
            self._launch_processes()
            print("\n" + "="*50)
            print("✅ All systems are running.")
            print("➡️ ACTION REQUIRED: Physically move your camera to map the environment.")
            print("➡️ When you are finished, press Enter in THIS terminal.")
            print("="*50 + "\n")
            input() # Wait for user to finish mapping
        finally:
            self._shutdown_processes()
            self._export_point_cloud()