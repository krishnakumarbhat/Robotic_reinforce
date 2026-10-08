import numpy as np
import time
from typing import Dict, List, Any

# --- Constants and Configuration ---
# Thresholds for detecting motion that is NOT the robot itself
MOTION_THRESHOLD_M = 0.1 # Minimum movement (meters) to flag as dynamic
IMU_COMPENSATION_THRESHOLD = 0.05 # How closely IMU data cancels out observed point cloud motion

# --- Data Structures ---
# Dictionary to hold tracked dynamic objects
# Key: object_id (int)
# Value: Dict[str, Any]
# 'history': List of (timestamp, centroid_position, velocity)
# 'predicted_pos': The object's estimated next position
DynamicObjects = Dict[int, Dict[str, Any]]


class DynamicObjectTracker:
    def __init__(self):
        print("--- Initializing Dynamic Object Tracking Pipeline (Project 3) ---")
        self.tracked_objects: DynamicObjects = {}
        self.next_object_id = 1
        # Mock class for the Mapper to prevent integration of dynamic points
        self.static_mapper = MockStaticMapper()

class MockStaticMapper:
    """Simulates the map integration logic (like the RTAB-Map component)."""
    def __init__(self):
        self.last_static_points_count = 0
        
    def integrate_static_points(self, static_points: np.ndarray):
        """Only integrates points that are confirmed static."""
        self.last_static_points_count = len(static_points)
        print(f"   [Mapper] Integrated {len(static_points)} points into the static map.")
        
    def reject_dynamic_points(self, dynamic_points: np.ndarray, labels: List[int]):
        """Logs and rejects dynamic points to maintain map integrity."""
        # This prevents 'ghosting' or blurring in the static 3D map.
        unique_labels = set(labels)
        print(f"   [Mapper] Rejected {len(dynamic_points)} dynamic points belonging to object IDs: {unique_labels}")


class DynamicObjectTracker:
    def __init__(self):
        print("--- Initializing Dynamic Object Tracking Pipeline (Project 3) ---")
        self.tracked_objects: DynamicObjects = {}
        self.next_object_id = 1
        self.static_mapper = MockStaticMapper()
        
        # Store the last processed data for motion calculation
        self._last_points = None
        self._last_time = time.time()
        self._last_imu = np.array([0.0, 0.0, 0.0]) # Linear acceleration components (mock)


    def _process_depth_for_motion(self, depth_frame: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulates:
        1. Converting depth to a 3D point cloud.
        2. Comparing the current points to the last frame to find movement clusters.
        3. Simple background subtraction.
        """
        current_time = time.time()
        
        # Mock: Generate a random set of 3D points
        num_points = depth_frame.size // 100 
        current_points = np.random.rand(num_points, 3) * 5 # Random points in 5m cube
        
        if self._last_points is None:
            self._last_points = current_points
            return {'static': current_points, 'dynamic': np.array([]).reshape(0, 3)}

        # Conceptual motion detection using point cloud distance (e.g., ICP or simple subtraction)
        
        # Mock logic: randomly assign a portion of points as 'dynamic'
        split_idx = int(num_points * np.random.uniform(0.05, 0.15)) # 5% to 15% are moving
        
        dynamic_points = current_points[:split_idx]
        static_points = current_points[split_idx:]
        
        # Store for next iteration
        self._last_points = current_points
        self._last_time = current_time
        
        return {'static': static_points, 'dynamic': dynamic_points}

    def _compensate_for_robot_motion(self, dynamic_points: np.ndarray, imu_data: np.ndarray) -> np.ndarray:
        """
        Uses IMU data (linear acceleration/velocity) to determine if observed 
        motion is due to the robot moving or an external object moving.
        """
        # IMU data is [Accel_X, Accel_Y, Accel_Z] or similar.
        
        # Mock logic: Assume if IMU reading is high, any observed point cloud motion 
        # is largely due to robot motion and would be handled by the SLAM (RTAB-Map) loop.
        
        # However, for *Dynamic Object Tracking*, we assume the SLAM/IMU fusion
        # already corrects the camera pose. What's left is relative motion.
        
        # Mock Logic for filtering:
        if np.linalg.norm(imu_data - self._last_imu) > IMU_COMPENSATION_THRESHOLD:
            # If the robot accelerated significantly, motion tracking becomes harder/less reliable
            print("   [IMU] Robot accelerated. Tracking stability temporarily reduced.")
            
        self._last_imu = imu_data
        
        # In a real system, points whose *apparent* velocity matches the *inverse* # of the robot's calculated velocity (from IMU/Odometry) are static.
        # Points whose apparent velocity *does not* match are truly dynamic.
        
        # For this conceptual code, we just return the points for tracking:
        return dynamic_points
        

    def track_dynamic_objects(self, depth_frame: np.ndarray, imu_data: np.ndarray):
        """
        Main method combining depth and IMU data to detect and track moving objects.
        """
        print("\n--- Starting Dynamic Object Tracking Step (Project 3) ---")
        
        # 1. Detect Motion using Depth/Point Cloud
        motion_data = self._process_depth_for_motion(depth_frame)
        dynamic_points = motion_data['dynamic']
        static_points = motion_data['static']

        if dynamic_points.size == 0:
            print("   [Tracker] No significant external motion detected.")
            # Even if no external motion is found, integrate all points as static.
            self.static_mapper.integrate_static_points(static_points)
            return
            
        # 2. Compensate for Robot's Own Motion (using IMU)
        external_motion_points = self._compensate_for_robot_motion(dynamic_points, imu_data)
        
        if external_motion_points.size == 0:
            print("   [Tracker] Observed motion was successfully compensated by IMU (robot self-motion).")
            self.static_mapper.integrate_static_points(static_points)
            return

        # 3. Cluster and Track Objects (e.g., using DBSCAN or Kalman Filter)
        # Mock: Assume all external motion belongs to one object for simplicity
        object_id = 99
        self.next_object_id += 1 

        # Conceptual calculation of object centroid and velocity
        centroid = np.mean(external_motion_points, axis=0)
        velocity = np.array([np.random.uniform(-0.5, 0.5), 0, 0]) # Mock velocity (e.g., person walking)
        
        if object_id not in self.tracked_objects:
            self.tracked_objects[object_id] = {'history': [], 'class': 'unknown', 'last_seen': time.time()}
            
        self.tracked_objects[object_id]['history'].append((time.time(), centroid, velocity))
        self.tracked_objects[object_id]['predicted_pos'] = centroid + velocity * 1.0 # Simple prediction
        
        print(f"   [Tracker] Detected & Tracking Dynamic Object {object_id}:")
        print(f"     - Centroid: {centroid[:2].round(2)}m | Velocity: {velocity[0].round(2)} m/s")
        
        # 4. Filter Dynamic Points from Static Map Integration
        # The key integration point: we DO NOT send these points to the RTAB-Map core for SLAM.
        self.static_mapper.integrate_static_points(static_points)
        self.static_mapper.reject_dynamic_points(external_motion_points, [object_id] * len(external_motion_points))
        

# --- Execution Example ---
if __name__ == "__main__":
    tracker = DynamicObjectTracker()
    
    # Simulate first frame (robot starts moving, but no external object yet)
    print("\n[Frame 1: Robot starts]")
    mock_depth_1 = np.ones((480, 640)) 
    mock_imu_1 = np.array([0.1, 0.0, 9.8]) # Slight acceleration from start
    tracker.track_dynamic_objects(mock_depth_1, mock_imu_1)
    
    time.sleep(0.5)
    
    # Simulate second frame (external object moves)
    print("\n[Frame 2: External object moves]")
    # We use a larger depth frame mock to ensure the motion detection finds something
    mock_depth_2 = np.ones((480, 640)) * 2 
    mock_imu_2 = np.array([0.05, 0.0, 9.8]) # Robot stabilizing
    tracker.track_dynamic_objects(mock_depth_2, mock_imu_2)
    
    # Check the tracked objects list
    if tracker.tracked_objects:
        obj_id = list(tracker.tracked_objects.keys())[0]
        print(f"\n✅ Final Tracked Object Count: {len(tracker.tracked_objects)}")
        print(f"   Object {obj_id} last predicted position: {tracker.tracked_objects[obj_id]['predicted_pos'].round(2)}")