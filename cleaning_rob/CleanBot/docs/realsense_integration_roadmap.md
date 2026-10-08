# RealSense D435i Integration Roadmap

Integration plan for an Intel RealSense D435i camera (static/eye-to-hand mount) into the CleanBot workspace for object detection, 3D localization, and MoveIt/MTC-driven pick-and-place.

> **Scope:** Real hardware first — no Gazebo camera simulation.

---

## Phase 1: Camera Driver & ROS2 Topics

### Goal
Get the RealSense D435i publishing RGB, depth, and point cloud data as ROS2 topics.

### Steps

1. **Install the `realsense2_camera` ROS2 wrapper**
   ```bash
   sudo apt install ros-${ROS_DISTRO}-realsense2-camera ros-${ROS_DISTRO}-realsense2-description
   ```

2. **Create a launch file** (e.g. `src/clean_bot/launch/realsense.launch.py`) that starts the RealSense node with project-specific parameters:
   ```python
   from launch import LaunchDescription
   from launch_ros.actions import Node

   def generate_launch_description():
       return LaunchDescription([
           Node(
               package='realsense2_camera',
               executable='realsense2_camera_node',
               name='camera',
               namespace='camera',
               parameters=[{
                   'enable_color': True,
                   'enable_depth': True,
                   'enable_infra1': False,
                   'enable_infra2': False,
                   'align_depth.enable': True,
                   'pointcloud.enable': True,
                   'rgb_camera.color_profile': '640x480x30',   # adjust as needed
                   'depth_module.depth_profile': '640x480x30',
               }],
               output='screen',
           ),
       ])
   ```

3. **Verify topics** are publishing:
   ```bash
   ros2 topic list | grep camera
   # Expected:
   #   /camera/color/image_raw
   #   /camera/depth/image_rect_raw
   #   /camera/aligned_depth_to_color/image_raw
   #   /camera/depth/color/points
   #   /camera/color/camera_info

   ros2 topic hz /camera/color/image_raw          # confirm ~30 Hz
   ros2 topic hz /camera/depth/color/points        # confirm publishing
   ```

4. **View in RViz** — add Image and PointCloud2 displays to confirm data quality.

### Verification
- All four topic categories (RGB, depth, aligned depth, point cloud) are publishing at the configured FPS.
- Image data is visually correct in `rqt_image_view` or RViz.

---

## Phase 2: Camera-to-World Calibration (Eye-to-Hand)

### Goal
Establish the static transform from `camera_link` to the `world` frame so that 3D points from the camera can be expressed in the robot's coordinate system.

### Steps

1. **Physically mount the camera** with a clear, unobstructed view of the robot's workspace.

2. **Determine the static transform** (`camera_link` → `world`). Two options:

   | Approach | Pros | Cons |
   |----------|------|------|
   | **A: Manual measurement** — measure XYZ translation and orientation with a ruler/protractor, publish via `static_transform_publisher` | Simple, no extra deps | Less accurate, tedious to iterate |
   | **B: Calibration tool** — use `easy_handeye2` or ArUco marker-based calibration | Sub-cm accuracy, repeatable | Requires printing markers, extra setup |

   **Recommended:** Start with Option A for a quick sanity check, then refine with Option B.

3. **Publish the static TF.** Add to the camera launch file (or a separate one):
   ```python
   Node(
       package='tf2_ros',
       executable='static_transform_publisher',
       name='camera_to_world_tf',
       arguments=[
           '--x', '0.5',    # measured translation
           '--y', '0.0',
           '--z', '0.6',
           '--roll', '0.0',
           '--pitch', '0.78',  # ~45 deg downward tilt
           '--yaw', '3.14',
           '--frame-id', 'world',
           '--child-frame-id', 'camera_link',
       ],
   ),
   ```

4. **Verify in RViz** — display the robot model + point cloud in the `world` fixed frame. The point cloud of the physical workspace should align with the robot model's surroundings.

### Verification
- In RViz (fixed frame = `world`), the point cloud of the table/workspace aligns with the robot base.
- TF tree shows `world → camera_link` with the correct transform (`ros2 run tf2_tools view_frames`).

---

## Phase 3: Object Detection (RGB)

### Goal
Detect objects (starting with the cup) in the RGB image and publish 2D bounding boxes.

### Steps

1. **Choose a detection approach:**

   | Approach | When to use |
   |----------|-------------|
   | **YOLOv8/YOLOv11 via `ultralytics`** (recommended) | General-purpose detection of common objects (cups, bottles, etc.) |
   | **Color/contour-based** (OpenCV) | Single known object with distinctive color, simpler but fragile |

2. **Create a new ROS2 package** `src/perception/`:
   ```
   src/perception/
   ├── perception/
   │   ├── __init__.py
   │   └── detection_node.py
   ├── launch/
   │   └── perception.launch.py
   ├── config/
   │   └── detection_params.yaml
   ├── package.xml
   ├── setup.py
   └── setup.cfg
   ```

3. **Implement the detection node** (`detection_node.py`):
   - Subscribe to `/camera/color/image_raw` (`sensor_msgs/Image`)
   - Run YOLOv8 inference (or OpenCV pipeline) on each frame
   - Publish results on `/detections` as `vision_msgs/Detection2DArray`
   - Publish an annotated debug image on `/detection_image` for visualization
   - Key parameters (via `detection_params.yaml`):
     - `model_path` (e.g. `yolov8n.pt`)
     - `confidence_threshold` (default: `0.5`)
     - `target_classes` (e.g. `['cup']`)

4. **Install dependencies:**
   ```bash
   pip install ultralytics           # YOLOv8
   sudo apt install ros-${ROS_DISTRO}-vision-msgs
   ```

### Verification
- Run the detection node, view `/detection_image` in `rqt_image_view` — bounding boxes should appear around detected objects.
- `ros2 topic echo /detections` shows `Detection2DArray` messages with correct class labels and bounding box coordinates.

---

## Phase 4: 3D Object Localization (Depth Projection)

### Goal
Convert 2D detections into 3D poses in the `world` frame using depth data and TF2.

### Steps

1. **Extend the `perception` package** with a localization node (or add to the detection node) that:
   - Subscribes to:
     - `/detections` (`vision_msgs/Detection2DArray`)
     - `/camera/aligned_depth_to_color/image_raw` (`sensor_msgs/Image`)
     - `/camera/color/camera_info` (`sensor_msgs/CameraInfo`)
   - For each detection:
     1. Extract the bounding box center pixel `(u, v)`
     2. Sample depth at `(u, v)` from the aligned depth image (use a small patch median to reduce noise)
     3. Back-project to 3D using camera intrinsics:
        ```
        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth
        ```
     4. Transform `camera_optical_frame` → `world` using `tf2_ros.Buffer.transform()`
   - Publish results on `/detected_objects` as `geometry_msgs/PoseStamped` (or `PoseArray`)

2. **Handle edge cases:**
   - Missing/zero depth at the target pixel → expand search region or skip
   - Depth noise → use median of a small patch (e.g. 5x5 pixels) around the center
   - Object at image edge → reject detections with center too close to borders
   - Multiple detections → publish a `PoseArray` or pick the highest-confidence one

3. **Use `message_filters`** to time-synchronize RGB detections with depth frames if running detection and localization as separate nodes.

### Verification
- `ros2 topic echo /detected_objects` — the published pose should match the real object position.
- Measure the real object position with a ruler and compare to the reported XYZ (expect < 2 cm error at typical workspace distances).
- Visualize the pose as a Marker in RViz overlaid on the point cloud.

---

## Phase 5: MTC Integration — Dynamic Object Poses

### Goal
Replace the hardcoded object pose in the MTC node with the detected pose from Phase 4.

### Current State (what changes)
In `src/mtc/src/mtc_node.cpp`, `setupPlanningScene()` currently hardcodes the cup pose:
```cpp
// mtc_node.cpp:114-119
geometry_msgs::msg::Pose pose;
pose.position.x = -0.3;
pose.position.y = 0.0;
pose.position.z = 0.033;
pose.orientation.w = 1.0;
object.pose = pose;
```

### Steps

1. **Add a subscriber** to `MTCTaskNode` for `/detected_objects`:
   ```cpp
   // New member variables
   rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr object_pose_sub_;
   std::optional<geometry_msgs::msg::PoseStamped> latest_object_pose_;
   std::mutex pose_mutex_;

   // In constructor
   object_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
       "/detected_objects", 10,
       [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
           std::lock_guard<std::mutex> lock(pose_mutex_);
           latest_object_pose_ = *msg;
       });
   ```

2. **Modify `setupPlanningScene()`** to use the detected pose:
   ```cpp
   void MTCTaskNode::setupPlanningScene() {
       // ... ground setup unchanged ...

       // Wait for a detection
       geometry_msgs::msg::PoseStamped detected_pose;
       {
           std::lock_guard<std::mutex> lock(pose_mutex_);
           if (!latest_object_pose_.has_value()) {
               RCLCPP_WARN(this->get_logger(), "No object detected yet, waiting...");
               return;
           }
           detected_pose = latest_object_pose_.value();
       }

       // Use detected pose instead of hardcoded values
       object.pose = detected_pose.pose;
       psi.applyCollisionObject(object);
   }
   ```

3. **Alternative: Service-based approach.** Instead of a topic subscriber, create a service (`perception/GetObjectPose`) that MTC calls on demand. This avoids stale data issues:
   ```
   # GetObjectPose.srv
   string object_class
   ---
   geometry_msgs::PoseStamped pose
   bool success
   ```

4. **Update the launch file** (`src/mtc/launch/mtc_node_launch.launch.py`) to include the camera and perception nodes so the full pipeline launches together.

5. **Consider grasp adaptation:** If the detected object orientation varies, `createTask()` may need to adjust the grasp approach direction (currently hardcoded as `-x` in `hand_frame`).

### Verification
- Place the cup at different positions in the workspace.
- Run the full pipeline (camera → detection → localization → MTC).
- Confirm MTC plans to the detected pose, not the old hardcoded `(-0.3, 0.0, 0.033)`.
- Verify planning succeeds for at least 3 different cup positions within the arm's reachable workspace.

---

## Phase 6: (Optional) Octomap / Point Cloud Collision Scene

### Goal
Give MoveIt2 awareness of arbitrary obstacles by consuming the camera point cloud as an Octomap.

### Steps

1. **Create `src/clean_bot_moveit_config/config/sensors_3d.yaml`:**
   ```yaml
   sensors:
     - sensor_plugin: occupancy_map_monitor/PointCloudOctomapUpdater
       point_cloud_topic: /camera/depth/color/points
       max_range: 2.0
       point_subsample: 1
       padding_offset: 0.03
       padding_scale: 1.0
       max_update_rate: 5.0
       filtered_cloud_topic: /filtered_cloud
   ```

2. **Pass sensor config to `move_group`** in the bringup launch file. Add to the `move_group` node parameters:
   ```python
   # In headless_bringup.launch.py or equivalent
   moveit_sensors_config = PathJoinSubstitution([
       FindPackageShare('clean_bot_moveit_config'), 'config', 'sensors_3d.yaml'
   ])
   # Add to move_group node parameters
   ```

3. **Filter out the robot itself** from the point cloud — MoveIt's self-filter or setting `filtered_cloud_topic` handles this.

4. **Tune Octomap resolution** (`octomap_resolution` parameter on `move_group`, typically 0.01–0.05 m) — lower = more precise but slower.

### Verification
- In RViz, enable the `PlanningScene` display and check "Show Scene Geometry" — you should see a voxelized representation of obstacles.
- Place an obstacle (not the target object) in the workspace and confirm MoveIt plans around it.

---

## Files Summary

| Action | Path | Phase |
|--------|------|-------|
| **New** | `src/clean_bot/launch/realsense.launch.py` | 1, 2 |
| **New** | `src/perception/` (full package) | 3, 4 |
| **Modified** | `src/mtc/src/mtc_node.cpp` | 5 |
| **Modified** | `src/mtc/package.xml` (add `geometry_msgs` dep) | 5 |
| **Modified** | `src/mtc/launch/mtc_node_launch.launch.py` | 5 |
| **New** | `src/clean_bot_moveit_config/config/sensors_3d.yaml` | 6 |
| **Modified** | Bringup launch file (sensor config param) | 6 |

## Dependencies to Install

```bash
# Phase 1
sudo apt install ros-${ROS_DISTRO}-realsense2-camera ros-${ROS_DISTRO}-realsense2-description

# Phase 3
sudo apt install ros-${ROS_DISTRO}-vision-msgs
pip install ultralytics  # or opencv-python for contour-based approach

# Phase 4
sudo apt install ros-${ROS_DISTRO}-tf2-geometry-msgs  # likely already present

# Phase 6
# octomap support is bundled with MoveIt2, no extra install needed
```
