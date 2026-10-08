# SLAM Package - Complete Module Guide

This directory contains all SLAM and mapping components for the ASMTCDR project.

## 📁 Module Overview

### Core Modules

#### 1. `ASMTCDR.py` - Integrated Mapping System (Projects 1-5)
**Purpose**: Core integrated system combining all five projects

**Key Classes**:
- `IntegratedMappingAndMonitoringSystem` - Main coordinator
- `MapperRTAB` - RTAB-Map SLAM interface
- `RealSenseDriver` - Sensor data acquisition (mock)
- `RobotController` - Navigation controller (mock)
- `MapUpdate` - SLAM update dataclass

**Usage**:
```python
from realsense.slam import IntegratedMappingAndMonitoringSystem

system = IntegratedMappingAndMonitoringSystem()
summary = system.start_autonomous_operation(duration_s=60)
```

**Projects**: 1 (Change Detection), 2 (Semantic), 3 (Tracking), 4 (Navigation), 5 (Mapping)

---

#### 2. `dynamic_obj_track.py` - Dynamic Object Tracking (Project 3)
**Purpose**: Track moving objects using depth + IMU data

**Key Classes**:
- `DynamicObjectTracker` - Main tracker with IMU compensation
- `MockStaticMapper` - Map integration interface

**Features**:
- Depth-based motion detection
- IMU compensation for robot motion
- Point cloud background subtraction
- Velocity estimation

**Usage**:
```python
from realsense.slam import DynamicObjectTracker

tracker = DynamicObjectTracker()
tracker.track_dynamic_objects(depth_frame, imu_data)
```

---

#### 3. `sematic_map.py` - Semantic Mapping Pipeline (Project 2)
**Purpose**: AI-powered semantic segmentation and 3D labeling

**Key Classes**:
- `SemanticMappingPipeline` - Core pipeline
- `SemanticMappingNode` - ROS 2 node
- `RTABMapAccumulator` - Point cloud aggregator
- `Detection2D` - 2D detection container
- `SemanticMapConfig` - Configuration

**Features**:
- YOLO/SAM integration
- 2D-to-3D projection
- Real-time point cloud labeling
- ROS topic subscription
- Mock detection fallback

**Usage**:
```python
from realsense.slam import SemanticMappingPipeline

pipeline = SemanticMappingPipeline(classes=["chair", "table"])
summary = pipeline.build_semantic_map_step(rgb, depth)
```

**ROS Usage**:
```bash
# With visualization
SEMANTIC_DISPLAY=1 python3 sematic_map.py
```

---

#### 4. `yolo_obj_slam.py` - YOLO Object Detection (Project 2 component)
**Purpose**: YOLOv8/v11 object detection for ROS integration

**Key Classes**:
- `YOLODetector` - YOLO wrapper
- `YoloObjSlamNode` - ROS 2 publisher
- `DetectorConfig` - Configuration

**Features**:
- Real-time object detection
- ROS image subscription
- Detection publishing to topics
- Optional visualization
- Configurable via environment variables

**Usage**:
```python
from realsense.slam import YOLODetector, DetectorConfig

config = DetectorConfig()
detector = YOLODetector(config)
detections = detector.infer(rgb_frame)
```

**ROS Usage**:
```bash
# With visualization
YOLO_DISPLAY=1 python3 yolo_obj_slam.py

# With custom model
YOLO_MODEL="yolov8m.pt" python3 yolo_obj_slam.py
```

---

#### 5. `room_map.py` - Room Mapping Logger
**Purpose**: Persistent logging of mapping sessions

**Key Classes**:
- `RoomMapLogger` - Session logger and persistence

**Features**:
- Map update logging
- Change event tracking
- JSON session export
- Artifact storage

**Usage**:
```python
from realsense.slam import RoomMapLogger, run_autonomous_room_mapping

# Quick run
summary = run_autonomous_room_mapping(duration_s=60)

# Advanced usage
logger = RoomMapLogger()
system.start_autonomous_operation(
    on_map_update=logger.handle_map_update,
    on_change_detected=logger.handle_change_event
)
logger.persist_session(summary)
```

---

#### 6. `launch_slam_rtmap.py` - ROS Launch Script (Project 5)
**Purpose**: Launch RealSense + RTAB-Map SLAM stack

**Launches**:
- Intel RealSense camera driver
- RTAB-Map SLAM node
- Static TF publisher (base_link → camera_link)

**Configuration**:
```bash
# Warmup time before RTAB-Map
export CAMERA_WARMUP_SECONDS=5

# Terminal preference
export ROS_LAUNCH_TERMINAL=gnome-terminal  # or xterm, konsole
```

**Usage**:
```bash
# Direct run
python3 launch_slam_rtmap.py

# Or as module
python3 -m realsense.slam.launch_slam_rtmap
```

---

### Utility Modules

#### 7. `open3d_slam.py` - Open3D SLAM Implementation
**Purpose**: Direct Open3D-based SLAM (alternative to RTAB-Map)

**Features**:
- Visual odometry
- Frame-to-frame tracking
- Live visualization
- Point cloud reconstruction

**Usage**:
```bash
python3 open3d_slam.py
```

---

#### 8. `orb3_slam.py` - ORB-SLAM3 Utilities
**Purpose**: Helper scripts for ORB-SLAM3 integration

**Features**:
- Sanity checks for ORB-SLAM3 installation
- ROS node launching
- Path validation

**Usage**:
```bash
python3 orb3_slam.py \
    --orbslam-root "$HOME/ORB_SLAM3" \
    --vocabulary Vocabulary/ORBvoc.txt \
    --settings Examples/ROS/ORB_SLAM3/Asus.yaml
```

---

#### 9. `demo_integrated.py` - Integrated Demo
**Purpose**: Comprehensive demo of all components

**Features**:
- All-in-one demonstration
- Individual component demos
- Command-line options
- No hardware required

**Usage**:
```bash
# Run all demos
python3 demo_integrated.py --all

# Integrated system only (60s)
python3 demo_integrated.py --integrated --duration 60

# Semantic mapping only
python3 demo_integrated.py --semantic

# Dynamic tracking only
python3 demo_integrated.py --tracking
```

---

## 🔗 Module Dependencies

```
ASMTCDR.py (Core)
    ├── Used by: room_map.py
    └── Dependencies: numpy, open3d

dynamic_obj_track.py
    └── Dependencies: numpy, open3d

sematic_map.py
    ├── Used by: ASMTCDR.py (conceptually)
    └── Dependencies: numpy, open3d, ROS 2 (optional), cv2 (optional)

yolo_obj_slam.py
    ├── Publishes to: sematic_map.py (via ROS topic)
    └── Dependencies: ultralytics, ROS 2 (optional), cv2 (optional)

room_map.py
    ├── Uses: ASMTCDR.py
    └── Dependencies: pathlib, json

launch_slam_rtmap.py
    ├── Launches: RealSense + RTAB-Map
    └── Dependencies: ROS 2, subprocess

demo_integrated.py
    ├── Uses: All above modules
    └── Dependencies: argparse, all module deps
```

## 📊 Data Flow

```
RealSense Camera
      ├─→ RGB/Depth → YOLO Detector → Detections
      ├─→ RGB/Depth → Semantic Pipeline → Labeled 3D Points
      ├─→ Depth/IMU → Dynamic Tracker → Moving Objects
      └─→ RGB/Depth → RTAB-Map → 3D Map + Pose

All data flows to → ASMTCDR Core → Room Map Logger → JSON Session
```

## 🎯 Quick Start Examples

### 1. Run Complete System (No Hardware)
```bash
python3 demo_integrated.py --integrated --duration 30
```

### 2. Run with RealSense Camera
```bash
# Terminal 1: Launch SLAM
python3 launch_slam_rtmap.py

# Terminal 2: YOLO Detection
YOLO_DISPLAY=1 python3 yolo_obj_slam.py

# Terminal 3: Semantic Mapping
SEMANTIC_DISPLAY=1 python3 sematic_map.py
```

### 3. Python API Usage
```python
# Import all components
from realsense.slam import (
    IntegratedMappingAndMonitoringSystem,
    RoomMapLogger,
    DynamicObjectTracker,
    SemanticMappingPipeline,
    YOLODetector
)

# Create system
system = IntegratedMappingAndMonitoringSystem()
logger = RoomMapLogger()

# Run
summary = system.start_autonomous_operation(
    duration_s=60,
    on_map_update=logger.handle_map_update,
    on_change_detected=logger.handle_change_event
)

# Save
logger.persist_session(summary)
```

## 🧪 Testing

All modules have corresponding tests in `tests/`:
- `test_asmtcdr.py` - Core system tests
- `test_dynamic_tracking.py` - Tracking tests
- `test_semantic_map.py` - Semantic pipeline tests
- `test_yolo_obj_slam.py` - YOLO tests
- `test_room_map.py` - Logger tests

Run tests:
```bash
pytest tests/ -v
pytest tests/test_asmtcdr.py -v  # Specific module
```

## 📚 See Also

- **Main README**: `/README.md` - Project overview
- **Quick Start**: `/QUICKSTART.md` - 5-minute setup
- **Project Structure**: `/PROJECT_STRUCTURE.md` - Complete file guide
- **Contributing**: `/CONTRIBUTING.md` - Development guidelines

---

**Package Version**: 0.1.0  
**Last Updated**: 2024-10-12
