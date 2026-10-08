"""SLAM and mapping components.

This package provides all SLAM-related functionality including:
- Core ASMTCDR integrated system
- Dynamic object tracking
- Semantic mapping pipeline
- YOLO object detection
- Room mapping logger
- SLAM launch utilities
"""

# Core ASMTCDR components
from realsense.slam.ASMTCDR import (
    IntegratedMappingAndMonitoringSystem,
    MapUpdate,
    MapperRTAB,
    RealSenseDriver,
    RobotController,
)

# Dynamic object tracking
from realsense.slam.dynamic_obj_track import (
    DynamicObjectTracker,
    MockStaticMapper,
)

# Room mapping logger
from realsense.slam.room_map import (
    RoomMapLogger,
    run_autonomous_room_mapping,
)

# Semantic mapping pipeline
from realsense.slam.sematic_map import (
    Detection2D,
    RTABMapAccumulator,
    SemanticMapConfig,
    SemanticMappingPipeline,
    SemanticMappingNode,
)

# YOLO object detection
from realsense.slam.yolo_obj_slam import (
    DetectorConfig,
    YOLODetector,
    YoloObjSlamNode,
)

__all__ = [
    # Core ASMTCDR
    "IntegratedMappingAndMonitoringSystem",
    "MapUpdate",
    "MapperRTAB",
    "RealSenseDriver",
    "RobotController",
    # Dynamic tracking
    "DynamicObjectTracker",
    "MockStaticMapper",
    # Room mapping
    "RoomMapLogger",
    "run_autonomous_room_mapping",
    # Semantic mapping
    "Detection2D",
    "RTABMapAccumulator",
    "SemanticMapConfig",
    "SemanticMappingPipeline",
    "SemanticMappingNode",
    # YOLO detection
    "DetectorConfig",
    "YOLODetector",
    "YoloObjSlamNode",
]
