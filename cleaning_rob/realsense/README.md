# realsense

This project provides a modular, SOLID-compliant framework for reading data from an Intel RealSense D435i camera and either publishing it to ROS 2 topics or saving it to a ROS 2 bag file.

RealSense Camera → realsense_node → ml_node → controller_node → Actuator
RealSense Camera → realsense_node → recorder_node → .bag files
.bag files → player_node → ml_node → controller_node → Actuator

## Structure

- `sense/reader.py` — CameraReader class for reading frames from the camera
- `store/bagstorage.py` — BagStorage class for saving frames to a ROS 2 bag file
- `ros_publisher.py` — RosPublisher class for publishing frames to a ROS topic
- `main.py` — Main entry point; run with `-bag` to save to bag, or without to publish to ROS


to run command 
ros2 launch realsense2_camera rs_launch.py enable_gyro:=true enable_accel:=true
