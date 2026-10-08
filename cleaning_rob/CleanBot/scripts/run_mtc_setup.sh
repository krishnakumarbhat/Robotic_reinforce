#!/bin/bash

# MTC Setup Sequence Script
# This script launches the CleanBot MTC task demonstration
# Usage: ./run_mtc_setup.sh

echo "=== CleanBot MTC Setup Sequence ==="
echo "Building workspace..."

# Build the workspace
# colcon build --symlink-install --mixin release
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Debug --packages-select clean_bot clean_bot_controller clean_bot_moveit_config clean_bot_moveit_cpp mtc # alias for colcon builld and source install dir
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo "Build completed successfully"
echo "Starting Terminal 1: headless_bringup.launch.py"

# Create logs directory
mkdir -p terminal_logs

# Kill any existing ROS processes
pkill -f "ros2\|rviz\|move_group\|mtc_node" 2>/dev/null
sleep 2

# Launch Terminal 1 in background with logging
nohup bash -c "source ./install/setup.bash && ros2 launch clean_bot headless_bringup.launch.py log_level:=info" > terminal_logs/terminal1.log 2>&1 &
TERMINAL1_PID=$!

# Wait a moment to ensure clean startup
sleep 3

echo "Terminal 1 started with PID: $TERMINAL1_PID"
echo "Waiting 10 seconds for move_group to fully load..."

# Wait for move_group to initialize
sleep 10

echo "Starting Terminal 2: mtc_node_launch.launch.py"

# Check if Terminal 1 is running properly before starting Terminal 2
if ! ps -p $TERMINAL1_PID > /dev/null; then
    echo "ERROR: Terminal 1 failed to start properly"
    exit 1
fi

# Launch Terminal 2 (MTC node) in background with logging
nohup bash -c "source ./install/setup.bash && ros2 launch mtc mtc_node_launch.launch.py log_level:=info" > terminal_logs/terminal2.log 2>&1 &
TERMINAL2_PID=$!

echo "Terminal 2 started with PID: $TERMINAL2_PID"
echo ""
echo "=== MTC Setup Complete ==="
echo "Both terminals are now running:"
echo "  - Terminal 1: Robot state, controllers, move_group, RViz"
echo "  - Terminal 2: MTC node executing grasp task"
echo ""
echo "To monitor outputs:"
echo "  tail -f terminal_logs/terminal1.log  # Terminal 1 output"
echo "  tail -f terminal_logs/terminal2.log  # Terminal 2 output"
echo ""
echo "Process PIDs:"
echo "  Terminal 1: $TERMINAL1_PID"
echo "  Terminal 2: $TERMINAL2_PID"
echo ""
echo "Press any key to stop all processes..."

# Wait for keypress
read -n 1 -s

echo ""
echo "Stopping all processes..."

# Kill the terminal processes
if ps -p $TERMINAL1_PID > /dev/null; then
    kill $TERMINAL1_PID
    echo "Stopped Terminal 1 (PID: $TERMINAL1_PID)"
fi

if ps -p $TERMINAL2_PID > /dev/null; then
    kill $TERMINAL2_PID
    echo "Stopped Terminal 2 (PID: $TERMINAL2_PID)"
fi

# Kill any remaining ROS processes
pkill -9 -f "ros2|rviz|move_group|mtc_node" 2>/dev/null
sleep 1
pkill -f "ros2|rviz|robot_state_publisher|joint_state_publisher|move_group|launch|gazebo"
sleep 1
killall -9 ros2 rviz2 python3 fastdds cyclonedds rviz robot_state_publisher move_group 2>/dev/null


echo "All processes stopped. Exiting."