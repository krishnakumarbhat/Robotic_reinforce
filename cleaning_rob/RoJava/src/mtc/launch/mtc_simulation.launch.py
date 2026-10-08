#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder
from ament_index_python.packages import get_package_share_directory



def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation (Gazebo) clock if true",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "gui",
            default_value="true",
            description="Start RViz2 automatically.",
        )
    )

    # Initialize Arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    gui = LaunchConfiguration("gui")

    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("clean_bot"), "description", "urdf", "robot.urdf.xacro"]
            ),
            " ",
            "is_sim:=true",
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    # Robot state publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    # Gazebo with the elevated world containing the cylinder object
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("clean_bot"), "/launch/", "gz.launch.py"]
        ),
    )

    # ros2_control_node for simulation hardware interface
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
        output="screen",
        condition=UnlessCondition(use_sim_time)
    )

    # Controller spawners
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    # Delay controller spawners to ensure ros2_control_node is ready
    delayed_joint_state_broadcaster_spawner = TimerAction(
        period=3.0,
        actions=[joint_state_broadcaster_spawner],
    )

    delayed_arm_controller_spawner = TimerAction(
        period=3.5,
        actions=[arm_controller_spawner],
    )

    delayed_gripper_controller_spawner = TimerAction(
        period=4.0,
        actions=[gripper_controller_spawner],
    )

    # MoveIt move_group with ExecuteTaskSolutionCapability for MTC
    move_group_capabilities = {"capabilities": "move_group/ExecuteTaskSolutionCapability"}

    moveit_config = MoveItConfigsBuilder("clean_bot", package_name='clean_bot_moveit_config') \
        .robot_description(
            file_path=os.path.join(
                get_package_share_directory("clean_bot"),
                "description",
                "urdf",
                "robot.urdf.xacro"
            )
        ) \
        .robot_description_semantic(file_path="config/clean_bot.srdf") \
        .trajectory_execution(file_path="config/moveit_controllers.yaml") \
        .planning_pipelines(pipelines=["ompl"]) \
        .to_moveit_configs()

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), 
                    move_group_capabilities,
                    {"use_sim_time": use_sim_time},
                    {"publish_robot_description_semantic": True}],
        arguments=["--ros-args", "--log-level", "info"],
    )

    # RViz2 with MoveIt configuration
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare("clean_bot_moveit_config"), "rviz", "mtc.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(gui),
    )

    # MTC node
    mtc_node = Node(
        package="mtc",
        executable="mtc_node",
        name="mtc_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": use_sim_time}
        ],
    )

    # Delay MTC node to ensure MoveIt is ready
    delayed_mtc_node = TimerAction(
        period=10.0,
        actions=[mtc_node],
    )

    nodes = [
        # Core simulation components
        robot_state_publisher_node,
        gazebo,
        ros2_control_node,
        
        # Controllers (delayed)
        delayed_joint_state_broadcaster_spawner,
        delayed_arm_controller_spawner,
        delayed_gripper_controller_spawner,
        
        # MoveIt (delayed to allow controllers to start)
        TimerAction(
            period=5.0,
            actions=[move_group_node],
        ),
        
        # Visualization
        rviz_node,
        
        # MTC task (delayed to ensure everything is ready)
        delayed_mtc_node,
    ]

    return LaunchDescription(declared_arguments + nodes)