from launch import LaunchDescription
import os
from pathlib import Path
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():

    pkg_path = get_package_share_directory('clean_bot')

    # Absolute path to model
    declare_model_path = DeclareLaunchArgument(
        name="model", 
        default_value=os.path.join(pkg_path, "description", "urdf", "robot.urdf.xacro"),
        description="Absolute path to robot urdf file"
    )

    # Parse xacro
    robot_description = ParameterValue(Command([
            "xacro ",
            LaunchConfiguration("model"),
        ]),
        value_type=str
    )

    # robot state publisher node 
    rsp = Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': robot_description,
                'use_sim_time': True
            }]
        )
    
    # --- Gazebo config ---
    
    # gazebo world file
    world = os.path.join(
        pkg_path, "worlds", "elevated.sdf"
    )

    # gazebo models
    models_path = os.path.join(
        pkg_path, "models"
    )

    # Set resources to be accessible by gz via env
    gz_resource_path = SetEnvironmentVariable(
        name = "GZ_SIM_RESOURCE_PATH",
        value=[
            os.environ.get("GZ_SIM_RESOURCE_PATH", ""),":",
            str(Path(pkg_path).parent.resolve()),":",
            models_path,
        ]
    )
    # Gazebo launch
    gazebo = IncludeLaunchDescription (
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory("ros_gz_sim"), 
                'launch'
            ), '/gz_sim.launch.py'
        ]),
        launch_arguments = {
            "gz_args": ["-v 4 -r ", world],
        }.items()
    )

    # Gazebo entity spawn
    spawn_entity = Node(
        package = "ros_gz_sim",
        executable = "create",
        output = "screen",
        arguments = [
            '-topic', 'robot_description',
            '-name', 'clean_bot'
        ]
    )

    # Gazebo param bridge
    param_bridge = Node(
        package = "ros_gz_bridge",
        executable = "parameter_bridge",
        arguments = [
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"
        ]
    )

    return LaunchDescription([
        declare_model_path,
        gz_resource_path,
        rsp,
        gazebo,
        spawn_entity,
        param_bridge
    ])