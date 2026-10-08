import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from launch.conditions import IfCondition

def generate_launch_description():

    description_path = get_package_share_directory('clean_bot')

    set_sim_time = DeclareLaunchArgument(
        name = "is_sim",
        default_value = 'true', 
        description = "Setting the flag to determine whether the simulated clock is to be used instead of system clock"
    )

    # Parse xacro
    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                os.path.join(
                    description_path,
                    'description',
                    "urdf",
                    "robot.urdf.xacro",
                ),
                " is_sim:=", LaunchConfiguration("is_sim"),
            ]
        ),
        value_type=str,
    )

    joint_state_publisher_gui = Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
            condition= IfCondition(LaunchConfiguration("is_sim"))
        )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
    )

    rviz = Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', os.path.join(description_path, 'config', 'arm_view.yaml.rviz')]
                if os.path.join(description_path, 'config', 'arm_view.yaml.rviz') else []
        )

    # Launch nodes
    return LaunchDescription([
        set_sim_time,
        joint_state_publisher_gui,
        robot_state_publisher_node,
        rviz
    ])
