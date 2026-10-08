import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder
from launch.conditions import IfCondition, UnlessCondition
from launch.actions import DeclareLaunchArgument

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    
    set_mtc_mode = DeclareLaunchArgument(
        name='use_mtc',
        default_value='True',
    )

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
        .planning_scene_monitor( \
                publish_robot_description=False, \
                publish_robot_description_semantic=True, \
                publish_planning_scene=True, \
        ) \
        .to_moveit_configs()

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
       # prefix=["gdbserver localhost:3000"],
        output="screen",
        parameters=[moveit_config.to_dict(), 
                    move_group_capabilities,
                    {"use_sim_time": True},
                    {"publish_robot_description_semantic": True}],
        arguments=["--ros-args", "--log-level", "info"],
    )

    mtc_rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(get_package_share_directory("clean_bot_moveit_config"), "rviz", "mtc.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
        condition=IfCondition(LaunchConfiguration('use_mtc'))
    )

    moveit_rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", os.path.join(get_package_share_directory("clean_bot_moveit_config"), "rviz", "moveit.rviz")],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            {"use_sim_time": True},
        ],
        condition=UnlessCondition(LaunchConfiguration('use_mtc'))
    )

    return LaunchDescription([
        set_mtc_mode,
        move_group_node,
        mtc_rviz_node,
        moveit_rviz_node
    ])