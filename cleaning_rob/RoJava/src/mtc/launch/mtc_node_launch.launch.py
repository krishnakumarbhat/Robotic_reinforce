import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    set_log_lv = DeclareLaunchArgument(
        name='log_level',
        default_value='info',
        choices=['debug', 'info']
    )

    set_sim_time_arg = DeclareLaunchArgument(
        name='is_sim',
        default_value="true"
    )

    use_sim_time = LaunchConfiguration('is_sim')


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
        .to_dict()

    initial_positions_file_path = os.path.join(get_package_share_directory("clean_bot_moveit_config"), "config", "initial_positions.yaml")

    # MTC Demo node
    pick_place_demo = Node(
        package="mtc",
        executable="mtc_node",
        # prefix=["gdbserver localhost:3000"],
        output="screen",
        parameters=[
            moveit_config,
            {'use_sim_time': use_sim_time},
            {'start_state': {'content': initial_positions_file_path}},
        ],
       ros_arguments=['--log-level', LaunchConfiguration('log_level')]
    )

    return LaunchDescription([set_sim_time_arg, set_log_lv, pick_place_demo])