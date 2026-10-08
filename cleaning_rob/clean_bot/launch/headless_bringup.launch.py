import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument

from moveit_configs_utils import MoveItConfigsBuilder


os.environ["SPDLOG_LEVEL"] = "debug"
os.environ["SPDLOG_PATTERN"] = "[%H:%M:%S.%e] [%^%l%$] %v"

def generate_launch_description():

    set_log_lv = DeclareLaunchArgument(
        name='log_level',
        default_value='info',
        choices=['debug', 'info']
    )

    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                os.path.join(
                    get_package_share_directory("clean_bot"),
                    'description',
                    "urdf",
                    "robot.urdf.xacro",
                ),
                " is_mock:=true",
                " is_sim:=false"
            ]
        ),
        value_type=str,
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": robot_description}],
    )

    controller_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description}, 
            os.path.join(
                get_package_share_directory("clean_bot_controller"),
                "config",
                "clean_bot_controller.yaml",
            ),
        ],

        output="both",
    ) 
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller", "--controller-manager", "/controller_manager"],
    )

    gripper_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller", "--controller-manager", "/controller_manager"],
    )
    
    # Load  ExecuteTaskSolutionCapability to execute found solutions in simulation
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
        .planning_scene_monitor(publish_planning_scene=True) \
        .to_moveit_configs()

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), 
                    move_group_capabilities,
                    {"publish_robot_description_semantic": True}
                    ],
        arguments=["--ros-args", "--log-level", LaunchConfiguration('log_level')],
    )

    # RViz
    rviz_config = os.path.join(
        get_package_share_directory("clean_bot_moveit_config"),
            "rviz",
            "mtc.rviz",
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            moveit_config.planning_pipelines,
        ],
    )

    return LaunchDescription(
        [
            set_log_lv,
            robot_state_publisher_node,
            controller_node,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
            move_group_node,
            rviz_node,
        ]
    )