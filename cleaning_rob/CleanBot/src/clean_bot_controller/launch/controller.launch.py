import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetLaunchConfiguration, LogInfo
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory



os.environ["SPDLOG_LEVEL"] = "debug"
os.environ["SPDLOG_PATTERN"] = "[%H:%M:%S.%e] [%^%l%$] %v"

def generate_launch_description():

    set_sim_time = DeclareLaunchArgument(
        "is_sim",
        default_value="True"
    )

    set_if_calib = DeclareLaunchArgument(
        "is_calib",
        default_value="False"
    )

    set_debug_lv = DeclareLaunchArgument(
        "is_deb",
        default_value="False"
    )

    def check_valid_args(context, *args, **kwargs):
        sim_time_arg = LaunchConfiguration('is_sim').perform(context)
        is_clib_arg = LaunchConfiguration('is_calib').perform(context)
        is_deb_arg = LaunchConfiguration('is_deb').perform(context)

        if sim_time_arg=="True" and (is_clib_arg == "True" or is_deb_arg == "True"):
            raise RuntimeError('Invalid config: Calibration Mode and/or Debug Server Mode cannot be run in simulation')
        
    run_arg_validity_checker = OpaqueFunction(function=check_valid_args)

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
                " is_sim:=", LaunchConfiguration("is_sim"),
                " is_calib:=", LaunchConfiguration("is_calib"),
            ]
        ),
        value_type=str,
    )
    
    # unified controller config for both sim and hardware
    controller_config_path = os.path.join(
        get_package_share_directory("clean_bot_controller"),
        "config",
        "clean_bot_controller.yaml",
    )


    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description}],
        condition=UnlessCondition(LaunchConfiguration("is_sim")),
    )

    

    controller_node_debug = Node(
        package="controller_manager",
        executable="ros2_control_node",
        prefix=["gdbserver localhost:3000"],
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": LaunchConfiguration("is_sim")
            }, 
            controller_config_path
        ],
        output="screen",
        condition=IfCondition(LaunchConfiguration("is_deb")),
    )

    controller_node_normal = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": LaunchConfiguration("is_sim")
            }, 
            controller_config_path
        ],
        output="screen",
        condition=IfCondition(
            PythonExpression([
                "not (", LaunchConfiguration("is_deb"), ") and not (", LaunchConfiguration("is_sim"), ")"
            ]),
        ),
    ) # should be: true -> false, false

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

    return LaunchDescription(
        [
            set_if_calib,
            set_debug_lv,
            set_sim_time,
            run_arg_validity_checker,
            robot_state_publisher_node,
            controller_node_debug,
            controller_node_normal,
            joint_state_broadcaster_spawner,
            arm_controller_spawner,
            gripper_controller_spawner,
        ]
    )
