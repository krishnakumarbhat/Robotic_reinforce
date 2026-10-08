import os
from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    use_rviz_arg = DeclareLaunchArgument(
        name='use_rviz',
        default_value='true',
        description='Whether to launch RViz'
    )
    
    use_sim_time_arg = DeclareLaunchArgument(
        name='use_sim_time', 
        default_value='true',
        description='Use simulation time'
    )

    # Zero pose interactive button node
    zero_pose_node = Node(
        package="clean_bot_moveit_cpp",
        executable="zero_pose_interactive",
        name="zero_pose_interactive",
        output="screen",
        parameters=[{
            "use_sim_time": LaunchConfiguration('use_sim_time')
        }]
    )

    # RViz node (optional)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(
            get_package_share_directory("clean_bot_moveit_cpp"), 
            "rviz", 
            "zero_pose_interactive.rviz"
        )],
        parameters=[{
            "use_sim_time": LaunchConfiguration('use_sim_time')
        }],
        condition=LaunchConfiguration('use_rviz')
    )

    return LaunchDescription([
        use_rviz_arg,
        use_sim_time_arg,
        zero_pose_node,
        rviz_node
    ])