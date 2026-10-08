import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file for starting a complete RTAB-Map SLAM pipeline with a RealSense camera.
    This file handles the camera driver, IMU filtering, the main SLAM node,
    and the RTAB-Map visualizer.
    """

    # ===================================================================================
    # === 1. DECLARE LAUNCH ARGUMENTS ===================================================
    # ===================================================================================

    # A single switch to control simulation time for all nodes.
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo) clock if true'
    )

    # ===================================================================================
    # === 2. DEFINE CONFIGURATIONS & PARAMETERS =========================================
    # ===================================================================================

    # Get the value of the launch argument
    use_sim_time = LaunchConfiguration('use_sim_time')

    # Centralized topic remappings for consistency
    remappings = {
        'rgb_topic': '/camera/color/image_raw',
        'depth_topic': '/camera/aligned_depth_to_color/image_raw',
        'camera_info_topic': '/camera/color/camera_info',
        'imu_topic_raw': '/camera/imu',
        'imu_topic_filtered': '/imu/data'
    }

    # Parameters for the RealSense camera node
    realsense_params = {
        'align_depth.enable': 'true',
        'enable_gyro': 'true',
        'enable_accel': 'true',
        'unite_imu_method': '2', # 2 for 'copy'
        'base_frame_id': 'base_link',
        'depth_fps': '15.0',
        'color_fps': '15.0',
        'enable_sync': 'true',
    }

    # Parameters for the RTAB-Map SLAM node
    rtabmap_params = {
        'rgb_topic': remappings['rgb_topic'],
        'depth_topic': remappings['depth_topic'],
        'camera_info_topic': remappings['camera_info_topic'],
        'imu_topic': remappings['imu_topic_filtered'],
        'approx_sync': 'true',
        'approx_sync_max_interval': '0.05',
        'wait_imu_to_init': 'true',
        'rtabmap_args': '--delete_db_on_start',
        'qos': '2',
    }

    # ===================================================================================
    # === 3. DEFINE ACTIONS (NODES & LAUNCH INCLUDES) ===================================
    # ===================================================================================

    # --- RealSense Camera Driver ---
    realsense_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={**realsense_params, 'use_sim_time': use_sim_time}.items()
    )

    # --- IMU Filter ---
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{'use_mag': False, 'use_sim_time': use_sim_time}],
        remappings=[('/imu/data_raw', remappings['imu_topic_raw'])]
    )

    # --- RTAB-Map SLAM Node ---
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={**rtabmap_params, 'use_sim_time': use_sim_time}.items()
    )

    # --- RTAB-Map Visualizer ---
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/rgb/image', remappings['rgb_topic']),
            ('/depth/image', remappings['depth_topic']),
            ('/rgb/camera_info', remappings['camera_info_topic']),
        ]
    )

    # ===================================================================================
    # === 4. RETURN THE LAUNCH DESCRIPTION ==============================================
    # ===================================================================================

    return LaunchDescription([
        # Launch Arguments
        use_sim_time_arg,

        # Nodes & Other Launch Files
        realsense_camera_launch,
        imu_filter_node,
        rtabmap_launch,
        rtabmap_viz_node
    ])
