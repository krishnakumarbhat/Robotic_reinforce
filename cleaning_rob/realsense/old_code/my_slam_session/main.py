import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():

    # ===================================================================================
    # === Terminal 1: Camera (The Fix) ==================================================
    # ===================================================================================
    # This command correctly configures the camera driver to create base_link itself,
    # solving all TF issues. We also enable hardware sync for better timestamps.
    realsense_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('realsense2_camera'), 'launch', 'rs_launch.py')
        ),
        launch_arguments={
            'align_depth.enable': 'true',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2',
            'base_frame_id': 'base_link',
            'use_sim_time': 'false',
            'depth_fps': '15.0',
            'color_fps': '15.0',
            'enable_sync': 'true', # Crucial for better timestamp alignment
        }.items()
    )

    # ===================================================================================
    # === Terminal 2: IMU Filter ========================================================
    # ===================================================================================
    # This node adds the necessary orientation data to the IMU stream.
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{'use_mag': False, 'use_sim_time': False}],
        remappings=[('/imu/data_raw', '/camera/camera/imu')]
    )

    # ===================================================================================
    # === Terminal 3: RTAB-Map ==========================================================
    # ===================================================================================
    # This is the tuned launch command for the main SLAM node.
    rtabmap_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rtabmap_launch'), 'launch', 'rtabmap.launch.py')
        ),
        launch_arguments={
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'imu_topic': '/imu/data',
            'approx_sync': 'true',
            'approx_sync_max_interval': '0.05',
            'wait_imu_to_init': 'true',
            'rtabmap_args': '--delete_db_on_start',
            'qos': '2', # Best effort for live data
            'use_sim_time': 'false'
        }.items()
    )
    
    # ===================================================================================
    # === Terminal 4: Visualizer ========================================================
    # ===================================================================================
    rtabmap_viz_node = Node(
        package='rtabmap_viz',
        executable='rtabmap_viz',
        name='rtabmap_viz',
        output='screen',
        parameters=[{'use_sim_time': False}],
        remappings=[
            ('/rgb/image', '/camera/camera/color/image_raw'),
            ('/depth/image', '/camera/camera/aligned_depth_to_color/image_raw'),
            ('/rgb/camera_info', '/camera/camera/color/camera_info'),
        ]
    )
    
    # ===================================================================================
    # === Terminal 5: Map Saver Script ==================================================
    # ===================================================================================
    # This is your custom Python script to save the map on key press.
    # Note: We will make this an executable in Step 4.
    map_saver_node = Node(
        package='my_slam_project',
        executable='map_saver',
        name='map_saver_node',
        output='screen' # Important to see the "Press s to save" prompt
    )

    return LaunchDescription([
        realsense_camera_launch,
        imu_filter_node,
        rtabmap_launch,
        rtabmap_viz_node,
        map_saver_node
    ])
