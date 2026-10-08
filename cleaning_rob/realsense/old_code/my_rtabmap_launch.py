# my_rtabmap_launch.py

import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():

    # 1. Node for the IMU filter (our custom part)
    # This takes the raw IMU data and adds the orientation
    imu_filter_node = Node(
        package='imu_filter_madgwick',
        executable='imu_filter_madgwick_node',
        name='imu_filter',
        output='screen',
        parameters=[{
            'use_mag': False,
            'publish_tf': False,
            'use_sim_time': True,
        }],
        remappings=[
            ('/imu/data_raw', '/camera/camera/imu'),
            ('/imu/data', '/imu/data_filtered') # Publish the result on a new topic
        ]
    )

    # 2. Include the official RTAB-Map launch file (the part that works)
    # We pass all the necessary arguments to it
    rtabmap_launch_include = IncludeLaunchDescription(
        # Find the launch file from the installed 'rtabmap_launch' package
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rtabmap_launch'),
                'launch',
                'rtabmap.launch.py'
            )
        ),
        # Pass arguments to the included launch file
        launch_arguments={
            'rgb_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'imu_topic': '/imu/data_filtered',  # IMPORTANT: Point it to our filtered topic
            'approx_sync': 'true',
            'wait_imu_to_init': 'true',
            'qos': '2',
            'use_sim_time': 'true',
            'rtabmap_args': '--delete_db_on_start'
        }.items()
    )
    
    # 3. Return a launch description that starts both our filter and the official RTAB-Map
    return LaunchDescription([
        imu_filter_node,
        rtabmap_launch_include
    ])