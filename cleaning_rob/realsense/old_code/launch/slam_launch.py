#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    # Launch arguments
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    use_realsense = LaunchConfiguration('use_realsense', default='true')
    
    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Whether to start RViz2'
        ),
        
        DeclareLaunchArgument(
            'use_realsense',
            default_value='true',
            description='Whether to start RealSense camera'
        ),
        
        # RealSense Camera Node
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense2_camera_node',
            parameters=[{
                'enable_pointcloud': True,
                'enable_sync': True,
                'align_depth.enable': True,
                'pointcloud_texture_stream': 'RS2_STREAM_COLOR'
            }],
            condition=IfCondition(use_realsense)
        ),
        
        # RTAB-Map Node
        Node(
            package='rtabmap_ros',
            executable='rtabmap',
            name='rtabmap',
            parameters=[{
                'frame_id': 'camera_link',
                'approx_sync': True,
                'rgb_topic': '/camera/color/image_raw',
                'depth_topic': '/camera/aligned_depth_to_color/image_raw',
                'camera_info_topic': '/camera/color/camera_info',
                'database_path': '~/.ros/rtabmap.db'
            }],
            remappings=[
                ('rgb/image', '/camera/color/image_raw'),
                ('depth/image', '/camera/aligned_depth_to_color/image_raw'),
                ('rgb/camera_info', '/camera/color/camera_info')
            ]
        ),
        
        # RViz2 Node
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', 'config/rviz_config.rviz'],
            condition=IfCondition(use_rviz)
        )
    ]) 