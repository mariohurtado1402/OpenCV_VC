import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    base_frame = LaunchConfiguration('base_frame')
    laser_frame = LaunchConfiguration('laser_frame')
    scan_topic = LaunchConfiguration('scan_topic')

    base_frame_arg = DeclareLaunchArgument(
        'base_frame',
        default_value='base_link',
        description='Base frame of the robot'
    )
    laser_frame_arg = DeclareLaunchArgument(
        'laser_frame',
        default_value='laser',
        description='Frame id published in /scan'
    )
    scan_topic_arg = DeclareLaunchArgument(
        'scan_topic',
        default_value='/scan',
        description='LaserScan topic'
    )

    sllidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('sllidar_ros2'),
                'launch',
                'sllidar_c1_launch.py'
            )
        ),
        launch_arguments={
            'frame_id': laser_frame,
            'scan_mode': 'Standard',
            'angle_compensate': 'true'
        }.items(),
    )

    rf2o_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('rf2o_laser_odometry'),
                'launch',
                'rf2o_laser_odometry.launch.py'
            )
        )
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='laser_to_base',
        arguments=[
            '--x', '0.0',
            '--y', '0.0',
            '--z', '0.0',
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', '0.0',
            '--frame-id', base_frame,
            '--child-frame-id', laser_frame,
        ],
    )

    nav2_params = os.path.join(
        get_package_share_directory('control_pkg'),
        'config',
        'nav2_params.yaml'
    )

    slam_params = os.path.join(
        get_package_share_directory('control_pkg'),
        'config',
        'slam_params.yaml'
    )

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nav2_bringup'),
                'launch',
                'bringup_launch.py'
            )
        ),
        launch_arguments={
            'use_sim_time': 'false',
            'autostart': 'true',
            'params_file': nav2_params,
            'slam': 'True',
            'slam_params_file': slam_params,
        }.items(),
    )

    map_saver = Node(
        package='control_pkg',
        executable='map_to_png',
        name='map_to_png',
        parameters=[{
            'map_topic': '/map',
            'output_dir': str(Path.home() / 'maps')
        }]
    )

    return LaunchDescription([
        base_frame_arg,
        laser_frame_arg,
        scan_topic_arg,
        sllidar_launch,
        rf2o_launch,
        static_tf,
        nav2_bringup,
        map_saver
    ])
