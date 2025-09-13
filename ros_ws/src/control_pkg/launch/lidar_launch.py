import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    sllidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('sllidar_ros2'),
                'launch',
                'sllidar_c1_launch.py'
            )
        )
    )

    lidar_avoid_node = Node(
            package='control_pkg',
            executable='lidar_avoid',
            name='lidar_avoid',
            output='screen',
            parameters=[
                {'scan_topic': '/scan'},
                {'stop_dist': 0.30},
                {'clear_dist': 0.60},
                {'deadband_deg': 15.0},
                {'avg_window': 7},
            ]
        )


    return LaunchDescription([
        sllidar_launch,
        lidar_avoid_node
    ])

