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

    closest_object_node = Node(
        package='control_pkg',
        executable='closest_object',
        name='closest_object_node',
        output='screen'
    )


    return LaunchDescription([
        sllidar_launch,
        closest_object_node
    ])

