from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    # lidar_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         str(Path(get_package_share_directory('sllidar_ros2')) / 'launch' / 'sllidar_c1_launch.py')
    #     )
    # )

    # cmd_mux = Node(
    #     package='control_pkg',
    #     executable='cmd_mux',
    #     name='cmd_mux'
    # )

    # lidar_avoid = Node(
    #     package='control_pkg',
    #     executable='lidar_avoid',
    #     name='lidar_avoid'
    # )

    # vision = Node(
    #     package='yolo_pkg',
    #     executable='yolo_main',
    #     name='yolo_main'
    # )

    serial_bridge = Node(
        package='control_pkg',
        executable='serial_bridge',
        name='serial_bridge'
    )

    metrics_bridge = Node(
        package='control_pkg',
        executable='metrics_bridge',
        name='metrics_bridge'
    )

    return LaunchDescription([
        # lidar_launch,
        # cmd_mux,
        # lidar_avoid,
        # vision,
        serial_bridge,
        metrics_bridge,
    ])
