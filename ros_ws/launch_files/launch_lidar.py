import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    slidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                os.getenv('HOME'),
                'vc_ws',
                'ros_ws',
                'src',
                'sllidar_ros2',
                'launch',
                'sllidar_c1_launch.py'
            )
        )
    )

    closest_object_node = Node(
        package='lidar_slam_pkg',
        executable='closest_object.py',
        name='closest_object_node',
        output='screen'
    )

    return LaunchDescription([
        slidar_launch,
        closest_object_node
    ])

if __name__ == '__main__':
    ld = generate_launch_description()
    ls = launch.LaunchService()
    ls.include_launch_description(ld)
    sys.exit(ls.run())