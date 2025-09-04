import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/user/vc_ws/ros_ws/install/lidar_slam_pkg'
