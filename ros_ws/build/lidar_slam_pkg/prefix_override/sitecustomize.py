import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mario/OpenCV_VC/ros_ws/install/lidar_slam_pkg'
