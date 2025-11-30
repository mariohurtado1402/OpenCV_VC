#!/usr/bin/env bash
set -e

WS="/home/mario/OpenCV_VC/ros_ws"

# Build once
cd "$WS"
colcon build

# Common source
COMMON_SOURCE="cd $WS && source install/setup.bash"

launch_lidar="$COMMON_SOURCE && ros2 launch control_pkg lidar_launch.py"
run_mux="$COMMON_SOURCE && ros2 run control_pkg cmd_mux"
run_vision="$COMMON_SOURCE && ros2 run yolo_pkg yolo_main"

gnome-terminal -- bash -lc "$launch_lidar; exec bash"
gnome-terminal -- bash -lc "$run_mux; exec bash"
gnome-terminal -- bash -lc "$run_vision; exec bash"
