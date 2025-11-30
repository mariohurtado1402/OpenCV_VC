#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="/home/mario/OpenCV_VC/ros_ws"

cd "$WORKSPACE"
colcon build

# Avoid unbound variables from upstream setup scripts under -u
set +u
export COLCON_TRACE=${COLCON_TRACE:-0}
export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-0}
source "$WORKSPACE/install/setup.bash"
set -u

ros2 launch control_pkg full_launch.py
