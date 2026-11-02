#!/usr/bin/env bash
set -e
source install/setup.bash
ros2 launch rangermini_dynamic_semantic doorway_closed_loop.launch.py use_gazebo:=false human_mode:=right_bias
