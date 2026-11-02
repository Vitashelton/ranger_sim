#!/usr/bin/env bash
set -euo pipefail

WS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source /opt/ros/humble/setup.bash
source "$WS/install/setup.bash"

# The interactive Ranger shell binds CycloneDDS to the hardware Wi-Fi only.
# Keep Gazebo and its ROS bridge local and isolated from the real robot.
unset CYCLONEDDS_URI
export ROS_DOMAIN_ID=99
# Reuse the local DDS backend of the established Office RPG entry point.
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/ranger_lab_ros_logs}"

door_state="${1:-half_open}"
shift || true
case "$door_state" in
  open|half_open|angled|closed) ;;
  *) echo "Usage: $0 [open|half_open|angled|closed] [launch arguments ...]" >&2; exit 2 ;;
esac
world_file="$WS/install/rangermini_dynamic_semantic/share/rangermini_dynamic_semantic/worlds/ranger_lab_corridor_firedoor_${door_state}.sdf"

exec ros2 launch rangermini_dynamic_semantic ranger_mppi_corridor.launch.py \
  world_file:="$world_file" "$@"
