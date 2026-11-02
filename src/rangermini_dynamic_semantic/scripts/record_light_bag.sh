#!/usr/bin/env bash
set -e
OUT_DIR=${1:-$HOME/bags/rangermini_doorway/doorway_trial_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$(dirname "$OUT_DIR")"

ros2 bag record \
  /cmd_vel_raw \
  /cmd_vel_safe \
  /odom \
  /tf \
  /min_distance \
  /intervention_score \
  /risk_score \
  /debug/candidate_paths \
  /debug/doorway_markers \
  /debug/executed_path \
  -o "$OUT_DIR"
