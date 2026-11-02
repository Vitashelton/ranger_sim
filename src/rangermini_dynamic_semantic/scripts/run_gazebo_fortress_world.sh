#!/usr/bin/env bash
set -e

# For Gazebo Fortress this is commonly:
#   ign gazebo -r worlds/narrow_doorway.sdf
#
# If your installation uses the newer command:
#   gz sim -r worlds/narrow_doorway.sdf

if command -v ign >/dev/null 2>&1; then
  ign gazebo -r worlds/narrow_doorway.sdf
elif command -v gz >/dev/null 2>&1; then
  gz sim -r worlds/narrow_doorway.sdf
else
  echo "Neither 'ign' nor 'gz' command found. Please install Gazebo Fortress."
  exit 1
fi
