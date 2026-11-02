#!/usr/bin/env bash
# Read-only recording: this script never publishes motion commands.
set -euo pipefail
if ! command -v ros2 >/dev/null 2>&1; then
  echo '请先 source ROS 2 和当前实机工作区的 setup.bash。' >&2
  exit 1
fi
record_dir="${1:-$PWD/ranger_motion_response_$(date +%Y%m%d_%H%M%S)}"
if [[ -e "$record_dir" || -e "${record_dir}_topics.txt" ]]; then
  echo "输出已存在，换一个目录：$record_dir" >&2
  exit 1
fi
mkdir -p "$(dirname "$record_dir")"
ros2 topic list -t > "${record_dir}_topics.txt"
for feedback_topic in /odom /motion_state /actuator_state; do
  if ! ros2 topic info "$feedback_topic" >/dev/null 2>&1; then
    echo "缺少反馈话题 $feedback_topic；先确认驱动和 ROS_DOMAIN_ID，再采数据。" >&2
    exit 1
  fi
done
ros2 interface show ranger_msgs/msg/ActuatorStateArray > "${record_dir}_actuator_interface.txt"
ros2 interface show ranger_msgs/msg/ActuatorState >> "${record_dir}_actuator_interface.txt"
echo '仅录制已有话题；用现有遥控器操作，Ctrl-C 停止。'
echo '先静止约 5 秒；低速做：直行→斜移→横移；直行→原地旋转→前进；各重复 3 次。'
echo '动作间保留稳定段；实际模式切换由现有驱动处理，不由本脚本发指令。'
echo '另拍视频标清前左/前右/后左/后右；录包不会验证舵角通道、单位或零位。'
exec ros2 bag record -o "$record_dir" \
  /cmd_vel /cmd_vel_safe /odom /motion_state /actuator_state /joint_states /system_state /tf /tf_static
