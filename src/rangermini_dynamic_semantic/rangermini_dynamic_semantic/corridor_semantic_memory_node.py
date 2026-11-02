#!/usr/bin/env python3
"""
Semantic memory + target selector for the corridor benchmark.

This node mimics the structure of a YOLO-based semantic navigation stack:
  detector observations -> persistent semantic memory -> target door-front pose.

It subscribes:
  /semantic_detections_json   std_msgs/String

It publishes:
  /semantic_target_pose       geometry_msgs/PoseStamped
  /semantic_memory_debug      std_msgs/String

The selected target defaults to:
  room = 906, material = glass, label = door_front
"""
import json
import math

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


class CorridorSemanticMemoryNode(Node):
    def __init__(self):
        super().__init__("corridor_semantic_memory_node")
        self.declare_parameter("input_topic", "/semantic_detections_json")
        self.declare_parameter("target_pose_topic", "/semantic_target_pose")
        self.declare_parameter("debug_topic", "/semantic_memory_debug")
        self.declare_parameter("target_room", "906")
        self.declare_parameter("target_material", "glass")
        self.declare_parameter("target_label", "door_front")
        self.declare_parameter("publish_rate_hz", 5.0)

        self.memory = {}
        self.target = None

        self.create_subscription(String, self.get_parameter("input_topic").value, self.on_detections, 10)
        self.create_subscription(String, "/task_goal", self.on_task_goal, 10)
        self.target_pub = self.create_publisher(PoseStamped, self.get_parameter("target_pose_topic").value, 10)
        self.debug_pub = self.create_publisher(String, self.get_parameter("debug_topic").value, 10)
        self.timer = self.create_timer(1.0 / float(self.get_parameter("publish_rate_hz").value), self.on_timer)
        self.get_logger().info("Semantic memory node started. Waiting for detector observations.")

    def on_detections(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Bad semantic JSON: {exc}")
            return

        for det in payload.get("detections", []):
            key = f"{det.get('label','unknown')}:{det.get('room','unknown')}:{det.get('material','unknown')}"
            old = self.memory.get(key)
            if old is None:
                self.memory[key] = dict(det)
                self.memory[key]["times_seen"] = 1
            else:
                a = 0.35
                old["x"] = (1.0 - a) * float(old["x"]) + a * float(det.get("x", old["x"]))
                old["y"] = (1.0 - a) * float(old["y"]) + a * float(det.get("y", old["y"]))
                old["confidence"] = max(float(old.get("confidence", 0.0)), float(det.get("confidence", 0.0)))
                old["times_seen"] = int(old.get("times_seen", 1)) + 1

        self.select_target()

    def on_task_goal(self, msg):
        room = msg.data.strip()
        if room not in {"902", "904", "906", "908"}:
            return
        self.set_parameters([
            Parameter("target_room", value=room),
            Parameter("target_material", value="wood" if room in {"902", "904"} else "glass")
        ])
        self.select_target()

    def select_target(self):
        target_room = str(self.get_parameter("target_room").value)
        target_material = str(self.get_parameter("target_material").value)
        target_label = str(self.get_parameter("target_label").value)

        best = None
        best_score = -1.0
        for obj in self.memory.values():
            if str(obj.get("room", "")) != target_room:
                continue
            if str(obj.get("material", "")) != target_material:
                continue
            if str(obj.get("label", "")) != target_label:
                continue
            score = float(obj.get("confidence", 0.0)) + 0.05 * int(obj.get("times_seen", 1))
            if score > best_score:
                best = obj
                best_score = score

        self.target = best

    def on_timer(self):
        dbg = {
            "memory_size": len(self.memory),
            "target_room": self.get_parameter("target_room").value,
            "target_material": self.get_parameter("target_material").value,
            "target": self.target,
        }
        self.debug_pub.publish(String(data=json.dumps(dbg, ensure_ascii=False)))

        if self.target is None:
            return

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "odom"
        pose.pose.position.x = float(self.target["x"])
        pose.pose.position.y = float(self.target["y"])
        pose.pose.orientation.w = 1.0
        self.target_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = CorridorSemanticMemoryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
