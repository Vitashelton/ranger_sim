#!/usr/bin/env python3
import json
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .task_landmark_core import greedy_select, load_problem, route_metrics


class TaskLandmarkPlanner(Node):
    def __init__(self):
        super().__init__("task_landmark_planner")
        default_cfg = os.path.join(get_package_share_directory(
            "rangermini_dynamic_semantic"), "config", "task_landmarks.json")
        self.declare_parameter("config_path", default_cfg)
        self.declare_parameter("target_room", "906")
        self.declare_parameter("budget", 4)
        self.cfg, self.landmarks = load_problem(
            str(self.get_parameter("config_path").value))
        self.room = str(self.get_parameter("target_room").value)
        self.plan_pub = self.create_publisher(String, "/task_landmark_plan", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/task_landmark_markers", 10)
        self.create_subscription(String, "/task_goal", self.on_task, 10)
        self.create_timer(0.5, self.publish_plan)

    def on_task(self, msg):
        room = msg.data.strip()
        if room in self.cfg["tasks"]:
            self.room = room
        else:
            self.get_logger().warn(f"Unknown task room: {room}")

    def publish_plan(self):
        route = self.cfg["tasks"][self.room]
        selected = greedy_select(route, self.landmarks, self.cfg,
                                 int(self.get_parameter("budget").value))
        result = {
            "target_room": self.room,
            "method": "task_conditioned_d_optimal",
            "selected_ids": sorted(selected),
            "selected_names": [self.landmarks[i].name for i in sorted(selected)],
            **route_metrics(route, selected, self.landmarks, self.cfg),
        }
        self.plan_pub.publish(String(data=json.dumps(result, ensure_ascii=False)))
        self.marker_pub.publish(self.markers(selected))

    def markers(self, selected):
        arr = MarkerArray()
        for lid, lm in self.landmarks.items():
            m = Marker()
            m.header.frame_id = "odom"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "task_selected" if lid in selected else "task_candidates"
            m.id = lid
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = lm.x
            m.pose.position.y = lm.y
            m.pose.position.z = 1.25
            m.pose.orientation.w = 1.0
            m.scale.x, m.scale.y, m.scale.z = 0.24, 0.04, 0.24
            if lid in selected:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.05, 0.85, 0.20, 1.0
            else:
                m.color.r, m.color.g, m.color.b, m.color.a = 0.55, 0.55, 0.55, 0.28
            arr.markers.append(m)
            t = Marker()
            t.header = m.header
            t.ns, t.id, t.type, t.action = "task_landmark_labels", 100 + lid, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = lm.x, lm.y - 0.12, 1.55
            t.pose.orientation.w = 1.0
            t.scale.z = 0.18
            t.color.r, t.color.g, t.color.b, t.color.a = 0.05, 0.25, 0.05, 1.0
            t.text = f"Tag {lid} {lm.name}" + (" [ACTIVE]" if lid in selected else "")
            arr.markers.append(t)
        return arr


def main(args=None):
    rclpy.init(args=args)
    node = TaskLandmarkPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
