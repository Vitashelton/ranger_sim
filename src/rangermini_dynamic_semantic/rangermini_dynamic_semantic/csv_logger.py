#!/usr/bin/env python3
import csv
import json
import math
import os
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, String


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class CsvLogger(Node):
    def __init__(self):
        super().__init__("doorway_csv_logger")
        self.declare_parameter("output_dir", "/tmp/rangermini_doorway_logs")
        self.declare_parameter("file_prefix", "doorway_trial")
        self.declare_parameter("rate_hz", 20.0)

        self.output_dir = Path(self.get_parameter("output_dir").value).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.output_dir / f"{self.get_parameter('file_prefix').value}_{stamp}.csv"

        self.cmd_h = Twist()
        self.cmd_safe = Twist()
        self.cmd_executed = Twist()
        self.motion_mode = "UNKNOWN"
        self.motion_reason = ""
        self.min_distance = float("nan")
        self.intervention_score = 0.0
        self.risk_score = 0.0
        self.x = float("nan")
        self.y = float("nan")
        self.yaw = float("nan")

        self.file = open(self.path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "time","vx_h","vy_h","wz_h","vx_safe","vy_safe","wz_safe",
            "vx_executed","vy_executed","wz_executed","motion_mode","motion_reason",
            "min_distance","intervention_score","risk_score","x","y","yaw"
        ])
        self.t0 = self.get_clock().now()
        self.rows = 0

        self.create_subscription(Twist, "/cmd_vel_raw", self.on_raw, 10)
        self.create_subscription(Twist, "/cmd_vel_safe", self.on_safe, 10)
        self.create_subscription(Twist, "/cmd_vel_gazebo", self.on_executed, 10)
        self.create_subscription(String, "/motion_mode_debug", self.on_motion_mode, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Float32, "/min_distance", self.on_min_distance, 10)
        self.create_subscription(Float32, "/intervention_score", self.on_intervention, 10)
        self.create_subscription(Float32, "/risk_score", self.on_risk, 10)

        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.on_timer)
        self.get_logger().info(f"CSV logging to: {self.path}")

    def elapsed(self):
        return (self.get_clock().now() - self.t0).nanoseconds * 1e-9

    def on_raw(self, msg):
        self.cmd_h = msg

    def on_safe(self, msg):
        self.cmd_safe = msg

    def on_executed(self, msg):
        self.cmd_executed = msg

    def on_motion_mode(self, msg):
        try:
            payload = json.loads(msg.data)
            self.motion_mode = str(payload.get("mode", "UNKNOWN"))
            self.motion_reason = str(payload.get("reason", ""))
        except (ValueError, TypeError):
            self.motion_mode = "UNKNOWN"
            self.motion_reason = msg.data

    def on_min_distance(self, msg):
        self.min_distance = float(msg.data)

    def on_intervention(self, msg):
        self.intervention_score = float(msg.data)

    def on_risk(self, msg):
        self.risk_score = float(msg.data)

    def on_odom(self, msg):
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        self.yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def on_timer(self):
        self.writer.writerow([
            f"{self.elapsed():.4f}",
            f"{self.cmd_h.linear.x:.6f}",
            f"{self.cmd_h.linear.y:.6f}",
            f"{self.cmd_h.angular.z:.6f}",
            f"{self.cmd_safe.linear.x:.6f}",
            f"{self.cmd_safe.linear.y:.6f}",
            f"{self.cmd_safe.angular.z:.6f}",
            f"{self.cmd_executed.linear.x:.6f}",
            f"{self.cmd_executed.linear.y:.6f}",
            f"{self.cmd_executed.angular.z:.6f}",
            self.motion_mode,
            self.motion_reason,
            f"{self.min_distance:.6f}",
            f"{self.intervention_score:.6f}",
            f"{self.risk_score:.6f}",
            f"{self.x:.6f}",
            f"{self.y:.6f}",
            f"{self.yaw:.6f}",
        ])
        self.rows += 1
        if self.rows % 20 == 0:
            self.file.flush()
            os.fsync(self.file.fileno())

    def destroy_node(self):
        try:
            self.file.flush()
            self.file.close()
            self.get_logger().info(f"CSV saved: {self.path}")
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CsvLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
