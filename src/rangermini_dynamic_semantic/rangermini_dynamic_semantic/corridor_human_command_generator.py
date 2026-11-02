#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

class CorridorHumanCommandGenerator(Node):
    def __init__(self):
        super().__init__("corridor_human_command_generator")
        self.declare_parameter("topic", "/cmd_vel_raw")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("mode", "unsafe_centerline")
        self.declare_parameter("forward_speed", 0.55)
        self.declare_parameter("vertical_bias", 0.0)
        self.declare_parameter("goal_x", 13.35)
        self.declare_parameter("auto_stop_margin", 0.20)
        self.declare_parameter("enabled", False)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.enabled = bool(self.get_parameter("enabled").value)
        self.pub = self.create_publisher(Twist, self.get_parameter("topic").value, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(String, "/task_goal", self.on_task_goal, 10)
        self.create_subscription(String, "/task_control", self.on_task_control, 10)
        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.step)
        self.get_logger().info(f"Publishing corridor human command, mode={self.get_parameter('mode').value}")

    def on_odom(self, msg):
        self.x = float(msg.pose.pose.position.x)
        self.y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def on_task_goal(self, msg):
        goals = {"lobby": 1.2, "902": 3.0, "904": 8.0, "906": 13.35,
                 "908": 18.55, "corridor_junction": 11.0}
        if msg.data.strip() in goals:
            self.goal_x = goals[msg.data.strip()]

    def on_task_control(self, msg):
        command = msg.data.strip().upper()
        if command == "START":
            self.enabled = True
            self.get_logger().info("Task command generation enabled")
        elif command == "STOP":
            self.enabled = False
            self.pub.publish(Twist())
            self.get_logger().warn("Task stopped")

    def step(self):
        msg = Twist()
        if not self.enabled:
            self.pub.publish(msg)
            return
        goal_x = self.goal_x
        margin = float(self.get_parameter("auto_stop_margin").value)
        if abs(self.x - goal_x) <= margin:
            self.pub.publish(msg)
            return
        mode = self.get_parameter("mode").value
        # The raw command is expressed in the chassis frame. After a spin-mode
        # turnaround, moving toward a lower world-X goal is again +body-X.
        world_dx = goal_x - self.x
        local_dx = math.cos(self.yaw) * world_dx
        fwd = math.copysign(
            float(self.get_parameter("forward_speed").value), local_dx)
        bias = float(self.get_parameter("vertical_bias").value)

        if mode == "unsafe_centerline":
            # Raw human baseline: goes straight and would collide with S2.
            msg.linear.x = fwd
            msg.linear.y = 0.0
        elif mode == "wrong_top_drift":
            msg.linear.x = fwd
            msg.linear.y = 0.18
        elif mode == "wrong_bottom_drift":
            msg.linear.x = fwd
            msg.linear.y = -0.18
        elif mode == "manual_bias":
            msg.linear.x = fwd
            msg.linear.y = bias
        else:
            msg.linear.x = fwd
            msg.linear.y = 0.0
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CorridorHumanCommandGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
