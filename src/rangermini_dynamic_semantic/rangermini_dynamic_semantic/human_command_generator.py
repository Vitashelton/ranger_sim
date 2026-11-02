#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class HumanCommandGenerator(Node):
    def __init__(self):
        super().__init__("human_command_generator")
        self.declare_parameter("topic", "/cmd_vel_raw")
        self.declare_parameter("mode", "right_bias")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("forward_speed", 0.32)
        self.declare_parameter("lateral_bias", -0.22)
        self.declare_parameter("duration", 25.0)
        self.declare_parameter("goal_y", 2.75)
        self.y = -999.0
        self.t0 = self.get_clock().now()
        self.pub = self.create_publisher(Twist, self.get_parameter("topic").value, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.step)
        self.get_logger().info(f"Publishing human command, mode={self.get_parameter('mode').value}")

    def on_odom(self, msg):
        self.y = float(msg.pose.pose.position.y)

    def step(self):
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        msg = Twist()
        if t > float(self.get_parameter("duration").value) or self.y >= float(self.get_parameter("goal_y").value):
            self.pub.publish(msg); return
        mode = self.get_parameter("mode").value
        f = float(self.get_parameter("forward_speed").value)
        b = float(self.get_parameter("lateral_bias").value)
        if mode == "center":
            msg.linear.x = f
        elif mode == "left_bias":
            msg.linear.x = f; msg.linear.y = abs(b)
        elif mode == "right_bias":
            msg.linear.x = f; msg.linear.y = -abs(b)
        elif mode == "sine":
            msg.linear.x = f; msg.linear.y = b * math.sin(1.6 * t); msg.angular.z = 0.12 * math.sin(0.8*t)
        elif mode == "stop":
            pass
        else:
            msg.linear.x = f; msg.linear.y = b
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    n = HumanCommandGenerator()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node(); rclpy.shutdown()

if __name__ == "__main__":
    main()
