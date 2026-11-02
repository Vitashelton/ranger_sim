#!/usr/bin/env python3
"""Convert Gazebo ground-truth model pose into the benchmark /odom contract."""
import math

import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


def yaw_from_quat(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class GazeboOdomAdapter(Node):
    def __init__(self):
        super().__init__("gazebo_odom_adapter")
        self.declare_parameter("input_topic", "/model/rangermini_2_0/pose")
        self.declare_parameter("output_topic", "/odom")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        self.pub = self.create_publisher(
            Odometry, str(self.get_parameter("output_topic").value), 20)
        self.tf = TransformBroadcaster(self)
        self.received_first_pose = False
        self.previous = None
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("input_topic").value),
            self.on_pose,
            20)
        self.create_timer(3.0, self.report_waiting)

    def report_waiting(self):
        if not self.received_first_pose:
            self.get_logger().warn(
                "Waiting for Gazebo ground-truth pose on "
                f"{self.get_parameter('input_topic').value}")

    def on_pose(self, msg):
        out = Odometry()
        out.header.stamp = msg.header.stamp
        if out.header.stamp.sec == 0 and out.header.stamp.nanosec == 0:
            out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = str(self.get_parameter("odom_frame").value)
        out.child_frame_id = str(self.get_parameter("base_frame").value)
        out.pose.pose = msg.pose

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        yaw = yaw_from_quat(msg.pose.orientation)
        stamp = float(out.header.stamp.sec) + 1.0e-9 * out.header.stamp.nanosec
        if self.previous is not None:
            px, py, pyaw, previous_stamp = self.previous
            dt = stamp - previous_stamp
            if 1.0e-4 < dt < 1.0:
                world_vx = (x - px) / dt
                world_vy = (y - py) / dt
                out.twist.twist.linear.x = (
                    math.cos(yaw) * world_vx + math.sin(yaw) * world_vy)
                out.twist.twist.linear.y = (
                    -math.sin(yaw) * world_vx + math.cos(yaw) * world_vy)
                out.twist.twist.angular.z = wrap_angle(yaw - pyaw) / dt
        self.previous = (x, y, yaw, stamp)
        self.pub.publish(out)

        if not self.received_first_pose:
            self.received_first_pose = True
            self.get_logger().info(
                "Gazebo ground-truth odometry active: "
                f"world=({x:.2f}, {y:.2f}), yaw={yaw:.2f}")

        transform = TransformStamped()
        transform.header = out.header
        transform.child_frame_id = out.child_frame_id
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = float(msg.pose.position.z)
        transform.transform.rotation = msg.pose.orientation
        self.tf.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboOdomAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
