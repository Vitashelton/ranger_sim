#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster

def quat(yaw):
    h = yaw * 0.5
    return (0.0, 0.0, math.sin(h), math.cos(h))

class KinematicRangerMiniSim(Node):
    def __init__(self):
        super().__init__("kinematic_rangermini_sim")
        self.declare_parameter("cmd_topic", "/cmd_vel_safe")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("start_x", 0.0)
        self.declare_parameter("start_y", -1.65)
        self.declare_parameter("start_yaw_deg", 90.0)
        self.declare_parameter("goal_y", 2.75)
        self.declare_parameter("world_x_limit", 2.8)
        self.declare_parameter("world_y_min", -2.2)
        self.declare_parameter("world_y_max", 3.2)

        self.x = float(self.get_parameter("start_x").value)
        self.y = float(self.get_parameter("start_y").value)
        self.yaw = math.radians(float(self.get_parameter("start_yaw_deg").value))
        self.cmd = Twist()

        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.path_pub = self.create_publisher(Path, "/debug/executed_path", 10)
        self.robot_pub = self.create_publisher(Marker, "/debug/robot_footprint", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, self.get_parameter("cmd_topic").value, self.on_cmd, 10)

        self.path = Path()
        self.path.header.frame_id = "odom"
        self.last = self.get_clock().now()
        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.step)

    def on_cmd(self, msg):
        self.cmd = msg

    def step(self):
        now = self.get_clock().now()
        dt = min(max((now - self.last).nanoseconds * 1e-9, 0.0), 0.08)
        self.last = now

        vx, vy, wz = self.cmd.linear.x, self.cmd.linear.y, self.cmd.angular.z
        if self.y >= float(self.get_parameter("goal_y").value):
            vx = vy = wz = 0.0

        dx = math.cos(self.yaw) * vx - math.sin(self.yaw) * vy
        dy = math.sin(self.yaw) * vx + math.cos(self.yaw) * vy
        self.x += dx * dt
        self.y += dy * dt
        self.yaw += wz * dt

        xlim = float(self.get_parameter("world_x_limit").value)
        self.x = min(max(self.x, -xlim), xlim)
        self.y = min(max(self.y, float(self.get_parameter("world_y_min").value)),
                     float(self.get_parameter("world_y_max").value))

        self.pub_odom(now, vx, vy, wz)
        self.pub_path(now)
        self.pub_robot(now)

    def pub_odom(self, now, vx, vy, wz):
        qx, qy, qz, qw = quat(self.yaw)
        o = Odometry()
        o.header.stamp = now.to_msg()
        o.header.frame_id = "odom"
        o.child_frame_id = "base_link"
        o.pose.pose.position.x = self.x
        o.pose.pose.position.y = self.y
        o.pose.pose.orientation.x = qx
        o.pose.pose.orientation.y = qy
        o.pose.pose.orientation.z = qz
        o.pose.pose.orientation.w = qw
        o.twist.twist.linear.x = vx
        o.twist.twist.linear.y = vy
        o.twist.twist.angular.z = wz
        self.odom_pub.publish(o)

        t = TransformStamped()
        t.header.stamp = now.to_msg()
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw
        self.tf.sendTransform(t)

    def pub_path(self, now):
        qx, qy, qz, qw = quat(self.yaw)
        p = PoseStamped()
        p.header.stamp = now.to_msg()
        p.header.frame_id = "odom"
        p.pose.position.x = self.x
        p.pose.position.y = self.y
        p.pose.orientation.x = qx
        p.pose.orientation.y = qy
        p.pose.orientation.z = qz
        p.pose.orientation.w = qw
        self.path.header.stamp = now.to_msg()
        self.path.poses.append(p)
        self.path.poses = self.path.poses[-600:]
        self.path_pub.publish(self.path)

    def pub_robot(self, now):
        qx, qy, qz, qw = quat(self.yaw)
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = now.to_msg()
        m.ns = "rangermini_footprint"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = self.x
        m.pose.position.y = self.y
        m.pose.position.z = 0.08
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.scale.x = 0.62
        m.scale.y = 0.82
        m.scale.z = 0.16
        m.color.r = 0.1
        m.color.g = 0.35
        m.color.b = 1.0
        m.color.a = 0.9
        self.robot_pub.publish(m)

def main(args=None):
    rclpy.init(args=args)
    n = KinematicRangerMiniSim()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
