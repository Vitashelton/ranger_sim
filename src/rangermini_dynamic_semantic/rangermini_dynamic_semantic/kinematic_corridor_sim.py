\
#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster

def quat(yaw):
    h = yaw * 0.5
    return (0.0, 0.0, math.sin(h), math.cos(h))

class KinematicCorridorSim(Node):
    """
    Kinematic chassis simulator with RangerMini 2.0-style four steering modules.

    This is not a full Gazebo physics plugin. It is a ROS2 kinematic approximation:
    /cmd_vel_safe is treated as the chassis twist, and each steering module angle
    is computed from the local wheel velocity:
        v_i = [vx - wz*y_i, vy + wz*x_i]
        steer_i = atan2(v_i_y, v_i_x)
    """
    def __init__(self):
        super().__init__("kinematic_corridor_sim")
        self.declare_parameter("cmd_topic", "/cmd_vel_safe")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("start_x", 1.2)
        self.declare_parameter("start_y", 2.3)
        self.declare_parameter("start_yaw_deg", 0.0)
        self.declare_parameter("goal_x", 13.35)
        self.declare_parameter("goal_y", 4.15)
        self.declare_parameter("stop_radius", 0.40)
        self.declare_parameter("world_x_min", 0.0)
        self.declare_parameter("world_x_max", 22.0)
        self.declare_parameter("world_y_min", -0.2)
        self.declare_parameter("world_y_max", 7.8)
        self.declare_parameter("body_length", 0.72)
        self.declare_parameter("body_width", 0.55)
        # Ranger Mini 2.0 values from agilexrobotics/ranger_ros2 (humble).
        self.declare_parameter("wheelbase", 0.494)
        self.declare_parameter("track", 0.364)
        self.declare_parameter("max_linear_speed", 1.5)
        self.declare_parameter("max_angular_speed", 4.8)
        self.declare_parameter("min_turn_radius", 0.4764)
        self.declare_parameter("max_steer_angle_ackermann", 0.6981)
        self.declare_parameter("max_steer_angle_parallel", 1.570)
        self.declare_parameter("enforce_ranger_motion_modes", True)
        self.declare_parameter("cmd_timeout", 0.35)
        self.declare_parameter("max_steer_rate", 2.5)

        self.x = float(self.get_parameter("start_x").value)
        self.y = float(self.get_parameter("start_y").value)
        self.yaw = math.radians(float(self.get_parameter("start_yaw_deg").value))
        self.cmd = Twist()
        self.last_cmd = self.get_clock().now()
        self.steer_angles = [0.0] * 4
        self.watchdog_active = False
        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)

        self.odom_pub = self.create_publisher(Odometry, "/odom", 20)
        self.path_pub = self.create_publisher(Path, "/debug/executed_path", 10)
        self.robot_pub = self.create_publisher(Marker, "/debug/robot_footprint", 10)
        self.modules_pub = self.create_publisher(MarkerArray, "/debug/rangermini2_steer_modules", 10)
        self.mode_pub = self.create_publisher(String, "/motion_mode_debug", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Twist, self.get_parameter("cmd_topic").value, self.on_cmd, 10)
        self.create_subscription(String, "/task_goal", self.on_task_goal, 10)

        self.path = Path()
        self.path.header.frame_id = "odom"
        self.last = self.get_clock().now()
        self.timer = self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.step)

    def on_cmd(self, msg):
        self.cmd = msg
        self.last_cmd = self.get_clock().now()

    def on_task_goal(self, msg):
        goals = {"902": (3.0, 4.15), "904": (8.0, 4.15),
                 "906": (13.35, 4.15), "908": (18.55, 4.15)}
        room = msg.data.strip()
        if room in goals:
            self.goal_x, self.goal_y = goals[room]

    @staticmethod
    def wrap(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def reached_goal(self):
        sr = float(self.get_parameter("stop_radius").value)
        return math.hypot(self.x - self.goal_x, self.y - self.goal_y) <= sr

    def ranger_motion_constraint(self, vx, vy, wz):
        """Mirror the motion-mode selection in the official Ranger ROS2 driver.

        Ranger Mini 2.0 does not execute arbitrary holonomic vx/vy/wz together:
        lateral input selects parallel steering, pure yaw selects spinning, and
        the remaining commands use dual Ackermann steering. Infeasible tight
        arcs retain Ackermann mode and clamp curvature to the physical limit,
        matching this workspace's patched ranger_messenger.cpp.
        """
        max_v = float(self.get_parameter("max_linear_speed").value)
        max_w = float(self.get_parameter("max_angular_speed").value)
        speed = math.hypot(vx, vy)
        if speed > max_v:
            scale = max_v / max(speed, 1e-9)
            vx, vy = vx * scale, vy * scale
        wz = max(-max_w, min(max_w, wz))

        if abs(vy) > 1e-4:
            # Match the driver's signed-speed/common-steering representation.
            steer = math.atan2(vy, vx if abs(vx) > 1e-9 else 0.0)
            signed_speed = speed
            if steer > math.pi / 2.0:
                steer -= math.pi
                signed_speed = -signed_speed
            elif steer < -math.pi / 2.0:
                steer += math.pi
                signed_speed = -signed_speed
            limit = float(self.get_parameter("max_steer_angle_parallel").value)
            steer = max(-limit, min(limit, steer))
            signed_speed = max(-max_v, min(max_v, signed_speed))
            return (signed_speed * math.cos(steer),
                    signed_speed * math.sin(steer), 0.0, "PARALLEL")

        if abs(vx) <= 1e-4 and abs(wz) > 1e-4:
            return 0.0, 0.0, wz, "SPINNING"
        if abs(vx) <= 1e-4 or abs(wz) <= 1e-4:
            return vx, 0.0, 0.0, "DUAL_ACKERMANN"

        radius = abs(vx) / abs(wz)
        min_radius = float(self.get_parameter("min_turn_radius").value)
        if radius < min_radius:
            wz = math.copysign(abs(vx) / min_radius, wz)
            radius = min_radius

        # Same inner-to-central conversion used by ranger_ros2 odometry.
        wheelbase = float(self.get_parameter("wheelbase").value)
        track = float(self.get_parameter("track").value)
        sign = 1.0 if vx * wz >= 0.0 else -1.0
        inner = sign * min(
            math.atan((wheelbase * 0.5) / radius),
            float(self.get_parameter("max_steer_angle_ackermann").value))
        central = math.atan2(
            wheelbase * math.sin(abs(inner)),
            wheelbase * math.cos(abs(inner)) + track * math.sin(abs(inner)))
        central = math.copysign(central, inner)
        actual_wz = 2.0 * vx * math.sin(central) / wheelbase
        return vx, 0.0, actual_wz, "DUAL_ACKERMANN"

    def step(self):
        now = self.get_clock().now()
        dt = min(max((now - self.last).nanoseconds * 1e-9, 0.0), 0.08)
        self.last = now

        cmd_age = (now - self.last_cmd).nanoseconds * 1e-9
        stale = cmd_age > float(self.get_parameter("cmd_timeout").value)
        if stale:
            vx = vy = wz = 0.0
            if not self.watchdog_active:
                self.get_logger().warn(
                    f"Command timeout ({cmd_age:.2f}s): forcing chassis stop")
            self.watchdog_active = True
        else:
            vx, vy, wz = self.cmd.linear.x, self.cmd.linear.y, self.cmd.angular.z
            self.watchdog_active = False
        mode = "STOPPED"
        if not stale and bool(self.get_parameter("enforce_ranger_motion_modes").value):
            vx, vy, wz, mode = self.ranger_motion_constraint(vx, vy, wz)
        elif not stale:
            mode = "UNCONSTRAINED"
        if self.reached_goal():
            vx = vy = wz = 0.0
            mode = "GOAL_STOP"

        mode_msg = String()
        mode_msg.data = mode
        self.mode_pub.publish(mode_msg)

        dx = math.cos(self.yaw) * vx - math.sin(self.yaw) * vy
        dy = math.sin(self.yaw) * vx + math.cos(self.yaw) * vy
        self.x += dx * dt
        self.y += dy * dt
        self.yaw += wz * dt

        self.x = min(max(self.x, float(self.get_parameter("world_x_min").value)),
                     float(self.get_parameter("world_x_max").value))
        self.y = min(max(self.y, float(self.get_parameter("world_y_min").value)),
                     float(self.get_parameter("world_y_max").value))

        self.pub_odom(now, vx, vy, wz)
        self.pub_path(now)
        self.pub_robot(now)
        self.pub_steer_modules(now, vx, vy, wz, dt)

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
        self.path.poses = self.path.poses[-1200:]
        self.path_pub.publish(self.path)

    def pub_robot(self, now):
        qx, qy, qz, qw = quat(self.yaw)
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = now.to_msg()
        m.ns = "rangermini_body"
        m.id = 0
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = self.x
        m.pose.position.y = self.y
        m.pose.position.z = 0.11
        m.pose.orientation.x = qx
        m.pose.orientation.y = qy
        m.pose.orientation.z = qz
        m.pose.orientation.w = qw
        m.scale.x = float(self.get_parameter("body_length").value)
        m.scale.y = float(self.get_parameter("body_width").value)
        m.scale.z = 0.18
        m.color.r = 0.10
        m.color.g = 0.35
        m.color.b = 1.00
        m.color.a = 0.82
        self.robot_pub.publish(m)

    def pub_steer_modules(self, now, vx, vy, wz, dt):
        L = float(self.get_parameter("wheelbase").value)
        W = float(self.get_parameter("track").value)
        wheel_positions = [
            ("front_left",  L/2,  W/2),
            ("front_right", L/2, -W/2),
            ("rear_left",  -L/2,  W/2),
            ("rear_right", -L/2, -W/2),
        ]
        arr = MarkerArray()
        for i, (name, wx, wy) in enumerate(wheel_positions):
            vix = vx - wz * wy
            viy = vy + wz * wx
            # Choose the equivalent steering solution requiring the least rotation.
            # Reversing wheel speed makes delta and delta+pi kinematically identical.
            if abs(vix) + abs(viy) > 1e-4:
                desired = math.atan2(viy, vix)
                alternative = self.wrap(desired + math.pi)
                if abs(self.wrap(alternative - self.steer_angles[i])) < abs(
                        self.wrap(desired - self.steer_angles[i])):
                    desired = alternative
                max_step = float(self.get_parameter("max_steer_rate").value) * dt
                error = self.wrap(desired - self.steer_angles[i])
                self.steer_angles[i] = self.wrap(
                    self.steer_angles[i] + max(-max_step, min(max_step, error)))
            steer = self.steer_angles[i]
            _, _, qz, qw = quat(steer)

            # steering module / wheel
            m = Marker()
            m.header.frame_id = "base_link"
            m.header.stamp = now.to_msg()
            m.ns = "rangermini2_steer_wheels"
            m.id = i
            m.type = Marker.CUBE
            m.action = Marker.ADD
            m.pose.position.x = wx
            m.pose.position.y = wy
            m.pose.position.z = -0.03
            m.pose.orientation.z = qz
            m.pose.orientation.w = qw
            m.scale.x = 0.20
            m.scale.y = 0.075
            m.scale.z = 0.10
            m.color.r = 0.02
            m.color.g = 0.02
            m.color.b = 0.02
            m.color.a = 1.0
            arr.markers.append(m)

            # steering axis cap
            c = Marker()
            c.header.frame_id = "base_link"
            c.header.stamp = now.to_msg()
            c.ns = "rangermini2_steer_caps"
            c.id = 100 + i
            c.type = Marker.CYLINDER
            c.action = Marker.ADD
            c.pose.position.x = wx
            c.pose.position.y = wy
            c.pose.position.z = 0.055
            c.pose.orientation.w = 1.0
            c.scale.x = 0.09
            c.scale.y = 0.09
            c.scale.z = 0.04
            c.color.r = 0.15
            c.color.g = 0.15
            c.color.b = 0.16
            c.color.a = 0.95
            arr.markers.append(c)
        self.modules_pub.publish(arr)

def main(args=None):
    rclpy.init(args=args)
    n = KinematicCorridorSim()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        # The internal timeout is the actual safety guarantee. Clearing the stored
        # command here also makes orderly shutdown explicit for component reuse.
        n.cmd = Twist()
        n.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
