#!/usr/bin/env python3
"""Closed-loop waypoint MPPI controller for the Ranger corridor world."""
import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from rangermini_dynamic_semantic.command_semantics_mppi import (
    CommandSemanticsMPPI, MPPIConfig,
)


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RangerMPPIWaypointController(Node):
    def __init__(self):
        super().__init__("ranger_mppi_waypoint_controller")
        self.declare_parameter("require_sim_time", True)
        self.declare_parameter("trial_id", "mppi_corridor_trial")
        self.declare_parameter("scenario_id", "corridor_902_to_906")
        self.declare_parameter("seed", 0)
        self.declare_parameter("method_mode", "Ours")
        self.declare_parameter("mppi_method", "command_aware")
        self.declare_parameter("goal_x", 13.35)
        self.declare_parameter("goal_y", 4.15)
        self.declare_parameter("goal_tolerance", 0.42)
        self.declare_parameter("max_speed", 0.30)
        self.declare_parameter("samples_per_behavior", 32)
        self.declare_parameter("horizon", 16)
        self.declare_parameter("account_initial_mode_switch", True)
        self.declare_parameter("allow_reverse", True)
        self.declare_parameter("require_goal_heading", False)
        self.declare_parameter("goal_yaw", 0.0)
        self.declare_parameter("heading_tolerance", 0.15)
        if (bool(self.get_parameter("require_sim_time").value) and
                not bool(self.get_parameter("use_sim_time").value)):
            raise RuntimeError("Ranger MPPI corridor trial requires use_sim_time=true")
        self.goal = (float(self.get_parameter("goal_x").value),
                     float(self.get_parameter("goal_y").value))
        self.goal_id = "fixed_waypoint"
        self.goal_yaw = (float(self.get_parameter("goal_yaw").value)
                         if bool(self.get_parameter("require_goal_heading").value) else None)
        self.odom = None
        self.scan = None
        self.arrived = False
        self.last_plan = None
        self.current_mode = None
        self.mppi = CommandSemanticsMPPI(
            MPPIConfig(
                samples_per_behavior=int(self.get_parameter("samples_per_behavior").value),
                horizon=int(self.get_parameter("horizon").value),
                account_initial_mode_switch=bool(self.get_parameter("account_initial_mode_switch").value),
                allow_reverse=bool(self.get_parameter("allow_reverse").value),
                max_speed=float(self.get_parameter("max_speed").value),
                random_seed=int(self.get_parameter("seed").value) + 17),
            method=str(self.get_parameter("mppi_method").value))
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel_safe", 10)
        self.trace_pub = self.create_publisher(String, "/benchmark/mppi_trace", 10)
        self.result_pub = self.create_publisher(String, "/benchmark/navigation_result", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(LaserScan, "/scan", self.on_scan, 10)
        self.create_subscription(String, "/motion_mode_debug", self.on_mode, 10)
        self.create_subscription(PoseStamped, "/benchmark/mppi_goal", self.on_goal, 10)
        self.create_timer(0.05, self.tick)
        self.get_logger().info(
            f"Ranger corridor MPPI ready: method={self.mppi.method}, goal={self.goal}")

    def trial_context(self):
        return {
            "trial_id": str(self.get_parameter("trial_id").value),
            "scenario_id": str(self.get_parameter("scenario_id").value),
            "seed": int(self.get_parameter("seed").value),
            "method_mode": str(self.get_parameter("method_mode").value),
            "controller_backend": "mppi",
            "mppi_method": self.mppi.method,
        }

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        self.odom = msg

    def on_scan(self, msg):
        self.scan = msg

    def on_mode(self, msg):
        try:
            mode = json.loads(msg.data).get("mode")
        except (ValueError, AttributeError):
            return
        if mode in ("DUAL_ACKERMANN", "PARALLEL", "SPINNING"):
            self.current_mode = mode

    def on_goal(self, msg):
        self.goal = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.goal_id = msg.header.frame_id or "external_waypoint"
        self.goal_yaw = (yaw_from_quaternion(msg.pose.orientation)
                         if bool(self.get_parameter("require_goal_heading").value) else None)
        self.arrived = False
        self.get_logger().info(f"Accepted waypoint {self.goal_id}: {self.goal}")

    def stop(self):
        self.cmd_pub.publish(Twist())

    def tick(self):
        if self.odom is None or self.arrived:
            self.stop()
            return
        pose = self.odom.pose.pose
        state = (float(pose.position.x), float(pose.position.y),
                 yaw_from_quaternion(pose.orientation))
        distance = math.hypot(self.goal[0] - state[0], self.goal[1] - state[1])
        heading_error = (abs(math.atan2(math.sin(self.goal_yaw-state[2]),
                                       math.cos(self.goal_yaw-state[2])))
                         if self.goal_yaw is not None else 0.0)
        if (distance <= float(self.get_parameter("goal_tolerance").value) and
                heading_error <= float(self.get_parameter("heading_tolerance").value)):
            self.arrived = True
            self.stop()
            self.result_pub.publish(String(data=json.dumps({
                "status": "SUCCEEDED", "goal_id": self.goal_id,
                "goal": {"x": self.goal[0], "y": self.goal[1]},
                "distance_m": distance, "timestamp": self.now_sec(),
                "heading_error_rad": heading_error,
                "trial_context": self.trial_context(),
            }, separators=(",", ":"))))
            return
        scan = ((self.scan.ranges, self.scan.angle_min, self.scan.angle_increment)
                if self.scan is not None else None)
        plan = self.mppi.plan(state, self.goal, scan, current_mode=self.current_mode,
                              goal_yaw=self.goal_yaw)
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = plan["action"]
        self.cmd_pub.publish(cmd)
        row = {
            "timestamp": self.now_sec(), "goal_id": self.goal_id,
            "goal": {"x": self.goal[0], "y": self.goal[1]},
            "goal_distance_m": distance,
            "goal_heading_error_rad": heading_error,
            "pose": {"x": state[0], "y": state[1], "yaw": state[2]},
            "requested": {"vx": cmd.linear.x, "vy": cmd.linear.y,
                          "wz": cmd.angular.z},
            "trial_context": self.trial_context(), "plan": plan,
        }
        self.trace_pub.publish(String(data=json.dumps(row, separators=(",", ":"))))

    def destroy_node(self):
        self.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RangerMPPIWaypointController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
