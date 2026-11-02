#!/usr/bin/env python3
"""Apply RangerMini 4WS kinematics before sending body twist to Gazebo."""

import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.clock import Clock, ClockType
from std_msgs.msg import String
from .ranger_4ws_kinematics import Ranger4WSKinematics


def ranger_mini_v2_command(vx, vy, wz, *, wheelbase=0.494, track=0.364,
                           min_turn_radius=0.4764, max_linear_speed=1.5,
                           max_angular_speed=4.8, linear_deadband=1e-4,
                           angular_deadband=1e-4):
    """Return executed (vx, vy, wz), mode name, and reason.

    Mirrors ranger_messenger.cpp: lateral input selects PARALLEL and discards
    yaw; pure yaw selects SPINNING; all other motion is DUAL_ACKERMANN.  The
    workspace's patched driver clamps an infeasible Ackermann curvature to the
    minimum radius rather than reinterpreting it as a spin.
    """
    vx, vy, wz = float(vx), float(vy), float(wz)
    speed = math.hypot(vx, vy)
    if speed > max_linear_speed:
        scale = max_linear_speed / speed
        vx, vy = vx * scale, vy * scale
    wz = max(-max_angular_speed, min(max_angular_speed, wz))

    if math.hypot(vx, vy) <= linear_deadband and abs(wz) <= angular_deadband:
        return 0.0, 0.0, 0.0, "STOPPED", "zero command"

    if abs(vy) > linear_deadband:
        # Ranger's parallel command is speed + common steering angle; preserve
        # the requested body-frame translation and explicitly ignore yaw.
        reason = "parallel steering; angular.z ignored" if abs(wz) > angular_deadband else "parallel steering"
        return vx, vy, 0.0, "PARALLEL", reason

    if abs(vx) <= linear_deadband and abs(wz) > angular_deadband:
        return 0.0, 0.0, wz, "SPINNING", "pure angular command"

    if abs(vx) <= linear_deadband:
        return 0.0, 0.0, 0.0, "STOPPED", "below Ranger command deadband"

    if abs(wz) <= angular_deadband:
        return vx, 0.0, 0.0, "DUAL_ACKERMANN", "straight Ackermann motion"

    requested_radius = abs(vx / wz)
    if requested_radius < min_turn_radius:
        wz = math.copysign(abs(vx) / min_turn_radius, wz)
        reason = f"Ackermann radius clamped from {requested_radius:.4f} m to {min_turn_radius:.4f} m"
    else:
        reason = "feasible Ackermann arc"

    # The chassis steering command is speed + steering angle.  Reconstruct the
    # executed yaw rate from RangerMiniV2 wheel geometry, matching UpdateOdometry.
    radius = abs(vx / wz)
    inner = math.copysign(
        min(math.atan((wheelbase * 0.5) / radius), math.radians(40.0)),
        vx * wz,
    )
    inner_abs = abs(inner)
    central = math.atan2(
        wheelbase * math.sin(inner_abs),
        wheelbase * math.cos(inner_abs) + track * math.sin(inner_abs),
    )
    central = math.copysign(central, inner)
    executed_wz = 2.0 * vx * math.sin(central) / wheelbase
    return vx, 0.0, executed_wz, "DUAL_ACKERMANN", reason


class RangerMotionAdapter(Node):
    def __init__(self):
        super().__init__("ranger_mini_v2_motion_adapter")
        self.declare_parameter("input_topic", "/cmd_vel_safe")
        self.declare_parameter("output_topic", "/cmd_vel_gazebo")
        self.declare_parameter("debug_topic", "/motion_mode_debug")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("audit_rate_hz", 10.0)
        self.declare_parameter("command_timeout", 0.35)
        self.declare_parameter("wall_command_timeout", 1.5)
        self.declare_parameter("wheelbase", 0.494)
        self.declare_parameter("track", 0.364)
        self.declare_parameter("min_turn_radius", 0.4764)
        self.declare_parameter("max_linear_speed", 1.5)
        self.declare_parameter("max_angular_speed", 4.8)
        self.declare_parameter("max_steer_rate", 2.5)
        self.declare_parameter("max_ackermann_angle", math.radians(40.0))
        self.declare_parameter("max_parallel_angle", math.pi/2)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.timeout = float(self.get_parameter("command_timeout").value)
        self.wall_timeout = float(self.get_parameter("wall_command_timeout").value)
        self.last_command_wall = None
        self.params = {
            "wheelbase": float(self.get_parameter("wheelbase").value),
            "track": float(self.get_parameter("track").value),
            "min_turn_radius": float(self.get_parameter("min_turn_radius").value),
            "max_linear_speed": float(self.get_parameter("max_linear_speed").value),
            "max_angular_speed": float(self.get_parameter("max_angular_speed").value),
        }
        self.kinematics = Ranger4WSKinematics(
            wheelbase=self.params["wheelbase"], track=self.params["track"],
            max_speed=self.params["max_linear_speed"],
            max_steer_rate=float(self.get_parameter("max_steer_rate").value),
            max_ackermann_angle=float(self.get_parameter("max_ackermann_angle").value),
            max_parallel_angle=float(self.get_parameter("max_parallel_angle").value))
        self.latest = Twist()
        self.last_command_time = None
        self.last_debug_signature = None
        self.last_debug_publish = None
        self.last_kinematic_time = self.get_clock().now()
        self.audit_period = 1.0 / max(
            1.0, float(self.get_parameter("audit_rate_hz").value))
        self.output = self.create_publisher(Twist, self.output_topic, 10)
        self.debug = self.create_publisher(String, self.debug_topic, 10)
        self.create_subscription(Twist, self.input_topic, self.on_command, 10)
        hz = max(1.0, float(self.get_parameter("rate_hz").value))
        self.timer = self.create_timer(1.0 / hz, self.on_timer)
        self.wall_watchdog = self.create_timer(.05, self.on_wall_watchdog,
            clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.get_logger().info(
            f"Ranger Mini 2.0 mode adapter: {self.input_topic} -> {self.output_topic}; "
            f"Ackermann minimum radius={self.params['min_turn_radius']:.4f} m")

    def on_command(self, msg):
        values = (msg.linear.x, msg.linear.y, msg.angular.z)
        if not all(math.isfinite(v) for v in values):
            self.latest = Twist()
            self.last_command_wall = None
            self.output.publish(Twist())
            return
        self.latest = msg
        self.last_command_wall = time.monotonic()
        self.last_command_time = self.get_clock().now()

    def wall_stale(self):
        return (self.last_command_wall is None or
                time.monotonic()-self.last_command_wall > self.wall_timeout)

    def on_wall_watchdog(self):
        # This stop remains live if the ROS/simulation clock stops advancing.
        if self.wall_stale():
            self.output.publish(Twist())

    def on_timer(self):
        now = self.get_clock().now()
        dt = max(0.0, min(0.08, (now-self.last_kinematic_time).nanoseconds*1e-9))
        self.last_kinematic_time = now
        stale = (self.wall_stale() or self.last_command_time is None or
                 (now - self.last_command_time).nanoseconds * 1e-9 > self.timeout)
        source = Twist() if stale else self.latest
        cmd_vx, cmd_vy, cmd_wz, mode, reason = ranger_mini_v2_command(
            source.linear.x, source.linear.y, source.angular.z, **self.params)
        (vx, vy, wz), mode, steering, steering_transition = self.kinematics.step(
            cmd_vx, cmd_vy, cmd_wz, dt)
        out = Twist()
        out.linear.x, out.linear.y, out.angular.z = vx, vy, wz
        self.output.publish(out)
        payload = {
            "mode": mode,
            "reason": "command timeout; safe stop" if stale else reason,
            "stale": stale,
            "wall_stale": self.wall_stale(),
            "requested": {"vx": source.linear.x, "vy": source.linear.y, "wz": source.angular.z},
            "executed": {"vx": vx, "vy": vy, "wz": wz},
            "wheel_steering_rad": list(steering),
            "steering_transition_stop": steering_transition,
            "kinematic_model": "4WS rigid-body pure-rolling wheel constraints",
        }
        signature = (mode, payload["reason"], stale, steering_transition)
        audit_due = (self.last_debug_publish is None or
                     (now - self.last_debug_publish).nanoseconds * 1e-9 >= self.audit_period)
        if signature != self.last_debug_signature or audit_due:
            self.debug.publish(String(data=json.dumps(payload, separators=(",", ":"))))
            self.last_debug_signature = signature
            self.last_debug_publish = now


def main(args=None):
    rclpy.init(args=args)
    node = RangerMotionAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.output.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
