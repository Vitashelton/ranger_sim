#!/usr/bin/env python3
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import json


class CmdVelWatchdog(Node):
    """Fail-safe boundary between planning/filtering and the simulated chassis."""

    def __init__(self):
        super().__init__("cmd_vel_watchdog")
        self.declare_parameter("input_topic", "/cmd_vel_safe_raw")
        self.declare_parameter("output_topic", "/cmd_vel_safe")
        self.declare_parameter("timeout", 0.35)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("enable_depth_guard", False)
        self.declare_parameter("depth_topic", "/camera/depth_image")
        self.declare_parameter("depth_timeout", 0.80)
        self.declare_parameter("obstacle_stop_distance", 0.80)
        self.declare_parameter("obstacle_clear_distance", 0.95)
        self.declare_parameter("min_obstacle_pixels", 24)
        self.declare_parameter("confirm_frames", 2)
        self.declare_parameter("clear_frames", 4)
        self.pub = self.create_publisher(
            Twist, str(self.get_parameter("output_topic").value), 10)
        self.create_subscription(
            Twist, str(self.get_parameter("input_topic").value), self.on_cmd, 10)
        self.depth_guard = bool(self.get_parameter("enable_depth_guard").value)
        self.safety_pub = self.create_publisher(
            String, "/office_rpg/safety_status", 10)
        if self.depth_guard:
            self.create_subscription(
                Image, str(self.get_parameter("depth_topic").value),
                self.on_depth, 10)
        self.last_cmd = self.get_clock().now()
        self.last_depth = self.get_clock().now()
        self.cmd = Twist()
        self.have_cmd = False
        self.have_depth = False
        self.depth_blocked = False
        self.obstacle_frames = 0
        self.clear_frames = 0
        self.nearest_depth = None
        self.protection_pixels = 0
        self.stopped = True
        rate = float(self.get_parameter("publish_rate").value)
        self.create_timer(1.0 / rate, self.tick)

    def on_cmd(self, msg):
        self.cmd = msg
        self.last_cmd = self.get_clock().now()
        self.have_cmd = True

    def on_depth(self, msg):
        """Use only sensor depth; no Gazebo entity or NPC truth enters the gate."""
        self.last_depth = self.get_clock().now()
        self.have_depth = True
        if msg.height == 0 or msg.width == 0 or not msg.data:
            self.obstacle_frames += 1
            return
        try:
            if msg.encoding == "32FC1":
                dtype = np.dtype(">f4" if msg.is_bigendian else "<f4")
                row_values = msg.step // 4
                image = np.frombuffer(msg.data, dtype=dtype).reshape(
                    msg.height, row_values)[:, :msg.width]
                depth_m = image.astype(np.float32, copy=False)
            elif msg.encoding in ("16UC1", "mono16"):
                dtype = np.dtype(">u2" if msg.is_bigendian else "<u2")
                row_values = msg.step // 2
                image = np.frombuffer(msg.data, dtype=dtype).reshape(
                    msg.height, row_values)[:, :msg.width]
                depth_m = image.astype(np.float32) * 0.001
            else:
                self.get_logger().warn(
                    f"Unsupported depth encoding {msg.encoding}; fail-safe stop",
                    throttle_duration_sec=5.0)
                self.obstacle_frames += 1
                return
        except (ValueError, TypeError) as exc:
            self.get_logger().warn(
                f"Malformed depth frame: {exc}; fail-safe stop",
                throttle_duration_sec=5.0)
            self.obstacle_frames += 1
            return

        # Central 60% x 60% is the swept forward body corridor. Sampling every
        # fourth pixel keeps this callback cheap while retaining dense evidence.
        # Keep the lower image out of the ROI: the camera is only 0.42 m above
        # the floor, so including it would turn the floor into a false person.
        y0, y1 = int(msg.height * 0.22), int(msg.height * 0.56)
        x0, x1 = int(msg.width * 0.20), int(msg.width * 0.80)
        roi = depth_m[y0:y1:4, x0:x1:4]
        valid = roi[np.isfinite(roi) & (roi > 0.15)]
        if valid.size == 0:
            self.nearest_depth = None
            self.protection_pixels = 0
            # An empty depth frame is not evidence that the corridor is clear.
            self.obstacle_frames += 1
            self.clear_frames = 0
            return

        self.nearest_depth = float(np.min(valid))
        stop_distance = float(self.get_parameter("obstacle_stop_distance").value)
        clear_distance = float(self.get_parameter("obstacle_clear_distance").value)
        threshold = clear_distance if self.depth_blocked else stop_distance
        self.protection_pixels = int(np.count_nonzero(valid < threshold))
        occupied = self.protection_pixels >= int(
            self.get_parameter("min_obstacle_pixels").value)
        if occupied:
            self.obstacle_frames += 1
            self.clear_frames = 0
            if self.obstacle_frames >= int(self.get_parameter("confirm_frames").value):
                self.depth_blocked = True
        else:
            self.obstacle_frames = 0
            self.clear_frames += 1
            if self.clear_frames >= int(self.get_parameter("clear_frames").value):
                self.depth_blocked = False

    def publish_status(self, state, age):
        payload = {
            "state": state,
            "source": "rgbd_depth_guard" if self.depth_guard else "watchdog_only",
            "nearest_obstacle_distance": self.nearest_depth,
            "protection_point_count": self.protection_pixels,
            "last_depth_age": round(age, 3) if math.isfinite(age) else None,
            "ground_truth_used": False,
        }
        self.safety_pub.publish(String(data=json.dumps(payload)))

    def tick(self):
        age = (self.get_clock().now() - self.last_cmd).nanoseconds * 1e-9
        timed_out = (not self.have_cmd) or age > float(self.get_parameter("timeout").value)
        depth_age = (self.get_clock().now() - self.last_depth).nanoseconds * 1e-9
        depth_timed_out = self.depth_guard and (
            not self.have_depth or depth_age > float(
                self.get_parameter("depth_timeout").value))
        if timed_out or depth_timed_out or self.depth_blocked:
            safe_stop = Twist()
            # A front obstacle must never permit translation.  A pure spin
            # command from the existing retask controller is safe at the
            # measured clearance and lets the forward camera turn away instead
            # of deadlocking forever.  STOP remains a zero command upstream.
            if (self.depth_blocked and not timed_out and not depth_timed_out and
                    abs(self.cmd.linear.x) < 1.0e-3 and
                    abs(self.cmd.linear.y) < 1.0e-3):
                safe_stop.angular.z = self.cmd.angular.z
            self.pub.publish(safe_stop)
            if not self.stopped:
                reason = ("upstream command timeout" if timed_out else
                          "depth sensor timeout" if depth_timed_out else
                          "near obstacle in RGB-D protection zone")
                self.get_logger().warn(f"Safety stop: {reason}")
            self.stopped = True
            state = ("SENSOR_TIMEOUT" if depth_timed_out else
                     "YIELDING" if self.depth_blocked else "STOPPED")
        else:
            self.pub.publish(self.cmd)
            self.stopped = False
            state = "CLEAR"
        self.publish_status(state, depth_age if self.have_depth else math.inf)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelWatchdog()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Publish several zero frames for transports/controllers that sample commands.
        if rclpy.ok():
            for _ in range(3):
                node.pub.publish(Twist())
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
