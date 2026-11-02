"""RGB-D semantic observations without Gazebo pose or entity truth.

The default benchmark detector segments configured semantic colours from the
rendered image, measures range from the aligned depth image and projects the
centroid through odometry.  The detector is intentionally replaceable by a
TensorRT model on the D435i while preserving the output contract.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

try:
    import cv2
except ImportError:  # pragma: no cover - reported clearly at runtime
    cv2 = None


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class RgbdSemanticPerception(Node):
    def __init__(self):
        super().__init__("rgbd_semantic_perception")
        default = str(Path(get_package_share_directory(
            "rangermini_dynamic_semantic")) / "config" / "dynamic_benchmark.yaml")
        self.declare_parameter("config_file", default)
        self.declare_parameter("rgb_topic", "/camera/image")
        self.declare_parameter("depth_topic", "/camera/depth_image")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("camera_forward_offset_m", 0.34)
        self.declare_parameter("max_depth_age_sec", 0.20)
        self.declare_parameter("require_sim_time", True)
        self.declare_parameter("trial_id", "manual_trial")
        self.declare_parameter("scenario_id", "S6")
        self.declare_parameter("seed", 0)
        self.declare_parameter("method_mode", "Ours")
        self.assert_time_contract()
        with open(str(self.get_parameter("config_file").value), encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.classes = self.cfg.get("semantic_classes", {})
        self.rgb = None
        self.depth = None
        self.info = None
        self.odom = None
        self.frame_counter = 0
        self.processed_counter = 0
        self.dropped_counter = 0
        self.task_context_value = {}
        self.pub = self.create_publisher(String, "/semantic_observations", 20)
        self.stats_pub = self.create_publisher(String, "/semantic_perception/stats", 10)
        self.create_subscription(Image, str(self.get_parameter("rgb_topic").value), self.on_rgb, 5)
        self.create_subscription(Image, str(self.get_parameter("depth_topic").value), self.on_depth, 5)
        self.create_subscription(CameraInfo, str(self.get_parameter("camera_info_topic").value), self.on_info, 5)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(String, "/semantic_perception/trigger", self.on_trigger, 10)
        task_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/task_context/current", self.on_task_context, task_qos)
        self.create_timer(1.0, self.publish_stats)
        if cv2 is None:
            self.get_logger().error("python3-opencv is required by rgbd_semantic_perception")
        else:
            self.get_logger().info(
                "RGB-D semantic perception active; no Gazebo truth subscription")

    def assert_time_contract(self):
        if (bool(self.get_parameter("require_sim_time").value) and
                not bool(self.get_parameter("use_sim_time").value)):
            raise RuntimeError("rgbd_semantic_perception requires use_sim_time=true")

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1.0e-9

    def trial_context(self, trigger=None):
        if isinstance(trigger, dict):
            inherited = trigger.get("trial_context")
            if isinstance(inherited, dict):
                return inherited
        return {
            "trial_id": str(self.get_parameter("trial_id").value),
            "scenario_id": str(self.get_parameter("scenario_id").value),
            "seed": int(self.get_parameter("seed").value),
            "method_mode": str(self.get_parameter("method_mode").value),
            "task_context": dict(self.task_context_value),
        }

    def on_task_context(self, msg):
        try:
            payload = json.loads(msg.data)
            self.task_context_value = payload.get("task_context", {})
        except ValueError:
            return

    def on_rgb(self, msg):
        self.rgb = msg
        self.frame_counter += 1

    def on_depth(self, msg):
        self.depth = msg

    def on_info(self, msg):
        self.info = msg

    def on_odom(self, msg):
        self.odom = msg

    @staticmethod
    def stamp(msg):
        return float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)

    @staticmethod
    def decode_rgb(msg):
        channels = 4 if msg.encoding.lower() in ("rgba8", "bgra8") else 3
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        rows = raw.reshape(msg.height, msg.step)[:, :msg.width * channels]
        image = rows.reshape(msg.height, msg.width, channels)[:, :, :3]
        if msg.encoding.lower().startswith("bgr"):
            image = image[:, :, ::-1]
        return image

    @staticmethod
    def decode_depth(msg):
        if msg.encoding.upper() == "32FC1":
            dtype, scale = np.float32, 1.0
        elif msg.encoding.upper() in ("16UC1", "MONO16"):
            dtype, scale = np.uint16, 0.001
        else:
            raise ValueError(f"unsupported depth encoding {msg.encoding}")
        itemsize = np.dtype(dtype).itemsize
        raw = np.frombuffer(msg.data, dtype=dtype)
        rows = raw.reshape(msg.height, msg.step // itemsize)[:, :msg.width]
        return rows.astype(np.float32) * scale

    def project(self, u, v, depth_m):
        fx, fy = float(self.info.k[0]), float(self.info.k[4])
        cx, cy = float(self.info.k[2]), float(self.info.k[5])
        if fx <= 1e-6 or fy <= 1e-6:
            return None
        # Optical x points right and optical z points forward.
        lateral_left = -(float(u) - cx) * depth_m / fx
        forward = depth_m + float(self.get_parameter("camera_forward_offset_m").value)
        pose = self.odom.pose.pose
        yaw = yaw_from_quaternion(pose.orientation)
        return {
            "x": float(pose.position.x) + math.cos(yaw) * forward - math.sin(yaw) * lateral_left,
            "y": float(pose.position.y) + math.sin(yaw) * forward + math.cos(yaw) * lateral_left,
            "z": max(0.0, -(float(v) - cy) * depth_m / fy + 0.48),
        }

    def on_trigger(self, msg):
        started = time.perf_counter()
        try:
            trigger = json.loads(msg.data)
        except ValueError:
            trigger = {"reason": msg.data}
        if cv2 is None or any(value is None for value in (self.rgb, self.depth, self.info, self.odom)):
            self.dropped_counter += 1
            return
        if abs(self.stamp(self.rgb) - self.stamp(self.depth)) > float(
                self.get_parameter("max_depth_age_sec").value):
            self.dropped_counter += 1
            return
        try:
            rgb = self.decode_rgb(self.rgb)
            depth = self.decode_depth(self.depth)
        except (ValueError, TypeError) as exc:
            self.dropped_counter += 1
            self.get_logger().warn(f"RGB-D decode failed: {exc}")
            return

        detections = []
        rgb_i16 = rgb.astype(np.int16)
        for category, spec in self.classes.items():
            target = np.asarray(spec["rgb"], dtype=np.int16).reshape(1, 1, 3)
            distance = np.linalg.norm(rgb_i16 - target, axis=2)
            mask = (distance <= float(spec.get("tolerance", 70))).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area < int(spec.get("min_pixels", 40)):
                    continue
                x, y, w, h = (int(stats[component, key]) for key in (
                    cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                    cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                region_depth = depth[y:y+h, x:x+w]
                region_mask = mask[y:y+h, x:x+w] > 0
                valid = region_depth[region_mask & np.isfinite(region_depth) &
                                     (region_depth > 0.15) & (region_depth < 12.0)]
                if valid.size < max(8, area // 20):
                    continue
                range_m = float(np.median(valid))
                u, v = centroids[component]
                point = self.project(u, v, range_m)
                if point is None:
                    continue
                colour_quality = max(0.0, 1.0 - float(np.mean(
                    distance[y:y+h, x:x+w][region_mask])) /
                    max(1.0, float(spec.get("tolerance", 70))))
                confidence = min(0.99, 0.45 + 0.35 * colour_quality +
                                 0.20 * min(1.0, area / 900.0))
                detections.append({
                    "category": category, "confidence": round(confidence, 4),
                    "position": {k: round(vv, 4) for k, vv in point.items()},
                    "range_m": round(range_m, 4), "pixel_area": area,
                    "bbox_xywh": [x, y, w, h],
                    "task_tags": list(spec.get("task_tags", [])),
                })
        self.processed_counter += 1
        payload = {
            "observation_id": self.processed_counter,
            "source": "rendered_rgbd", "uses_sim_ground_truth": False,
            "sensor_stamp": self.stamp(self.rgb), "timestamp": self.now_sec(),
            "trigger": trigger, "detections": detections,
            "processing_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "trial_context": self.trial_context(trigger),
        }
        self.pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def publish_stats(self):
        self.stats_pub.publish(String(data=json.dumps({
            "rgb_frames_received": self.frame_counter,
            "frames_processed": self.processed_counter,
            "triggers_dropped": self.dropped_counter,
            "processing_ratio": round(self.processed_counter /
                                       max(1, self.frame_counter), 4),
            "uses_sim_ground_truth": False,
            "timestamp": self.now_sec(), "trial_context": self.trial_context(),
        }, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = RgbdSemanticPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
