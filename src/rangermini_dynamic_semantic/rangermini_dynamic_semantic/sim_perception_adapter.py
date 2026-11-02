"""Strictly gated simulation-only person perception adapter.

Gazebo world truth terminates here. Public outputs contain only detection
events, negative observations and last-seen knowledge. Exact truth positions
are exposed separately for the UI's explicit developer mode and are never
consumed by mission planning or search execution.
"""
import json
import math
import random
import time
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


NPC_IDS = ("teacher_zhang", "student_li", "visitor")


class SimPerceptionAdapter(Node):
    def __init__(self):
        super().__init__("office_rpg_sim_perception_adapter")
        default_cfg = str(Path(get_package_share_directory(
            "rangermini_dynamic_semantic")) / "config" / "npc_schedules.yaml")
        self.declare_parameter("schedule_file", default_cfg)
        self.declare_parameter("miss_rate", 0.0)
        self.declare_parameter("false_positive_rate", 0.0)
        self.declare_parameter("confidence_noise", 0.03)
        self.declare_parameter("seed", 17)
        self.declare_parameter("last_seen_confirmed_sec", 10.0)
        self.declare_parameter("forced_target_misses", 0)
        with open(str(self.get_parameter("schedule_file").value), encoding="utf-8") as stream:
            self.cfg = yaml.safe_load(stream)
        self.rng = random.Random(int(self.get_parameter("seed").value))
        self.forced_target_misses = int(
            self.get_parameter("forced_target_misses").value)
        self.truth_poses = {}
        self.schedule_meta = {}
        self.robot = None
        self.knowledge = {
            identity: {"identity_id": identity,
                       "display_name": self.cfg[identity]["display_name"],
                       "knowledge_state": "UNKNOWN", "last_seen": None}
            for identity in NPC_IDS
        }
        self.event_pub = self.create_publisher(
            String, "/office_rpg/person_detections", 10)
        self.public_states_pub = self.create_publisher(
            String, "/office_rpg/npc_states", 10)
        self.truth_pub = self.create_publisher(
            String, "/office_rpg/sim/npc_ground_truth", 10)
        self.create_subscription(
            TFMessage, "/world/corridor_902_904_906_908/pose/info",
            self.on_world_poses, 10)
        self.create_subscription(String, "/office_rpg/sim/npc_schedule_state",
                                 self.on_schedule, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(String, "/office_rpg/perception_trigger",
                                 self.on_trigger, 10)
        self.create_timer(0.5, self.publish_states)
        self.get_logger().warn(
            "SIMULATION PERCEPTION STUB: Gazebo truth is isolated inside this adapter")

    def on_world_poses(self, msg):
        for transform in msg.transforms:
            identity = transform.child_frame_id.rsplit("/", 1)[-1]
            if identity in NPC_IDS:
                self.truth_poses[identity] = (
                    float(transform.transform.translation.x),
                    float(transform.transform.translation.y))

    def on_schedule(self, msg):
        try:
            payload = json.loads(msg.data)
            self.schedule_meta = {
                item["identity_id"]: item for item in payload.get("npcs", [])}
        except (ValueError, KeyError) as exc:
            self.get_logger().warn(f"Bad private NPC schedule metadata: {exc}")

    def on_odom(self, msg):
        self.robot = (float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def in_zone(self, region, pose):
        zone = self.cfg.get("detection_zones", {}).get(region)
        if not zone or pose is None:
            return False
        x, y = pose
        return (float(zone["xmin"]) <= x <= float(zone["xmax"]) and
                float(zone["ymin"]) <= y <= float(zone["ymax"]))

    def ranger_at_observation_point(self, region):
        point = self.cfg.get("observation_points", {}).get(region)
        if not point or self.robot is None:
            return False
        return math.hypot(self.robot[0] - float(point["x"]),
                          self.robot[1] - float(point["y"])) <= float(
                              point.get("arrival_radius", 0.6))

    def visible_detection(self, identity, region):
        pose = self.truth_poses.get(identity)
        zone = self.cfg.get("detection_zones", {}).get(region, {})
        if self.robot is None or not self.in_zone(region, pose):
            return None, "target_outside_detection_zone"
        # The region-specific zone is the simulation wall/room visibility rule:
        # entities in a different zone are occluded even if Euclidean distance is small.
        distance = math.hypot(self.robot[0] - pose[0], self.robot[1] - pose[1])
        if distance < float(zone.get("min_distance", 0.0)):
            return None, "inside_minimum_safe_distance"
        if distance > float(zone.get("max_distance", 2.0)):
            return None, "outside_sensor_range"
        if self.rng.random() < float(self.get_parameter("miss_rate").value):
            return None, "configured_simulated_miss"
        meta = self.schedule_meta.get(identity, {})
        confidence = float(meta.get(
            "simulated_detection_confidence",
            self.cfg.get(identity, {}).get("simulated_detection_confidence", 0.8)))
        confidence += self.rng.uniform(-1.0, 1.0) * float(
            self.get_parameter("confidence_noise").value)
        detection = {
            "target_id": identity,
            "display_name": self.cfg[identity]["display_name"],
            "observed_region": region,
            "simulated_confidence": round(max(0.01, min(0.99, confidence)), 3),
            "distance": round(distance, 3), "timestamp": time.time(),
            "source": "simulation_perception_adapter",
        }
        return detection, ""

    def remember(self, detection):
        identity = detection["target_id"]
        self.knowledge[identity] = {
            "identity_id": identity, "display_name": detection["display_name"],
            "knowledge_state": "CONFIRMED",
            "last_seen": {
                "region": detection["observed_region"],
                "timestamp": detection["timestamp"],
                "simulated_confidence": detection["simulated_confidence"],
            },
        }

    def on_trigger(self, msg):
        # Receipt of this message is the explicit active-observation condition.
        try:
            request = json.loads(msg.data)
        except ValueError:
            request = {"region": msg.data}
        request_id = request.get("request_id") or request.get("trigger_id", "")
        region = request.get("region", "")
        target_id = request.get("target_id", "")
        base = {"request_id": request_id, "trigger_id": request_id,
                "mission_id": request.get("mission_id", ""),
                "target_id": target_id, "observed_region": region,
                "timestamp": time.time(),
                "source": "simulation_perception_adapter",
                "simulation_stub": True}
        if request.get("mode") != "ACTIVE_OBSERVATION":
            base.update({"event_type": "TARGET_NOT_FOUND",
                         "reason": "inactive_or_invalid_observation_request"})
            self.event_pub.publish(String(data=json.dumps(base, ensure_ascii=False)))
            return
        if not self.ranger_at_observation_point(region):
            base.update({"event_type": "TARGET_NOT_FOUND",
                         "reason": "ranger_not_at_observation_point"})
            self.event_pub.publish(String(data=json.dumps(base, ensure_ascii=False)))
            return
        if target_id:
            if target_id not in NPC_IDS:
                base.update({"event_type": "TARGET_NOT_FOUND",
                             "reason": "unknown_target_id"})
            elif self.forced_target_misses > 0:
                self.forced_target_misses -= 1
                base.update({"event_type": "TARGET_NOT_FOUND",
                             "reason": "configured_acceptance_forced_miss"})
            else:
                detection, reason = self.visible_detection(target_id, region)
                if detection:
                    self.remember(detection)
                    base.update(detection)
                    base["event_type"] = "TARGET_DETECTED"
                else:
                    base.update({"event_type": "TARGET_NOT_FOUND", "reason": reason})
            self.event_pub.publish(String(data=json.dumps(base, ensure_ascii=False)))
            return
        detections = []
        for identity in NPC_IDS:
            detection, _reason = self.visible_detection(identity, region)
            if detection:
                self.remember(detection)
                detections.append(detection)
        if self.rng.random() < float(self.get_parameter("false_positive_rate").value):
            detections.append({"target_id": "unknown_person", "display_name": "未知人员",
                               "observed_region": region, "simulated_confidence": 0.35,
                               "distance": None, "timestamp": time.time(),
                               "source": "simulation_perception_adapter"})
        base.update({"event_type": "OBSERVATION_COMPLETE", "detections": detections})
        self.event_pub.publish(String(data=json.dumps(base, ensure_ascii=False)))

    def publish_states(self):
        now = time.time()
        confirmed_window = float(self.get_parameter("last_seen_confirmed_sec").value)
        public = []
        for identity in NPC_IDS:
            item = dict(self.knowledge[identity])
            last_seen = item.get("last_seen")
            if last_seen and now - float(last_seen["timestamp"]) > confirmed_window:
                item["knowledge_state"] = "LAST_SEEN"
            # Deliberately no x/y/current_region/schedule state in public output.
            public.append(item)
        self.public_states_pub.publish(String(data=json.dumps({
            "simulation_stub": True, "information_scope": "discovered_only",
            "timestamp": now, "npcs": public}, ensure_ascii=False)))

        truth = []
        for identity in NPC_IDS:
            if identity not in self.truth_poses:
                continue
            meta = self.schedule_meta.get(identity, {})
            truth.append({"identity_id": identity,
                          "display_name": self.cfg[identity]["display_name"],
                          "x": self.truth_poses[identity][0],
                          "y": self.truth_poses[identity][1],
                          "schedule_region": meta.get("current_region", "unknown")})
        self.truth_pub.publish(String(data=json.dumps({
            "simulation_ground_truth": True,
            "warning": "SIMULATION GROUND TRUTH — NOT AVAILABLE ON REAL ROBOT",
            "timestamp": now, "npcs": truth}, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = SimPerceptionAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
