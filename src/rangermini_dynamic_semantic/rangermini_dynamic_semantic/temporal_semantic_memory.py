"""Time-aware object memory with evidence fusion and explicit lifecycle."""
import json
import math
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class TemporalSemanticMemory(Node):
    def __init__(self):
        super().__init__("temporal_semantic_memory")
        default = str(Path(get_package_share_directory(
            "rangermini_dynamic_semantic")) / "config" / "dynamic_benchmark.yaml")
        self.declare_parameter("config_file", default)
        self.declare_parameter("require_sim_time", True)
        self.declare_parameter("trial_id", "manual_trial")
        self.declare_parameter("scenario_id", "S6")
        self.declare_parameter("seed", 0)
        self.declare_parameter("method_mode", "Ours")
        self.assert_time_contract()
        with open(str(self.get_parameter("config_file").value), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg.get("memory", {})
        self.regions = cfg.get("regions", {})
        self.half_life = float(self.cfg.get("confidence_half_life_sec", 35.0))
        self.stale_after = float(self.cfg.get("stale_after_sec", 28.0))
        self.remove_after = float(self.cfg.get("remove_after_sec", 95.0))
        self.association_radius = float(self.cfg.get("association_radius_m", 1.0))
        self.confirm_observations = int(self.cfg.get("confirm_observations", 2))
        self.update_threshold = float(self.cfg.get("task_update_threshold", 0.48))
        self.tracks = {}
        self.next_id = 1
        self.update_stats = {"accepted": 0, "task_deferred": 0}
        self.task = {}
        self.snapshot_pub = self.create_publisher(String, "/semantic_memory_v2/snapshot", 10)
        self.event_pub = self.create_publisher(String, "/semantic_memory_v2/events", 20)
        self.status_pub = self.create_publisher(String, "/streaming/memory_status", 10)
        self.create_subscription(String, "/semantic_observations", self.on_observation, 20)
        task_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/task_context/current", self.on_task, task_qos)
        self.create_timer(0.5, self.tick)

    def on_task(self, msg):
        try:
            payload = json.loads(msg.data)
            self.task = payload.get("task_context", payload)
        except ValueError:
            return

    def assert_time_contract(self):
        if (bool(self.get_parameter("require_sim_time").value) and
                not bool(self.get_parameter("use_sim_time").value)):
            raise RuntimeError("temporal_semantic_memory requires use_sim_time=true")

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1.0e-9

    def trial_context(self):
        return {
            "trial_id": str(self.get_parameter("trial_id").value),
            "scenario_id": str(self.get_parameter("scenario_id").value),
            "seed": int(self.get_parameter("seed").value),
            "method_mode": str(self.get_parameter("method_mode").value),
            "task_context": dict(self.task),
        }

    def nearest_region(self, x, y):
        best = ("unknown", float("inf"))
        for name, region in self.regions.items():
            distance = math.hypot(x - float(region["x"]), y - float(region["y"]))
            if distance < best[1]:
                best = (name, distance)
        return best[0]

    def relevance(self, detection):
        query = json.dumps(self.task, ensure_ascii=False).lower()
        category = str(detection.get("category", "")).lower()
        tags = [str(tag).lower() for tag in detection.get("task_tags", [])]
        if not query:
            return 0.2
        if category in query:
            return 1.0
        if any(tag in query for tag in tags):
            return 0.8
        return 0.15

    def find_track(self, category, position):
        candidates = []
        for track in self.tracks.values():
            if track["category"] != category or track["state"] == "REMOVED":
                continue
            distance = math.hypot(track["position"]["x"] - position["x"],
                                  track["position"]["y"] - position["y"])
            candidates.append((distance, track))
        if candidates:
            distance, track = min(candidates, key=lambda pair: pair[0])
            if distance <= self.association_radius:
                return track, distance
        return None, None

    def emit(self, event_type, track, **extra):
        now = self.now_sec()
        payload = {"event_type": event_type, "track_id": track["id"],
                   "category": track["category"], "timestamp": now,
                   "confidence": track.get("effective_confidence",
                                           track.get("confidence")),
                   "trial_context": self.trial_context()}
        payload.update(extra)
        self.event_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def on_observation(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        now = self.now_sec()
        for detection in payload.get("detections", []):
            category = str(detection.get("category", "unknown"))
            position = detection.get("position", {})
            if "x" not in position or "y" not in position:
                continue
            observation_confidence = float(detection.get("confidence", 0.0))
            track, distance = self.find_track(category, position)
            relevance = self.relevance(detection)
            novelty = 1.0 if track is None else min(1.0, float(distance or 0.0) /
                                                    max(1e-6, self.association_radius))
            staleness = 0.0 if track is None else min(
                1.0, (now - float(track["last_seen"])) /
                max(1e-6, self.stale_after))
            update_priority = (0.45 * relevance + 0.25 * observation_confidence +
                               0.20 * novelty + 0.10 * staleness)
            if self.task and update_priority < self.update_threshold:
                self.update_stats["task_deferred"] += 1
                continue
            self.update_stats["accepted"] += 1
            if track is None:
                track_id = f"sem_{self.next_id:04d}"
                self.next_id += 1
                track = {
                    "id": track_id, "category": category,
                    "position": dict(position), "confidence": observation_confidence,
                    "first_seen": now, "last_seen": now,
                    "observations": 1, "state": "TENTATIVE",
                    "region": self.nearest_region(position["x"], position["y"]),
                    "task_relevance": relevance,
                    "last_update_priority": round(update_priority, 4),
                    "task_tags": detection.get("task_tags", []),
                    "source": payload.get("source", "unknown"),
                }
                self.tracks[track_id] = track
                self.emit("TRACK_CREATED", track)
                continue
            age = max(0.0, now - float(track["last_seen"]))
            discounted = float(track["confidence"]) * math.exp(
                -math.log(2.0) * age / max(1e-6, self.half_life))
            fused = 1.0 - (1.0 - discounted) * (1.0 - observation_confidence)
            weight_old = max(0.05, discounted)
            weight_new = max(0.05, observation_confidence)
            for axis in ("x", "y", "z"):
                if axis in position:
                    old = float(track["position"].get(axis, position[axis]))
                    track["position"][axis] = (
                        weight_old * old + weight_new * float(position[axis])
                    ) / (weight_old + weight_new)
            previous_state = track["state"]
            track.update({
                "confidence": min(0.999, fused), "last_seen": now,
                "observations": track["observations"] + 1,
                "region": self.nearest_region(position["x"], position["y"]),
                "task_relevance": relevance,
                "last_update_priority": round(update_priority, 4),
            })
            track["state"] = ("CONFIRMED" if track["observations"] >=
                              self.confirm_observations else "TENTATIVE")
            if previous_state in ("STALE", "UNCERTAIN"):
                self.emit("TRACK_REOBSERVED", track, previous_state=previous_state)
            elif track["state"] != previous_state:
                self.emit("TRACK_CONFIRMED", track)

    def tick(self):
        now = self.now_sec()
        visible = []
        stale = 0
        for track in self.tracks.values():
            age = max(0.0, now - float(track["last_seen"]))
            track["age_sec"] = round(age, 3)
            track["effective_confidence"] = round(float(track["confidence"]) *
                math.exp(-math.log(2.0) * age / max(1e-6, self.half_life)), 4)
            previous = track["state"]
            if age >= self.remove_after:
                track["state"] = "REMOVED"
            elif age >= self.stale_after:
                track["state"] = "STALE"
                stale += 1
            elif track["effective_confidence"] < 0.35:
                track["state"] = "UNCERTAIN"
            if track["state"] != previous:
                self.emit("TRACK_STATE_CHANGED", track,
                          previous_state=previous, new_state=track["state"])
            if track["state"] != "REMOVED":
                visible.append(dict(track))
        snapshot = {"revision_time": now, "task": self.task,
                    "update_stats": dict(self.update_stats),
                    "track_count": len(visible), "tracks": visible,
                    "trial_context": self.trial_context()}
        self.snapshot_pub.publish(String(data=json.dumps(snapshot, ensure_ascii=False)))
        denominator = max(1, len(visible))
        self.status_pub.publish(String(data=json.dumps({
            "active_tracks": len(visible), "stale_tracks": stale,
            "stale_ratio": round(stale / denominator, 4),
            "task_relevant_tracks": sum(
                1 for track in visible if track.get("task_relevance", 0) >= 0.5),
            "timestamp": now, "trial_context": self.trial_context(),
        }, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = TemporalSemanticMemory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
