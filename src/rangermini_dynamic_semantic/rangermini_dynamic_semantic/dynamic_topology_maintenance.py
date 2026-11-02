"""Maintain task-queryable topology from navigation outcomes and semantic memory."""
import json
import math
from copy import deepcopy
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class DynamicTopologyMaintenance(Node):
    def __init__(self):
        super().__init__("dynamic_topology_maintenance")
        default = str(Path(get_package_share_directory(
            "rangermini_dynamic_semantic")) / "config" / "dynamic_benchmark.yaml")
        self.declare_parameter("config_file", default)
        self.declare_parameter("block_half_life_sec", 24.0)
        self.declare_parameter("require_sim_time", True)
        self.declare_parameter("trial_id", "manual_trial")
        self.declare_parameter("scenario_id", "S6")
        self.declare_parameter("seed", 0)
        self.declare_parameter("method_mode", "Ours")
        self.assert_time_contract()
        with open(str(self.get_parameter("config_file").value), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.nodes = {
            name: {"id": name, "kind": "place", "region_type": spec.get("type"),
                   "pose": {"x": spec["x"], "y": spec["y"]}, "active": True}
            for name, spec in cfg.get("regions", {}).items()
        }
        self.edges = {}
        for index, edge in enumerate(cfg.get("topology_edges", [])):
            key = self.edge_key(edge["from"], edge["to"])
            self.edges[key] = {
                "id": f"edge_{index:02d}", "from": edge["from"], "to": edge["to"],
                "base_cost": float(edge.get("cost", 1.0)),
                "route": edge.get("route", ""), "state": "FREE",
                "blocked_probability": 0.0, "last_update": self.now_sec(),
                "last_event": "INITIALIZED", "traversal_successes": 0,
                "traversal_failures": 0,
            }
        self.object_nodes = {}
        self.task = {}
        self.revision = 1
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.graph_pub = self.create_publisher(String, "/dynamic_semantic_graph", latched)
        self.task_pub = self.create_publisher(String, "/dynamic_semantic_graph/task_view", latched)
        self.event_pub = self.create_publisher(String, "/dynamic_semantic_graph/events", 20)
        self.create_subscription(String, "/semantic_memory_v2/snapshot", self.on_memory, 10)
        self.create_subscription(String, "/navigation_failure", self.on_failure, 20)
        self.create_subscription(String, "/navigation_success", self.on_success, 20)
        self.create_subscription(String, "/task_context/current", self.on_task, latched)
        self.create_timer(0.5, self.tick)

    @staticmethod
    def edge_key(a, b):
        return "|".join(sorted((str(a), str(b))))

    def assert_time_contract(self):
        if (bool(self.get_parameter("require_sim_time").value) and
                not bool(self.get_parameter("use_sim_time").value)):
            raise RuntimeError("dynamic_topology_maintenance requires use_sim_time=true")

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

    def on_task(self, msg):
        try:
            payload = json.loads(msg.data)
            self.task = payload.get("task_context", payload)
        except ValueError:
            self.task = {"query": msg.data}
        self.revision += 1

    def on_memory(self, msg):
        try:
            snapshot = json.loads(msg.data)
        except ValueError:
            return
        changed = False
        incoming = set()
        for track in snapshot.get("tracks", []):
            track_id = str(track.get("id", ""))
            if not track_id:
                continue
            incoming.add(track_id)
            node = {
                "id": track_id, "kind": "semantic_object",
                "category": track.get("category"), "position": track.get("position"),
                "region": track.get("region"), "confidence": track.get("effective_confidence"),
                "state": track.get("state"), "last_seen": track.get("last_seen"),
                "task_relevance": track.get("task_relevance", 0.0),
            }
            old = self.object_nodes.get(track_id)
            if self.material_object_change(old, node):
                changed = True
            self.object_nodes[track_id] = node
        for track_id in set(self.object_nodes) - incoming:
            del self.object_nodes[track_id]
            changed = True
        if changed:
            self.revision += 1

    @staticmethod
    def material_object_change(old, new):
        if old is None:
            return True
        for key in ("category", "region", "state"):
            if old.get(key) != new.get(key):
                return True
        old_position = old.get("position") or {}
        new_position = new.get("position") or {}
        if math.hypot(float(old_position.get("x", 0.0)) -
                      float(new_position.get("x", 0.0)),
                      float(old_position.get("y", 0.0)) -
                      float(new_position.get("y", 0.0))) > 0.15:
            return True
        return abs(float(old.get("confidence") or 0.0) -
                   float(new.get("confidence") or 0.0)) > 0.05

    def resolve_edge(self, payload):
        if payload.get("edge"):
            raw = str(payload["edge"]).replace("-", "|")
            parts = raw.split("|")
            if len(parts) == 2:
                return self.edges.get(self.edge_key(parts[0], parts[1]))
        if payload.get("from") and payload.get("to"):
            return self.edges.get(self.edge_key(payload["from"], payload["to"]))
        return None

    def publish_event(self, event_type, edge, payload):
        self.event_pub.publish(String(data=json.dumps({
            "event_type": event_type, "revision": self.revision,
            "edge": deepcopy(edge), "evidence": payload,
            "timestamp": self.now_sec(), "trial_context": self.trial_context(),
        }, ensure_ascii=False)))

    def on_failure(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        edge = self.resolve_edge(payload)
        if edge is None:
            return
        confidence = max(0.1, min(1.0, float(payload.get("confidence", 0.75))))
        edge["blocked_probability"] = 1.0 - (
            1.0 - edge["blocked_probability"]) * (1.0 - confidence)
        edge["last_update"] = self.now_sec()
        edge["last_event"] = str(payload.get("event_type", "NAVIGATION_FAILURE"))
        edge["traversal_failures"] += 1
        self.update_edge_state(edge)
        self.revision += 1
        self.publish_event("EDGE_EVIDENCE_FAILURE", edge, payload)

    def on_success(self, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        edge = self.resolve_edge(payload)
        if edge is None:
            return
        edge["blocked_probability"] *= 0.15
        edge["last_update"] = self.now_sec()
        edge["last_event"] = "TRAVERSAL_SUCCESS"
        edge["traversal_successes"] += 1
        self.update_edge_state(edge)
        self.revision += 1
        self.publish_event("EDGE_EVIDENCE_SUCCESS", edge, payload)

    @staticmethod
    def update_edge_state(edge):
        probability = float(edge["blocked_probability"])
        edge["state"] = ("TEMP_BLOCKED" if probability >= 0.65 else
                         "SUSPECTED" if probability >= 0.25 else "FREE")
        edge["effective_cost"] = round(float(edge["base_cost"]) *
                                       (1.0 + 8.0 * probability), 4)

    def task_view(self):
        query = json.dumps(self.task, ensure_ascii=False).lower()
        objects = []
        relevant_regions = set()
        for node in self.object_nodes.values():
            if (float(node.get("task_relevance", 0.0)) >= 0.5 or
                    str(node.get("category", "")).lower() in query):
                objects.append(node)
                if node.get("region"):
                    relevant_regions.add(node["region"])
        return {"revision": self.revision, "task": self.task,
                "relevant_objects": objects,
                "relevant_regions": sorted(relevant_regions),
                "non_free_edges": [deepcopy(edge) for edge in self.edges.values()
                                   if edge["state"] != "FREE"],
                "timestamp": self.now_sec(),
                "trial_context": self.trial_context()}

    def tick(self):
        now = self.now_sec()
        half_life = float(self.get_parameter("block_half_life_sec").value)
        for edge in self.edges.values():
            previous_state = edge["state"]
            dt = max(0.0, now - float(edge["last_update"]))
            if edge["blocked_probability"] > 0.0 and dt > 0.45:
                edge["blocked_probability"] *= math.exp(-math.log(2.0) * dt / half_life)
                edge["last_update"] = now
            self.update_edge_state(edge)
            if edge["state"] != previous_state:
                self.revision += 1
                self.publish_event("EDGE_STATE_DECAY_TRANSITION", edge, {
                    "previous_state": previous_state,
                    "new_state": edge["state"], "source": "temporal_decay"})
        graph = {"revision": self.revision, "timestamp": now,
                 "place_nodes": list(self.nodes.values()),
                 "semantic_nodes": list(self.object_nodes.values()),
                 "edges": list(self.edges.values()),
                 "trial_context": self.trial_context()}
        self.graph_pub.publish(String(data=json.dumps(graph, ensure_ascii=False)))
        self.task_pub.publish(String(data=json.dumps(self.task_view(), ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = DynamicTopologyMaintenance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
