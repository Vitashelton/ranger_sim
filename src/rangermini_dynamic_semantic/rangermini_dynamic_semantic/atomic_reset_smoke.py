"""Run two identical reset barriers and compare deterministic initial state."""
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .reset_contract import canonical_json
from .reset_protocol import RESET_READY_TOPIC, reset_qos


class AtomicResetSmoke(Node):
    def __init__(self):
        super().__init__("atomic_reset_smoke")
        self.declare_parameter(
            "report_path", "/tmp/rangermini_dynamic_benchmark/reset_smoke_report.json")
        self.client = self.create_client(Trigger, "/benchmark/reset_trial")
        self.create_subscription(String, RESET_READY_TOPIC, self.on_ready, reset_qos())
        self.expected_reset_id = None
        self.results = []
        self.request_pending = False
        self.finished = False
        self.create_timer(0.1, self.tick)

    def tick(self):
        if self.finished or self.request_pending or self.expected_reset_id:
            return
        if not self.client.service_is_ready():
            return
        self.request_pending = True
        future = self.client.call_async(Trigger.Request())
        future.add_done_callback(self.on_service_response)

    def on_service_response(self, future):
        self.request_pending = False
        try:
            response = future.result()
        except Exception as exc:
            self.finish(False, f"reset service failed: {exc}")
            return
        if response is None or not response.success:
            self.finish(False, getattr(response, "message", "reset rejected"))
            return
        self.expected_reset_id = response.message

    def on_ready(self, msg):
        try:
            ready = json.loads(msg.data)
        except ValueError:
            return
        if not self.expected_reset_id or ready.get("reset_id") != self.expected_reset_id:
            return
        self.results.append(ready)
        self.expected_reset_id = None
        if len(self.results) == 2:
            self.compare_results()

    @staticmethod
    def signature(ready):
        states = ready.get("ack_states", {})
        scene = states.get("dynamic_scene_manager", {})
        return {
            "manifest_hash": ready.get("manifest_hash"),
            "oracle_event_hash": scene.get("oracle_schedule_hash"),
            "initial_semantic_snapshot_hash": ready.get(
                "ack_state_hashes", {}).get("temporal_semantic_memory"),
            "initial_topology_hash": ready.get(
                "ack_state_hashes", {}).get("dynamic_topology_maintenance"),
            "memory_revision": states.get(
                "temporal_semantic_memory", {}).get("revision"),
            "graph_revision": states.get(
                "dynamic_topology_maintenance", {}).get("revision"),
            "edge_probabilities": [edge.get("blocked_probability") for edge in
                                   states.get("dynamic_topology_maintenance", {}).get(
                                       "edges", [])],
        }

    def compare_results(self):
        first = self.signature(self.results[0])
        second = self.signature(self.results[1])
        required = all(value is not None for key, value in first.items()
                       if key != "edge_probabilities")
        clean = (first["memory_revision"] == 0 and first["graph_revision"] == 0 and
                 all(value == 0.0 for value in first["edge_probabilities"]))
        self.finish(required and clean and first == second,
                    "two reset signatures match" if first == second else
                    "reset signatures differ", first, second)

    def finish(self, passed, message, first=None, second=None):
        self.finished = True
        report = {
            "passed": bool(passed), "message": message,
            "first": first, "second": second,
        }
        path = Path(str(self.get_parameter("report_path").value))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        self.get_logger().info(canonical_json(report))
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = AtomicResetSmoke()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
