"""Barrier coordinator for atomic benchmark-trial reset."""
import json
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .reset_protocol import (RESET_ACK_TOPIC, RESET_READY_TOPIC,
                             RESET_REQUEST_TOPIC, canonical_hash,
                             canonical_json, reset_qos)


DEFAULT_PARTICIPANTS = [
    "dynamic_scene_manager",
    "streaming_perception_scheduler",
    "rgbd_semantic_perception",
    "temporal_semantic_memory",
    "dynamic_topology_maintenance",
    "benchmark_navigation_controller",
    "dynamic_benchmark_metrics",
]


class TrialResetCoordinator(Node):
    def __init__(self):
        super().__init__("trial_reset_coordinator")
        self.declare_parameter("require_sim_time", True)
        self.declare_parameter("trial_id", "manual_trial")
        self.declare_parameter("scenario_id", "S6")
        self.declare_parameter("seed", 0)
        self.declare_parameter("method_mode", "Ours")
        default_task = str(Path(get_package_share_directory(
            "rangermini_dynamic_semantic")) / "config" / "task_context.yaml")
        self.declare_parameter("task_context_file", default_task)
        self.declare_parameter("task_context_json", "")
        self.declare_parameter("expected_reset_nodes", ",".join(DEFAULT_PARTICIPANTS))
        self.declare_parameter("auto_reset_on_start", True)
        self.declare_parameter("startup_delay_sec", 1.0)
        self.assert_time_contract()
        self.task_context_value = self.load_task_context()
        self.expected = self.parse_expected_nodes()
        self.epoch = 0
        self.active_request = None
        self.acks = {}
        self.started = self.now_sec()
        self.auto_requested = False
        self.request_pub = self.create_publisher(
            String, RESET_REQUEST_TOPIC, reset_qos())
        self.ready_pub = self.create_publisher(
            String, RESET_READY_TOPIC, reset_qos())
        self.manifest_pub = self.create_publisher(
            String, "/benchmark/trial_manifest", reset_qos())
        self.status_pub = self.create_publisher(
            String, "/benchmark/reset/status", reset_qos())
        self.create_subscription(String, RESET_ACK_TOPIC, self.on_ack, 50)
        self.create_service(Trigger, "/benchmark/reset_trial", self.on_trigger)
        self.create_timer(0.1, self.tick)

    def assert_time_contract(self):
        if (bool(self.get_parameter("require_sim_time").value) and
                not bool(self.get_parameter("use_sim_time").value)):
            raise RuntimeError("trial_reset_coordinator requires use_sim_time=true")

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1.0e-9

    def parse_expected_nodes(self):
        raw = str(self.get_parameter("expected_reset_nodes").value)
        nodes = [item.strip() for item in raw.split(",") if item.strip()]
        if len(nodes) != len(set(nodes)) or not nodes:
            raise ValueError("expected_reset_nodes must be a non-empty unique list")
        return nodes

    def load_task_context(self):
        inline = str(self.get_parameter("task_context_json").value).strip()
        task_file = str(self.get_parameter("task_context_file").value).strip()
        if inline:
            task_context = yaml.safe_load(inline)
        elif task_file:
            with open(task_file, encoding="utf-8") as stream:
                task_context = yaml.safe_load(stream)
        else:
            task_context = {}
        if not isinstance(task_context, dict):
            raise ValueError("task context must be a YAML/JSON object")
        return task_context

    def trial_context(self):
        return {
            "trial_id": str(self.get_parameter("trial_id").value),
            "scenario_id": str(self.get_parameter("scenario_id").value),
            "seed": int(self.get_parameter("seed").value),
            "method_mode": str(self.get_parameter("method_mode").value),
            "task_context": dict(self.task_context_value),
        }

    def build_manifest(self):
        return {
            "contract_version": "v0.20.0+ChangeSetB",
            "trial_context": self.trial_context(),
            "expected_reset_nodes": sorted(self.expected),
        }

    def request_reset(self):
        self.epoch += 1
        manifest = self.build_manifest()
        self.active_request = {
            "reset_id": f"{self.trial_context()['trial_id']}:{self.epoch:04d}",
            "reset_epoch": self.epoch,
            "requested_at": self.now_sec(),
            "trial_context": self.trial_context(),
            "manifest": manifest,
            "manifest_hash": canonical_hash(manifest),
        }
        self.acks = {}
        self.request_pub.publish(String(data=canonical_json(self.active_request)))
        self.publish_status("RESETTING")

    def on_trigger(self, _request, response):
        if self.active_request is not None and len(self.acks) < len(self.expected):
            response.success = False
            response.message = "reset already in progress"
            return response
        self.request_reset()
        response.success = True
        response.message = self.active_request["reset_id"]
        return response

    def on_ack(self, msg):
        if self.active_request is None:
            return
        try:
            ack = json.loads(msg.data)
        except ValueError:
            return
        if ack.get("reset_id") != self.active_request["reset_id"]:
            return
        node = str(ack.get("node", ""))
        if node not in self.expected or ack.get("status") != "READY":
            return
        self.acks[node] = ack
        if set(self.acks) == set(self.expected):
            self.finish_reset()
        else:
            self.publish_status("RESETTING")

    def finish_reset(self):
        ack_hashes = {node: self.acks[node]["state_hash"]
                      for node in sorted(self.acks)}
        ack_states = {node: self.acks[node]["state"]
                      for node in sorted(self.acks)}
        ready = {
            "reset_id": self.active_request["reset_id"],
            "reset_epoch": self.epoch,
            "status": "READY",
            "ready_at": self.now_sec(),
            "trial_context": self.trial_context(),
            "manifest_hash": self.active_request["manifest_hash"],
            "ack_state_hashes": ack_hashes,
            "ack_states": ack_states,
            "reset_state_hash": canonical_hash(ack_hashes),
        }
        self.ready_pub.publish(String(data=canonical_json(ready)))
        self.manifest_pub.publish(String(data=canonical_json({
            **self.active_request["manifest"],
            "manifest_hash": self.active_request["manifest_hash"],
            "reset_id": ready["reset_id"],
            "reset_epoch": self.epoch,
            "ack_state_hashes": ack_hashes,
            "reset_state_hash": ready["reset_state_hash"],
        })))
        self.publish_status("READY")

    def publish_status(self, status):
        reset_id = (self.active_request or {}).get("reset_id")
        missing = sorted(set(self.expected) - set(self.acks))
        self.status_pub.publish(String(data=canonical_json({
            "status": status, "reset_id": reset_id,
            "reset_epoch": self.epoch, "acknowledged": sorted(self.acks),
            "missing": missing, "timestamp": self.now_sec(),
            "trial_context": self.trial_context(),
        })))

    def tick(self):
        if (not self.auto_requested and
                bool(self.get_parameter("auto_reset_on_start").value) and
                self.now_sec() > 0.0 and
                self.now_sec() - self.started >=
                float(self.get_parameter("startup_delay_sec").value)):
            self.auto_requested = True
            self.request_reset()
        elif self.active_request is not None and len(self.acks) < len(self.expected):
            self.publish_status("RESETTING")


def main(args=None):
    rclpy.init(args=args)
    node = TrialResetCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
