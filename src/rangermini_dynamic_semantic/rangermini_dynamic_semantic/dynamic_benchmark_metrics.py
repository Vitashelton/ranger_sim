"""JSONL experiment logger and compact summary for benchmark comparisons."""
import json
import os
import re
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class DynamicBenchmarkMetrics(Node):
    def __init__(self):
        super().__init__("dynamic_benchmark_metrics")
        self.declare_parameter("config_file", "")
        self.declare_parameter("output_dir", "/tmp/rangermini_dynamic_benchmark")
        self.declare_parameter("require_sim_time", True)
        self.declare_parameter("trial_id", "manual_trial")
        self.declare_parameter("scenario_id", "S6")
        self.declare_parameter("seed", 0)
        self.declare_parameter("method_mode", "Ours")
        self.declare_parameter("controller_backend", "pure_pursuit")
        self.declare_parameter("mppi_method", "command_aware")
        self.assert_time_contract()
        output = Path(os.path.expanduser(str(self.get_parameter("output_dir").value)))
        output.mkdir(parents=True, exist_ok=True)
        trial_id = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                          str(self.get_parameter("trial_id").value))
        self.path = output / f"{trial_id}.jsonl"
        self.summary_path = output / f"{trial_id}_summary.json"
        self.path.write_text("", encoding="utf-8")
        self.started = self.now_sec()
        self.task_context = {}
        self.counts = {
            "perception_frames": 0, "scheduler_triggers": 0,
            "navigation_failures": 0, "navigation_successes": 0,
            "memory_events": 0, "ground_truth_events": 0,
            "motion_mode_events": 0,
            "mppi_planner_ticks": 0,
        }
        self.motion_mode_durations = {}
        self.motion_mode_since = None
        self.motion_mode_current = None
        self.last = {}
        topics = {
            "/semantic_observations": "perception",
            "/semantic_perception/stats": "perception_stats",
            "/streaming/scheduler_stats": "scheduler",
            "/semantic_memory_v2/snapshot": "memory",
            "/semantic_memory_v2/events": "memory_event",
            "/dynamic_semantic_graph": "graph",
            "/navigation_failure": "navigation_failure",
            "/navigation_success": "navigation_success",
            "/benchmark/navigation_result": "navigation_result",
            "/grounded_agent/status": "grounded_agent_status",
            "/grounded_agent/answer": "grounded_agent_answer",
            "/grounded_agent/error": "grounded_agent_error",
            "/benchmark/ground_truth/events": "ground_truth_event",
            "/task_context/current": "task_context",
            "/motion_mode_debug": "motion_mode",
            "/benchmark/mppi_trace": "mppi_trace",
        }
        task_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        for topic, kind in topics.items():
            self.create_subscription(
                String, topic, lambda msg, k=kind: self.record(k, msg),
                task_qos if kind == "task_context" else 20)
        self.create_timer(1.0, self.write_summary)
        self.get_logger().info(f"Benchmark metrics: {self.path}")

    def assert_time_contract(self):
        if (bool(self.get_parameter("require_sim_time").value) and
                not bool(self.get_parameter("use_sim_time").value)):
            raise RuntimeError("dynamic_benchmark_metrics requires use_sim_time=true")

    def now_sec(self):
        return self.get_clock().now().nanoseconds * 1.0e-9

    def trial_context(self):
        return {
            "trial_id": str(self.get_parameter("trial_id").value),
            "scenario_id": str(self.get_parameter("scenario_id").value),
            "seed": int(self.get_parameter("seed").value),
            "method_mode": str(self.get_parameter("method_mode").value),
            "controller_backend": str(self.get_parameter("controller_backend").value),
            "mppi_method": str(self.get_parameter("mppi_method").value),
            "task_context": dict(self.task_context),
        }

    def record(self, kind, msg):
        try:
            payload = json.loads(msg.data)
        except ValueError:
            payload = {"raw": msg.data}
        if kind == "task_context":
            self.task_context = payload.get("task_context", {})
        inherited = payload.get("trial_context") if isinstance(payload, dict) else None
        row = {"sim_time": self.now_sec(), "kind": kind,
               "trial_context": inherited or self.trial_context(),
               "payload": payload}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.last[kind] = payload
        if kind == "perception":
            self.counts["perception_frames"] += 1
        elif kind == "scheduler":
            self.counts["scheduler_triggers"] = int(payload.get(
                "trigger_count", self.counts["scheduler_triggers"]))
        elif kind == "navigation_failure":
            self.counts["navigation_failures"] += 1
        elif kind == "navigation_success":
            self.counts["navigation_successes"] += 1
        elif kind == "memory_event":
            self.counts["memory_events"] += 1
        elif kind == "ground_truth_event":
            self.counts["ground_truth_events"] += 1
        elif kind == "mppi_trace":
            self.counts["mppi_planner_ticks"] += 1
        elif kind == "motion_mode":
            self.counts["motion_mode_events"] += 1
            mode = str(payload.get("mode", "UNKNOWN"))
            now = self.now_sec()
            if self.motion_mode_current is not None and self.motion_mode_since is not None:
                elapsed = max(0.0, now - self.motion_mode_since)
                self.motion_mode_durations[self.motion_mode_current] = (
                    self.motion_mode_durations.get(self.motion_mode_current, 0.0) + elapsed)
            self.motion_mode_current = mode
            self.motion_mode_since = now

    def write_summary(self):
        perception = self.last.get("perception_stats", {})
        memory = self.last.get("memory", {})
        graph = self.last.get("graph", {})
        now = self.now_sec()
        mode_durations = dict(self.motion_mode_durations)
        if self.motion_mode_current is not None and self.motion_mode_since is not None:
            mode_durations[self.motion_mode_current] = (
                mode_durations.get(self.motion_mode_current, 0.0) +
                max(0.0, now - self.motion_mode_since))
        summary = {
            "started_at": self.started, "updated_at": now,
            "duration_sec": round(max(0.0, now - self.started), 3),
            "trial_context": self.trial_context(),
            "counts": self.counts,
            "sensor_frames_received": perception.get("rgb_frames_received", 0),
            "semantic_frames_processed": perception.get("frames_processed", 0),
            "semantic_processing_ratio": perception.get("processing_ratio", 0.0),
            "memory_track_count": memory.get("track_count", 0),
            "graph_revision": graph.get("revision", 0),
            "motion_mode_current": self.motion_mode_current,
            "motion_mode_durations_sec": {
                key: round(value, 3) for key, value in mode_durations.items()},
            "mppi_planner_ticks": self.counts["mppi_planner_ticks"],
            "latest_mppi_trace": self.last.get("mppi_trace"),
            "latest_navigation_result": self.last.get("navigation_result"),
            "online_uses_ground_truth": False,
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main(args=None):
    rclpy.init(args=args)
    node = DynamicBenchmarkMetrics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
