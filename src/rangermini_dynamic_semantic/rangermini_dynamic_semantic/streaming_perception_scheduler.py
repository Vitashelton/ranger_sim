"""Task- and failure-aware scheduler for streaming semantic perception."""
import json
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


class StreamingPerceptionScheduler(Node):
    def __init__(self):
        super().__init__("streaming_perception_scheduler")
        default = str(Path(get_package_share_directory(
            "rangermini_dynamic_semantic")) / "config" / "dynamic_benchmark.yaml")
        self.declare_parameter("config_file", default)
        self.declare_parameter("require_sim_time", True)
        self.declare_parameter("trial_id", "manual_trial")
        self.declare_parameter("scenario_id", "S6")
        self.declare_parameter("seed", 0)
        self.declare_parameter("method_mode", "Ours")
        self.declare_parameter("task_context_file", "")
        self.declare_parameter("task_context_json", "")
        self.assert_time_contract()
        self.assert_experiment_context()
        with open(str(self.get_parameter("config_file").value), encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg.get("scheduler", {})
        self.idle_rate = float(self.cfg.get("idle_rate_hz", 0.5))
        self.task_rate = float(self.cfg.get("task_rate_hz", 2.0))
        self.burst_rate = float(self.cfg.get("burst_rate_hz", 8.0))
        self.burst_duration = float(self.cfg.get("burst_duration_sec", 4.0))
        self.trigger_pub = self.create_publisher(String, "/semantic_perception/trigger", 10)
        self.stats_pub = self.create_publisher(String, "/streaming/scheduler_stats", 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.task_pub = self.create_publisher(
            String, "/task_context/current", latched)
        self.create_subscription(String, "/task_context", self.on_task_input, 10)
        self.create_subscription(String, "/task_goal", self.on_task_goal, 10)
        self.create_subscription(String, "/navigation_failure", self.on_failure, 20)
        self.create_subscription(String, "/streaming/memory_status", self.on_memory, 10)
        self.task = self.load_initial_task()
        self.failure = None
        self.stale_ratio = 0.0
        self.burst_until = 0.0
        self.last_trigger = 0.0
        self.trigger_count = 0
        self.mode_counts = {"idle": 0, "task": 0, "burst": 0}
        self.started = self.now_sec()
        self.create_timer(0.025, self.tick)
        self.create_timer(1.0, self.publish_stats)
        self.publish_task_context()

    def assert_time_contract(self):
        if (bool(self.get_parameter("require_sim_time").value) and
                not bool(self.get_parameter("use_sim_time").value)):
            raise RuntimeError("streaming_perception_scheduler requires use_sim_time=true")

    def assert_experiment_context(self):
        if not str(self.get_parameter("trial_id").value).strip():
            raise ValueError("trial_id must not be empty")
        allowed = {"B0", "B1", "B2", "B3", "Ours"}
        method = str(self.get_parameter("method_mode").value)
        if method not in allowed:
            raise ValueError(f"method_mode must be one of {sorted(allowed)}")

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

    @staticmethod
    def validate_task(payload):
        required = ("task_id", "task_type", "target", "target_region",
                    "priority_objects", "failure_policy")
        if not isinstance(payload, dict):
            raise ValueError("TaskContext must be a JSON/YAML object")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"TaskContext missing fields: {missing}")
        if not isinstance(payload["priority_objects"], list):
            raise ValueError("TaskContext priority_objects must be a list")
        return {key: payload[key] for key in required}

    def load_initial_task(self):
        inline = str(self.get_parameter("task_context_json").value).strip()
        task_file = str(self.get_parameter("task_context_file").value).strip()
        if inline:
            return self.validate_task(yaml.safe_load(inline))
        if task_file:
            with open(task_file, encoding="utf-8") as stream:
                return self.validate_task(yaml.safe_load(stream))
        raise ValueError("TaskContext requires task_context_json or task_context_file")

    def publish_task_context(self):
        self.task_pub.publish(String(data=json.dumps({
            "timestamp": self.now_sec(), "task_context": self.task,
            "trial_context": self.trial_context(),
        }, ensure_ascii=False)))

    def on_task_input(self, msg):
        try:
            self.task = self.validate_task(yaml.safe_load(msg.data))
        except (ValueError, yaml.YAMLError) as exc:
            self.get_logger().error(f"Rejected TaskContext: {exc}")
            return
        self.publish_task_context()

    def on_task_goal(self, msg):
        target_region = msg.data.strip()
        if not target_region:
            return
        self.task = dict(self.task)
        self.task["target_region"] = target_region
        self.publish_task_context()

    def on_failure(self, msg):
        try:
            self.failure = json.loads(msg.data)
        except ValueError:
            self.failure = {"event_type": msg.data}
        self.burst_until = self.now_sec() + self.burst_duration

    def on_memory(self, msg):
        try:
            self.stale_ratio = float(json.loads(msg.data).get("stale_ratio", 0.0))
        except (ValueError, TypeError):
            return
        if self.stale_ratio > 0.4 and self.task:
            self.burst_until = max(self.burst_until, self.now_sec() + 1.5)

    def mode_and_rate(self, now):
        if now < self.burst_until:
            return "burst", self.burst_rate
        if self.task and self.task.get("active", True):
            return "task", self.task_rate
        return "idle", self.idle_rate

    def tick(self):
        now = self.now_sec()
        mode, rate = self.mode_and_rate(now)
        if rate <= 0.0 or now - self.last_trigger < 1.0 / rate:
            return
        self.last_trigger = now
        self.trigger_count += 1
        self.mode_counts[mode] += 1
        reason = "periodic_idle"
        if mode == "task":
            reason = "active_task"
        elif mode == "burst":
            reason = "failure_or_stale_memory"
        self.trigger_pub.publish(String(data=json.dumps({
            "sequence": self.trigger_count, "mode": mode, "reason": reason,
            "task": self.task, "failure": self.failure if mode == "burst" else None,
            "stale_ratio": round(self.stale_ratio, 4), "timestamp": now,
            "trial_context": self.trial_context(),
        }, ensure_ascii=False)))

    def publish_stats(self):
        now = self.now_sec()
        elapsed = max(1e-6, now - self.started)
        mode, target_rate = self.mode_and_rate(now)
        self.stats_pub.publish(String(data=json.dumps({
            "mode": mode, "target_rate_hz": target_rate,
            "trigger_count": self.trigger_count,
            "average_rate_hz": round(self.trigger_count / elapsed, 3),
            "mode_counts": self.mode_counts, "stale_ratio": self.stale_ratio,
            "timestamp": now, "trial_context": self.trial_context(),
        }, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = StreamingPerceptionScheduler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
