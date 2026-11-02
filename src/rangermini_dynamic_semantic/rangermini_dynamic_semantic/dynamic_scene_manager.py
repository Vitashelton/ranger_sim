"""Reproducible disturbance generator for the dynamic indoor benchmark.

This node is an experiment fixture, not a perception source.  It moves Gazebo
entities according to YAML and publishes a separate ground-truth event stream.
Online perception and memory nodes never subscribe to that stream.
"""
import json
import math
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


class DynamicSceneManager(Node):
    def __init__(self):
        super().__init__("dynamic_scene_manager")
        default = str(Path(get_package_share_directory(
            "rangermini_dynamic_semantic")) / "config" / "dynamic_benchmark.yaml")
        self.declare_parameter("config_file", default)
        self.declare_parameter("scenario", "mixed_shift")
        self.declare_parameter("require_sim_time", True)
        self.declare_parameter("trial_id", "manual_trial")
        self.declare_parameter("scenario_id", "S6")
        self.declare_parameter("seed", 0)
        self.declare_parameter("method_mode", "Ours")
        self.declare_parameter("task_context_file", "")
        self.declare_parameter("task_context_json", "")
        self.assert_time_contract()
        self.declare_parameter(
            "set_pose_service", "/world/dynamic_indoor_benchmark/set_pose")
        with open(str(self.get_parameter("config_file").value), encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        scenario = str(self.get_parameter("scenario").value)
        self.events = list(self.cfg.get("scenarios", {}).get(scenario, {}).get("events", []))
        self.initial_entities = dict(self.cfg.get("initial_entities", {}))
        self.task_context = self.load_task_context()
        self.client = self.create_client(
            SetEntityPose, str(self.get_parameter("set_pose_service").value))
        self.event_pub = self.create_publisher(String, "/benchmark/ground_truth/events", 20)
        self.state_pub = self.create_publisher(String, "/benchmark/ground_truth/state", 10)
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.create_subscription(String, "/benchmark/reset", self.on_reset, 10)
        self.sim_time = None
        self.origin = None
        self.fired = set()
        self.entity_state = {}
        self.create_timer(0.1, self.tick)

    def assert_time_contract(self):
        if (bool(self.get_parameter("require_sim_time").value) and
                not bool(self.get_parameter("use_sim_time").value)):
            raise RuntimeError("dynamic_scene_manager requires use_sim_time=true")

    def load_task_context(self):
        inline = str(self.get_parameter("task_context_json").value).strip()
        task_file = str(self.get_parameter("task_context_file").value).strip()
        if inline:
            payload = yaml.safe_load(inline)
        elif task_file:
            with open(task_file, encoding="utf-8") as stream:
                payload = yaml.safe_load(stream)
        else:
            payload = {}
        return payload if isinstance(payload, dict) else {}

    def trial_context(self):
        return {
            "trial_id": str(self.get_parameter("trial_id").value),
            "scenario_id": str(self.get_parameter("scenario_id").value),
            "seed": int(self.get_parameter("seed").value),
            "method_mode": str(self.get_parameter("method_mode").value),
            "task_context": dict(self.task_context),
        }

    def on_clock(self, msg):
        self.sim_time = float(msg.clock.sec) + 1e-9 * float(msg.clock.nanosec)
        if self.origin is None:
            self.origin = self.sim_time

    def on_reset(self, _msg):
        self.origin = self.sim_time
        self.fired.clear()
        self.entity_state.clear()
        for entity, pose in self.initial_entities.items():
            self.request_pose(entity, pose)
        self.get_logger().info("Dynamic benchmark event clock reset")

    def request_pose(self, entity, pose):
        if len(pose) < 4 or not self.client.service_is_ready():
            return False
        request = SetEntityPose.Request()
        request.entity = Entity(name=str(entity), type=Entity.MODEL)
        request.pose = Pose()
        request.pose.position.x = float(pose[0])
        request.pose.position.y = float(pose[1])
        request.pose.position.z = float(pose[2])
        yaw = float(pose[3])
        request.pose.orientation.z = math.sin(yaw / 2.0)
        request.pose.orientation.w = math.cos(yaw / 2.0)
        self.client.call_async(request)
        return True

    def elapsed(self):
        if self.sim_time is not None and self.origin is not None:
            return max(0.0, self.sim_time - self.origin)
        return 0.0

    def apply_event(self, index, event):
        pose = event.get("pose", [])
        if not self.request_pose(event["entity"], pose):
            return False
        payload = dict(event)
        payload.update({"event_index": index, "benchmark_time_sec": self.elapsed(),
                        "timestamp": self.sim_time or 0.0,
                        "source": "evaluation_fixture", "online_visible": False,
                        "trial_context": self.trial_context()})
        self.entity_state[str(event["entity"])] = payload
        self.event_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.get_logger().info(
            f"Injected {event.get('event')} for {event.get('entity')}")
        return True

    def tick(self):
        elapsed = self.elapsed()
        for index, event in enumerate(self.events):
            if index not in self.fired and elapsed >= float(event.get("at_sec", 0.0)):
                if self.apply_event(index, event):
                    self.fired.add(index)
        self.state_pub.publish(String(data=json.dumps({
            "benchmark_time_sec": round(elapsed, 3),
            "events_fired": sorted(self.fired),
            "entity_state": self.entity_state,
            "warning": "evaluation ground truth; forbidden to online policy",
            "timestamp": self.sim_time or 0.0,
            "trial_context": self.trial_context(),
        }, ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = DynamicSceneManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
