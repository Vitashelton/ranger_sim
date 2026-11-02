"""ROS 2 adapter for the existing dynamic-indoor benchmark topic contracts.

Not a Nav2 Action client. Its goal/cancel/result protocol connects to the
included holonomic benchmark_navigation_controller, which simulates terminal
status separately from cancellation acknowledgement.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from .grounded_mission_core import (
    BoundedTaskCompiler, GroundedMissionEngine, MissionError,
    PlaceCatalog, SQLiteMissionLog, Step,
)
from .grounded_llm_bridge import BoundedLLMClient


class GroundedMissionNode(Node):
    def __init__(self):
        super().__init__("grounded_dynamic_mission_agent")
        root = Path(get_package_share_directory("rangermini_dynamic_semantic"))
        self.declare_parameter("places_file", str(root / "config" / "grounded_places_dynamic.json"))
        self.declare_parameter("ledger_file", "/tmp/rangermini_grounded_mission.sqlite3")
        self.declare_parameter("selector_mode", "full")
        self.declare_parameter("llm_enabled", False)
        self.declare_parameter("llm_timeout_sec", 8.0)
        self.declare_parameter("object_view_timeout_sec", 6.0)
        self.declare_parameter("trial_id", "manual_trial")
        self.catalog = PlaceCatalog.load(str(self.get_parameter("places_file").value))
        self.compiler = BoundedTaskCompiler(self.catalog)
        self.store = SQLiteMissionLog(str(self.get_parameter("ledger_file").value))
        self.engine = GroundedMissionEngine(
            self.catalog, self.store, selector_mode=str(self.get_parameter("selector_mode").value))
        self.llm_enabled = bool(self.get_parameter("llm_enabled").value)
        self.goal_pub = self.create_publisher(String, "/benchmark/goal", 10)
        self.cancel_pub = self.create_publisher(String, "/benchmark/cancel_goal", 10)
        self.task_pub = self.create_publisher(String, "/task_context", 10)
        self.trigger_pub = self.create_publisher(String, "/semantic_perception/trigger", 10)
        self.status_pub = self.create_publisher(String, "/grounded_agent/status", 10)
        self.answer_pub = self.create_publisher(String, "/grounded_agent/answer", 10)
        self.error_pub = self.create_publisher(String, "/grounded_agent/error", 10)
        self.create_subscription(String, "/grounded_agent/mission_json", self.on_mission_json, 10)
        self.create_subscription(String, "/grounded_agent/mission_text", self.on_mission_text, 10)
        self.create_subscription(String, "/grounded_agent/revise_json", self.on_revise_json, 10)
        self.create_subscription(String, "/grounded_agent/revise_text", self.on_revise_text, 10)
        self.create_subscription(String, "/grounded_agent/human_ack", self.on_human_ack, 10)
        self.create_subscription(String, "/grounded_agent/query", self.on_query, 10)
        self.create_subscription(String, "/benchmark/navigation_result", self.on_nav_result, 20)
        self.create_subscription(String, "/benchmark/cancel_ack", self.on_cancel_ack, 10)
        self.create_subscription(String, "/semantic_observations", self.on_observation, 20)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, "/dynamic_semantic_graph", self.on_graph, latched)
        self.create_timer(1.0, self.on_timer)

    def now_sec(self):
        return self.get_clock().now().nanoseconds / 1e9

    def _parse(self, msg):
        obj = json.loads(msg.data)
        if not isinstance(obj, dict):
            raise MissionError("message must contain a JSON object")
        return obj

    def report_error(self, label: str, exc: Exception):
        self.error_pub.publish(String(data=json.dumps({"stage": label, "error": str(exc)},
                                                     ensure_ascii=False)))
        self.get_logger().error(f"{label}: {exc}")

    def perform(self):
        for effect in self.engine.take_effects():
            action = effect.pop("action")
            if action == "SEND_GOAL":
                self.goal_pub.publish(String(data=json.dumps(effect, ensure_ascii=False)))
            elif action == "CANCEL_GOAL":
                self.cancel_pub.publish(String(data=json.dumps(effect, ensure_ascii=False)))
            elif action == "SET_TASK_CONTEXT":
                category = effect.get("category", "")
                goal = effect["target_region"]
                task = {"task_id": self.engine.mission_id,
                        "task_type": "retrieval" if category else "navigation",
                        "target": category or goal, "target_region": goal,
                        "priority_objects": [category] if category else [],
                        "failure_policy": "reroute"}
                self.task_pub.publish(String(data=json.dumps(task, ensure_ascii=False)))
            elif action == "REQUEST_PERCEPTION":
                self.trigger_pub.publish(String(data=json.dumps({
                    "reason": "grounded_object_verification", "task": effect,
                    "timestamp": self.now_sec()}, ensure_ascii=False)))
            elif action == "MISSION_FINISHED":
                self.get_logger().info("Grounded mission completed")
        self.publish_status()

    def on_timer(self):
        step = self.engine.current()
        if (step is not None and step.op == "SEARCH_OBJECT" and
                step.status == "WAITING_SENSOR_EVIDENCE" and
                self.now_sec() - self.engine.last_arrival_stamp >=
                float(self.get_parameter("object_view_timeout_sec").value)):
            try:
                self.engine.continue_search(now=self.now_sec())
                self.perform()
            except MissionError as exc:
                self.report_error("object_timeout", exc)
        self.publish_status()

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.engine.status(), ensure_ascii=False)))

    def _start_plan(self, plan, mission_id=None):
        self.engine.start(plan, mission_id=mission_id or "m_" + uuid.uuid4().hex[:14])
        self.perform()

    def on_mission_json(self, msg):
        try:
            obj = self._parse(msg)
            steps = self.compiler.compile(obj)
            self._start_plan(steps, obj.get("mission_id"))
        except (MissionError, ValueError) as exc:
            self.report_error("mission_json", exc)

    def on_mission_text(self, msg):
        try:
            if self.llm_enabled:
                if self.engine.active_goal:
                    raise MissionError("LLM blocking request not permitted during motion")
                response = BoundedLLMClient(
                    timeout=float(self.get_parameter("llm_timeout_sec").value)
                ).compile(msg.data, self.catalog, graph=self.engine.last_graph)
                steps = self.compiler.compile(response)
            else:
                steps = self.compiler.parse_short_visit(msg.data)
            self._start_plan(steps)
        except (MissionError, ValueError, OSError, KeyError) as exc:
            self.report_error("mission_text", exc)

    def _revise(self, steps):
        self.engine.revise_remaining(steps, now=self.now_sec())
        self.perform()

    def on_revise_json(self, msg):
        try:
            steps = self.compiler.compile(self._parse(msg))
            self._revise(steps)
        except (MissionError, ValueError) as exc:
            self.report_error("revise_json", exc)

    def on_revise_text(self, msg):
        try:
            self._revise(self.compiler.parse_short_visit(msg.data))
        except (MissionError, ValueError) as exc:
            self.report_error("revise_text", exc)

    def on_graph(self, msg):
        try:
            self.engine.update_graph(self._parse(msg))
            self.perform()
        except (MissionError, ValueError) as exc:
            self.report_error("graph", exc)

    def on_nav_result(self, msg):
        try:
            obj = self._parse(msg)
            if not obj.get("goal_id"):
                return  # Legacy/manual controller result is not part of our mission.
            self.engine.navigation_result(goal_id=str(obj["goal_id"]),
                                          status=str(obj.get("status", "")),
                                          stamp=float(obj.get("timestamp", self.now_sec())))
            self.perform()
        except (MissionError, ValueError) as exc:
            self.report_error("navigation_result", exc)

    def on_cancel_ack(self, msg):
        try:
            obj = self._parse(msg)
            self.engine.cancel_ack(goal_id=str(obj["goal_id"]),
                                   accepted=obj.get("accepted") is True)
            self.perform()
        except (MissionError, ValueError, KeyError) as exc:
            self.report_error("cancel_ack", exc)

    def on_observation(self, msg):
        try:
            self.engine.observe(self._parse(msg), now=self.now_sec())
            self.perform()
        except (MissionError, ValueError) as exc:
            self.report_error("sensor", exc)

    def on_human_ack(self, msg):
        try:
            obj = self._parse(msg)
            self.engine.human_ack(action=str(obj["action"]),
                                  operator_id=str(obj["operator_id"]),
                                  note=str(obj["note"]), now=self.now_sec())
            self.perform()
        except (MissionError, ValueError, KeyError) as exc:
            self.report_error("human_ack", exc)

    def on_query(self, _msg):
        self.answer_pub.publish(String(data=json.dumps(
            self.engine.answers_from_events(), ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = GroundedMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.store.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
