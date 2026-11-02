"""Information-limited deterministic search/patrol state machine.

Allowed information: MissionSpec, YAML schedule priors, public observation
points, Ranger odometry, requested perception events, negative observations and
last-seen records. This module intentionally has no Gazebo or /office_rpg/sim/*
subscriptions.
"""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from .graph_manager import GraphManager
from .llm_providers import (OfflineSemanticProvider,
                            SEMANTIC_PLANNER_SYSTEM_PROMPT,
                            semantic_provider_for)
from .plan_gate import PlanGate


TERMINAL_STATES = {"COMPLETED", "FAILED", "STOPPED"}
PEOPLE = {
    "teacher_zhang": "张老师", "student_li": "李同学", "visitor": "访客"
}


class OfficeSearchExecutor(Node):
    def __init__(self):
        super().__init__("office_rpg_search_executor")
        default_cfg = str(Path(get_package_share_directory(
            "rangermini_dynamic_semantic")) / "config" / "npc_schedules.yaml")
        self.declare_parameter("schedule_file", default_cfg)
        self.declare_parameter("initial_time", -1.0)
        self.declare_parameter("navigation_timeout_sec", 120.0)
        self.declare_parameter("observation_timeout_sec", 5.0)
        self.declare_parameter("planner_test_response", "")
        with open(str(self.get_parameter("schedule_file").value), encoding="utf-8") as stream:
            self.cfg = yaml.safe_load(stream)
        initial_override = float(self.get_parameter("initial_time").value)
        self.initial_time = (float(self.cfg.get("initial_time", 0.0))
                             if initial_override < 0.0 else initial_override)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, "/office_rpg/mission_status", latched)
        self.report_pub = self.create_publisher(String, "/office_rpg/patrol_report", latched)
        self.graph_pub = self.create_publisher(
            String, "/office_rpg/topology_graph", latched)
        self.llm_debug_pub = self.create_publisher(
            String, "/office_rpg/llm_planner_debug", latched)
        self.gate_pub = self.create_publisher(
            String, "/office_rpg/plan_gate_result", latched)
        self.event_pub = self.create_publisher(String, "/office_rpg/event_log", 50)
        self.reminder_pub = self.create_publisher(String, "/office_rpg/reminder_event", 10)
        self.goal_pub = self.create_publisher(String, "/task_goal", 10)
        self.task_control_pub = self.create_publisher(String, "/task_control", 10)
        self.perception_pub = self.create_publisher(
            String, "/office_rpg/perception_trigger", 10)
        self.create_subscription(String, "/office_rpg/mission_spec", self.on_spec, 10)
        self.create_subscription(String, "/office_rpg/executor_control", self.on_control, 10)
        self.create_subscription(String, "/office_rpg/person_detections",
                                 self.on_perception, 10)
        self.create_subscription(String, "/office_rpg/dynamic_observation",
                                 self.on_dynamic_observation, 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(Clock, "/clock", self.on_clock, 10)
        self.robot = {"x": 1.2, "y": 2.3}
        self.have_odom = False
        self.sim_time = 0.0
        self.sim_origin = 0.0
        self.graph = GraphManager()
        self.plan_gate = PlanGate()
        self.planner_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="office-semantic-planner")
        self.reset_state()
        self.create_timer(0.2, self.tick)
        self.create_timer(0.5, self.publish_status)

    def reset_state(self):
        self.spec = {}
        self.phase = "IDLE"
        self.phase_started = time.monotonic()
        self.mission_started = 0.0
        self.mission_ended = 0.0
        self.active_region = None
        self.active_kind = ""
        self.candidate_probabilities = {}
        self.search_queue = []
        self.searched_regions = []
        self.visited_regions = []
        self.negative_observations = []
        self.pending_request_id = ""
        self.request_counter = 0
        self.visual_triggered = False
        self.person_knowledge = {
            identity: {"identity_id": identity, "display_name": label,
                       "knowledge_state": "UNKNOWN", "last_seen": None}
            for identity, label in PEOPLE.items()
        }
        self.retry_count = 0
        self.fallback_active = False
        self.patrol_checkpoints = []
        self.patrol_index = 0
        self.checkpoint_results = []
        self.detected_people = []
        self.missed_checkpoints = []
        self.report = None
        self.last_error = ""
        self.message = "等待任务"
        self.planned_path = []
        self.graph_initialized = False
        self.planning_future = None
        self.planning_token = 0
        self.current_failure = None
        self.execution_history = []
        self.llm_debug = {
            "api_called": False, "api_configured": False,
            "request": None, "response": None, "error": "",
        }
        self.gate_result = {
            "accepted": False, "rejected_reason": "NOT_PLANNED",
            "graph_revision": self.graph.graph_revision,
        }
        self.graph.reset()
        self.publish_graph()

    def on_clock(self, msg):
        self.sim_time = float(msg.clock.sec) + 1.0e-9 * float(msg.clock.nanosec)

    def schedule_time(self):
        cycle = float(self.cfg.get("cycle_sec", 180.0))
        return (self.initial_time + max(0.0, self.sim_time - self.sim_origin)) % cycle

    def schedule_region_prior(self, person):
        schedule = self.cfg.get(person, {}).get("schedule", [])
        now = self.schedule_time()
        entry = next((item for item in schedule
                      if float(item["start_sec"]) <= now < float(item["end_sec"])),
                     schedule[-1] if schedule else {})
        prior_key = entry.get("region") or entry.get("transition") or "default"
        table = self.cfg.get("search_priors", {}).get(person, {})
        priors = table.get(prior_key, table.get("default", {}))
        candidates = self.spec.get("candidate_regions") or list(priors)
        filtered = {region: float(priors.get(region, 0.01)) for region in candidates}
        total = sum(filtered.values()) or 1.0
        return {region: value / total for region, value in filtered.items()}

    def emit(self, event_type, message, **extra):
        event = {"event_type": event_type, "message": message,
                 "timestamp": time.time(), "stamp": time.time(),
                 "mission_id": self.spec.get("mission_id", ""),
                 "state_machine_phase": self.phase,
                 "source": "office_search_executor"}
        event.update(extra)
        self.execution_history.append({
            key: event.get(key) for key in (
                "event_type", "timestamp", "target_id", "target_region",
                "observed_region", "reason", "message")
            if event.get(key) is not None
        })
        self.execution_history = self.execution_history[-30:]
        self.event_pub.publish(String(data=json.dumps(event, ensure_ascii=False)))

    def publish_graph(self):
        self.graph_pub.publish(String(data=json.dumps(
            self.graph.snapshot(), ensure_ascii=False)))

    def set_phase(self, phase, message=None):
        self.phase = phase
        self.phase_started = time.monotonic()
        if message is not None:
            self.message = message

    def on_spec(self, msg):
        try:
            spec = json.loads(msg.data)
        except ValueError as exc:
            self.fail(f"MissionSpec JSON 无效: {exc}")
            return
        self.task_control_pub.publish(String(data="STOP"))
        self.reset_state()
        self.spec = spec
        self.mission_started = time.time()
        self.set_phase("PARSE_TASK", "正在校验确定性 MissionSpec")
        self.emit("STATE_ENTERED", "进入 PARSE_TASK")

    def on_control(self, msg):
        try:
            command = json.loads(msg.data).get("command", "")
        except ValueError:
            command = msg.data
        if command == "STOP":
            self.task_control_pub.publish(String(data="STOP"))
            self.planning_token += 1
            self.planning_future = None
            self.planned_path = []
            self.pending_request_id = ""
            self.set_phase("STOPPED", "任务已立即停止")
            self.emit("MISSION_STOPPED", "搜索/导航/巡检状态机已停止")
        elif command == "RESET":
            self.task_control_pub.publish(String(data="STOP"))
            self.planning_token += 1
            self.sim_origin = self.sim_time
            self.reset_state()

    def on_odom(self, msg):
        self.robot = {"x": round(float(msg.pose.pose.position.x), 3),
                      "y": round(float(msg.pose.pose.position.y), 3)}
        self.have_odom = True
        if self.phase not in {"NAVIGATE_TO_REGION", "PATROL_CHECKPOINT"}:
            return
        point = self.cfg.get("observation_points", {}).get(self.active_region)
        if not point:
            return
        distance = math.hypot(self.robot["x"] - float(point["x"]),
                              self.robot["y"] - float(point["y"]))
        if distance <= float(point.get("arrival_radius", 0.6)):
            self.task_control_pub.publish(String(data="STOP"))
            if self.active_region not in self.visited_regions:
                self.visited_regions.append(self.active_region)
            self.graph.set_robot_region(self.active_region)
            self.publish_graph()
            self.emit("REGION_ARRIVED", f"到达观察区域 {self.active_region}",
                      observed_region=self.active_region)
            if self.active_kind == "navigate_only":
                self.complete("基础区域导航完成")
            else:
                self.request_observation()

    def on_dynamic_observation(self, msg):
        """Optional safety/topology event input; never carries NPC truth."""
        try:
            event = json.loads(msg.data)
        except ValueError:
            return
        if event.get("event_type") != "EDGE_BLOCKED" or not self.active_region:
            return
        if not self.graph.update_edge_blocked(self.active_region, event):
            return
        self.current_failure = {
            "event_type": "EDGE_BLOCKED",
            "target_region": self.graph.graph_region(self.active_region),
            "timestamp": event.get("timestamp", time.time()),
        }
        self.publish_graph()
        self.emit("EDGE_BLOCKED", "动态阻塞事件已写入拓扑图并触发重规划",
                  target_region=self.active_region,
                  graph_revision=self.graph.graph_revision)
        if self.phase in {"NAVIGATE_TO_REGION", "OBSERVE_REGION"} and \
                self.active_kind == "search":
            self.task_control_pub.publish(String(data="STOP"))
            self.planning_token += 1
            self.planning_future = None
            self.set_phase("PLAN_SEARCH", "边阻塞，重新请求语义规划")

    def navigation_command(self, region):
        return region.removeprefix("room_") if region.startswith("room_") else region

    def start_navigation(self, region, kind):
        if region not in self.cfg.get("observation_points", {}):
            self.fail(f"没有观察点配置: {region}")
            return
        self.active_region = region
        self.active_kind = kind
        point = self.cfg["observation_points"][region]
        self.planned_path = [[self.robot["x"], self.robot["y"]],
                             [float(point["x"]), float(point["y"])]]
        phase = "PATROL_CHECKPOINT" if kind in {"patrol", "conditional_patrol"} \
            else "NAVIGATE_TO_REGION"
        self.set_phase(phase, f"正在前往 {region}")
        self.goal_pub.publish(String(data=self.navigation_command(region)))
        self.task_control_pub.publish(String(data="START"))
        self.emit("NAVIGATION_STARTED", f"开始导航到 {region}",
                  target_region=region, navigation_kind=kind)

    def request_observation(self):
        self.request_counter += 1
        self.pending_request_id = (
            f"{self.spec.get('mission_id','mission')}_obs_{self.request_counter}")
        self.visual_triggered = True
        target = self.spec.get("target_person") if self.active_kind == "search" else ""
        request = {"request_id": self.pending_request_id,
                   "trigger_id": self.pending_request_id,
                   "mission_id": self.spec.get("mission_id", ""),
                   "region": self.active_region, "target_id": target,
                   "mode": "ACTIVE_OBSERVATION", "timestamp": time.time()}
        self.set_phase("OBSERVE_REGION", f"主动观察 {self.active_region}")
        self.perception_pub.publish(String(data=json.dumps(request, ensure_ascii=False)))
        self.emit("PERCEPTION_REQUESTED", "到达观察点后主动触发模拟感知",
                  observed_region=self.active_region, target_id=target)

    def on_perception(self, msg):
        try:
            event = json.loads(msg.data)
        except ValueError:
            return
        if not self.pending_request_id or event.get("request_id") != self.pending_request_id:
            return
        self.pending_request_id = ""
        if self.active_kind == "search":
            if event.get("event_type") == "TARGET_DETECTED" and \
                    event.get("target_id") == self.spec.get("target_person"):
                identity = event["target_id"]
                last_seen = {"region": event.get("observed_region"),
                             "timestamp": event.get("timestamp"),
                             "simulated_confidence": event.get("simulated_confidence")}
                self.person_knowledge[identity].update(
                    {"knowledge_state": "CONFIRMED", "last_seen": last_seen})
                if self.active_region not in self.searched_regions:
                    self.searched_regions.append(self.active_region)
                self.graph.update_target_detected(identity, self.active_region)
                self.publish_graph()
                self.detected_people.append({"identity_id": identity, **last_seen})
                self.emit("TARGET_DETECTED", f"在 {self.active_region} 检测到目标",
                          target_id=identity, observed_region=self.active_region,
                          simulated_confidence=event.get("simulated_confidence"),
                          distance=event.get("distance"))
                self.set_phase("IDENTIFY_TARGET", "正在进行模拟身份确认")
            else:
                observation = {"region": self.active_region, "timestamp": time.time(),
                               "reason": event.get("reason", "target_not_in_detection_zone")}
                self.negative_observations.append(observation)
                if self.active_region not in self.searched_regions:
                    self.searched_regions.append(self.active_region)
                self.emit("TARGET_NOT_FOUND", f"在 {self.active_region} 未发现目标",
                          target_id=self.spec.get("target_person"),
                          observed_region=self.active_region,
                          negative_observation=observation)
                self.set_phase("UPDATE_BELIEF", "根据 negative observation 更新候选概率")
        else:
            detections = event.get("detections", [])
            checkpoint = {"region": self.active_region, "arrival_time": time.time(),
                          "navigation_result": "ARRIVED", "detections": detections}
            self.checkpoint_results.append(checkpoint)
            for detection in detections:
                identity = detection.get("target_id")
                if identity in self.person_knowledge:
                    last_seen = {"region": self.active_region,
                                 "timestamp": detection.get("timestamp", time.time()),
                                 "simulated_confidence": detection.get("simulated_confidence")}
                    self.person_knowledge[identity].update(
                        {"knowledge_state": "LAST_SEEN", "last_seen": last_seen})
                    self.detected_people.append({"identity_id": identity, **last_seen})
                if identity == "visitor":
                    self.emit("VISITOR_DETECTED", "巡检发现访客",
                              observed_region=self.active_region)
                    self.emit("NOTIFICATION_QUEUED", "已在仿真事件队列中通知张老师",
                              notify_person="teacher_zhang")
            self.emit("PATROL_OBSERVATION_COMPLETE",
                      f"完成 checkpoint {self.active_region} 观察",
                      observed_region=self.active_region, detections=detections)
            if self.active_kind == "conditional_patrol":
                self.retry_count += 1
                self.searched_regions = []
                self.candidate_probabilities = self.schedule_region_prior(
                    self.spec.get("target_person"))
                self.graph.set_beliefs(
                    self.spec.get("target_person"),
                    self.candidate_probabilities,
                    clear_searched=True)
                self.current_failure = None
                self.publish_graph()
                self.set_phase("PLAN_SEARCH", "fallback 巡检完成，重新规划一次搜索")
            else:
                self.patrol_index += 1
                self.start_next_patrol()

    def normalize_belief(self):
        total = sum(self.candidate_probabilities.values()) or 1.0
        self.candidate_probabilities = {
            key: value / total for key, value in self.candidate_probabilities.items()}

    def update_negative_belief(self):
        target = self.spec.get("target_person")
        updated = self.graph.update_target_not_found(
            target, self.active_region,
            self.negative_observations[-1].get("reason", "target_not_found"))
        self.candidate_probabilities = {
            self.graph.navigation_region(region): probability
            for region, probability in updated.items()
        }
        self.current_failure = {
            "event_type": "TARGET_NOT_FOUND",
            "target_entity": target,
            "region": self.graph.graph_region(self.active_region),
            "timestamp": time.time(),
        }
        self.publish_graph()
        self.emit("BELIEF_UPDATED", "已降低 negative observation 区域概率",
                  candidate_probabilities=self.candidate_probabilities,
                  graph_revision=self.graph.graph_revision)

    def mission_for_planner(self):
        return {
            key: self.spec.get(key) for key in (
                "mission_id", "mission_type", "input_text", "target_person",
                "message", "completion_conditions", "failure_conditions")
            if self.spec.get(key) is not None
        }

    def planner_request(self, graph=None):
        return {
            "mission": self.mission_for_planner(),
            "graph": graph or self.graph.snapshot(),
            "execution_history": list(self.execution_history[-20:]),
            "current_failure": self.current_failure,
        }

    def provider_job(self, provider_name, request, test_response):
        debug = {
            "provider_requested": provider_name,
            "api_configured": False, "api_called": False,
            "system_prompt": SEMANTIC_PLANNER_SYSTEM_PROMPT,
            "request": request, "response": None, "error": "",
        }
        try:
            if test_response:
                debug["provider_used"] = "acceptance_test_override"
                raw_plan = json.loads(test_response)
            else:
                provider = semantic_provider_for(provider_name)
                debug["provider_used"] = provider.name
                debug["api_configured"] = bool(
                    getattr(provider, "configured", False))
                raw_plan = provider.plan(request)
                debug.update(getattr(provider, "last_metadata", {}))
            debug["response"] = raw_plan
            return raw_plan, debug
        except Exception as exc:  # timeout, network, empty or invalid JSON
            debug["error"] = f"{type(exc).__name__}: {exc}"
            return None, debug

    def start_semantic_planning(self):
        if not self.graph_initialized:
            self.candidate_probabilities = self.schedule_region_prior(
                self.spec.get("target_person"))
            self.graph.reset(
                robot_region=(self.visited_regions[-1]
                              if self.visited_regions else "lobby"),
                entity=self.spec.get("target_person"),
                beliefs=self.candidate_probabilities)
            self.graph_initialized = True
            self.publish_graph()
        graph = self.graph.snapshot()
        request = self.planner_request(graph)
        self.planning_token += 1
        token = self.planning_token
        provider_name = self.spec.get("llm_provider", "offline")
        test_response = str(self.get_parameter("planner_test_response").value)
        self.llm_debug = {
            "provider_requested": provider_name,
            "api_configured": False, "api_called": False,
            "system_prompt": SEMANTIC_PLANNER_SYSTEM_PROMPT,
            "request": request, "response": None, "error": "",
            "status": "PENDING", "graph_revision": graph["graph_revision"],
        }
        self.llm_debug_pub.publish(String(data=json.dumps(
            self.llm_debug, ensure_ascii=False)))
        self.planning_future = (
            token,
            self.planner_pool.submit(
                self.provider_job, provider_name, request, test_response))
        self.message = "正在请求单步语义规划（STOP仍可立即响应）"

    def consume_semantic_plan(self):
        if not self.planning_future:
            self.start_semantic_planning()
            return
        token, future = self.planning_future
        if not future.done():
            return
        self.planning_future = None
        if token != self.planning_token or self.phase != "PLAN_SEARCH":
            return
        raw_plan, debug = future.result()
        current_graph = self.graph.snapshot()
        if raw_plan is None:
            primary_gate = {
                "accepted": False, "graph_revision": current_graph["graph_revision"],
                "rejected_reason": "PROVIDER_ERROR", "plan": None,
            }
        else:
            primary_gate = self.plan_gate.validate(
                raw_plan, current_graph, PEOPLE.keys())

        fallback_used = not primary_gate["accepted"]
        final_plan = raw_plan
        final_gate = primary_gate
        if fallback_used:
            self.emit("PLAN_REJECTED", "远程/可选计划未通过门控，回退 Offline",
                      rejected_reason=primary_gate["rejected_reason"],
                      graph_revision=current_graph["graph_revision"])
            fallback_request = self.planner_request(current_graph)
            final_plan = OfflineSemanticProvider().plan(fallback_request)
            final_gate = self.plan_gate.validate(
                final_plan, current_graph, PEOPLE.keys())

        self.llm_debug = {
            **debug, "status": "COMPLETED",
            "fallback_used": fallback_used,
            "fallback_response": final_plan if fallback_used else None,
            "graph_revision": current_graph["graph_revision"],
        }
        self.gate_result = {
            **final_gate, "primary_result": primary_gate,
            "fallback_used": fallback_used,
        }
        self.llm_debug_pub.publish(String(data=json.dumps(
            self.llm_debug, ensure_ascii=False)))
        self.gate_pub.publish(String(data=json.dumps(
            self.gate_result, ensure_ascii=False)))
        if not final_gate["accepted"]:
            self.fail("远程计划和 Offline fallback 均未通过 plan_gate")
            return
        self.emit("PLAN_ACCEPTED", "单步语义计划通过证据门控",
                  action=final_plan.get("action"),
                  target_region=final_plan.get("target_region"),
                  graph_revision=current_graph["graph_revision"],
                  fallback_used=fallback_used)
        self.execute_semantic_action(final_plan)

    def execute_semantic_action(self, plan):
        action = plan.get("action")
        if action == "SEARCH_REGION":
            target = self.graph.navigation_region(plan["target_region"])
            beliefs = next((item["candidates"]
                            for item in self.graph.snapshot()["entity_beliefs"]
                            if item["entity"] == self.spec.get("target_person")), {})
            ranked = sorted(
                (self.graph.navigation_region(region) for region in beliefs
                 if not self.graph.nodes[region]["searched"]
                 and self.graph.nodes[region]["reachable"]),
                key=lambda region: beliefs[self.graph.graph_region(region)],
                reverse=True)
            self.search_queue = [target] + [item for item in ranked if item != target]
            self.start_navigation(target, "search")
        elif action == "PATROL_REGION":
            self.start_navigation(
                self.graph.navigation_region(plan["target_region"]), "patrol")
        elif action == "WAIT":
            self.task_control_pub.publish(String(data="STOP"))
            self.set_phase("STOPPED", "语义规划选择 WAIT，保持安全停止")
        elif action == "COMPLETE_TASK":
            self.complete("语义规划确认任务完成")
        elif action == "NOTIFY_PERSON" and any(
                item["knowledge_state"] == "CONFIRMED"
                for item in self.person_knowledge.values()):
            self.set_phase("DELIVER_MESSAGE", "语义规划请求发送提醒")
        else:
            self.fail("语义规划返回 ABORT_TASK 或当前不可执行动作")

    def replan_search_queue(self):
        scored = []
        for region, probability in self.candidate_probabilities.items():
            if region in self.searched_regions:
                continue
            point = self.cfg["observation_points"].get(region, {})
            distance = math.hypot(self.robot["x"] - float(point.get("x", 0.0)),
                                  self.robot["y"] - float(point.get("y", 0.0)))
            score = probability / (1.0 + distance / 12.0)
            scored.append((score, probability, region))
        scored.sort(reverse=True)
        self.search_queue = [item[2] for item in scored]
        self.emit("SEARCH_REPLANNED", "按概率、路径代价和访问历史生成搜索队列",
                  search_queue=self.search_queue,
                  candidate_probabilities=self.candidate_probabilities)

    def start_next_search(self):
        self.replan_search_queue()
        if not self.search_queue:
            self.fail("候选区域已耗尽，未找到目标")
            return
        self.start_navigation(self.search_queue[0], "search")

    def start_next_patrol(self):
        if self.patrol_index >= len(self.patrol_checkpoints):
            self.set_phase("GENERATE_REPORT", "巡检 checkpoint 已完成，生成结构化报告")
            return
        self.start_navigation(self.patrol_checkpoints[self.patrol_index], "patrol")

    def generate_report(self):
        self.mission_ended = time.time()
        visited = [item["region"] for item in self.checkpoint_results]
        self.report = {
            "mission_id": self.spec.get("mission_id", ""),
            "mission_type": self.spec.get("mission_type"),
            "scheduled_time": self.spec.get("scheduled_time", ""),
            "start_time": self.mission_started, "end_time": self.mission_ended,
            "visited_checkpoints": visited,
            "missed_checkpoints": [p for p in self.patrol_checkpoints if p not in visited]
                                  + self.missed_checkpoints,
            "checkpoint_results": self.checkpoint_results,
            "detected_people": self.detected_people,
            "failures": ([self.last_error] if self.last_error else []),
            "final_state": "COMPLETED",
            "source": "office_search_executor",
        }
        self.report_pub.publish(String(data=json.dumps(self.report, ensure_ascii=False)))
        self.emit("REPORT_GENERATED", "已生成结构化巡检简报", report=self.report)
        self.complete("巡检与日报生成完成")

    def complete(self, message):
        self.task_control_pub.publish(String(data="STOP"))
        self.mission_ended = time.time()
        self.planned_path = []
        self.set_phase("COMPLETED", message)
        self.emit("MISSION_COMPLETED", message)

    def fail(self, reason):
        self.task_control_pub.publish(String(data="STOP"))
        self.last_error = reason
        self.mission_ended = time.time()
        self.planned_path = []
        self.set_phase("FAILED", reason)
        self.emit("MISSION_FAILED", reason)

    def tick(self):
        if self.phase in {"IDLE", *TERMINAL_STATES}:
            return
        elapsed = time.monotonic() - self.phase_started
        if self.phase == "PARSE_TASK":
            required = {"mission_type", "input_text", "mission_id"}
            if not required.issubset(self.spec):
                self.fail("MissionSpec 缺少必要字段")
                return
            self.set_phase("PLAN_SEARCH", "规划信息受限任务")
            self.emit("STATE_ENTERED", "进入 PLAN_SEARCH")
        elif self.phase == "PLAN_SEARCH":
            mission_type = self.spec.get("mission_type")
            if mission_type == "NAVIGATE_TO_REGION":
                self.start_navigation(self.spec.get("target_region"), "navigate_only")
            elif mission_type == "PATROL_AND_REPORT":
                self.patrol_checkpoints = list(self.spec.get("checkpoints") or [])
                self.patrol_index = 0
                self.emit("SCHEDULE_TRIGGERED",
                          "仿真定时巡检触发（加速/立即执行，不含手机推送）",
                          scheduled_time=self.spec.get("scheduled_time"))
                self.start_next_patrol()
            elif mission_type in {"SEARCH_PERSON", "SEARCH_AND_NOTIFY", "CONDITIONAL_PATROL"}:
                self.consume_semantic_plan()
            else:
                self.fail(f"不支持的 mission_type: {mission_type}")
        elif self.phase == "UPDATE_BELIEF":
            self.update_negative_belief()
            if self.spec.get("mission_type") == "CONDITIONAL_PATROL" and \
                    not self.fallback_active and self.retry_count < int(self.spec.get("retry_limit", 0)):
                self.fallback_active = True
                self.patrol_checkpoints = list(self.spec.get("checkpoints") or ["room_908"])
                self.patrol_index = 0
                self.start_navigation(self.patrol_checkpoints[0], "conditional_patrol")
            else:
                self.set_phase("PLAN_SEARCH", "图已更新，重新请求单步语义规划")
        elif self.phase == "IDENTIFY_TARGET":
            identity = self.spec.get("target_person")
            self.emit("IDENTITY_CONFIRMED", f"模拟身份确认：{PEOPLE.get(identity, identity)}",
                      target_id=identity, observed_region=self.active_region)
            if self.spec.get("message"):
                self.set_phase("DELIVER_MESSAGE", "发送模拟提醒事件")
            else:
                self.complete("目标人物已找到并确认")
        elif self.phase == "DELIVER_MESSAGE":
            event = {"event_type": "MESSAGE_DELIVERED",
                     "target_id": self.spec.get("target_person"),
                     "message": self.spec.get("message", ""),
                     "observed_region": self.active_region,
                     "timestamp": time.time(), "source": "office_search_executor",
                     "mission_id": self.spec.get("mission_id", "")}
            self.reminder_pub.publish(String(data=json.dumps(event, ensure_ascii=False)))
            self.emit("MESSAGE_DELIVERED", "模拟提醒已送达",
                      target_id=event["target_id"],
                      observed_region=event["observed_region"],
                      delivered_message=event["message"])
            self.complete("找到目标并完成提醒")
        elif self.phase == "GENERATE_REPORT":
            self.generate_report()
        elif self.phase in {"NAVIGATE_TO_REGION", "PATROL_CHECKPOINT"} and \
                elapsed > float(self.get_parameter("navigation_timeout_sec").value):
            if self.active_kind == "patrol":
                self.missed_checkpoints.append(self.active_region)
                self.emit("CHECKPOINT_FAILED", f"导航到 {self.active_region} 超时")
                self.patrol_index += 1
                self.start_next_patrol()
            else:
                self.fail(f"导航到 {self.active_region} 超时")
        elif self.phase == "OBSERVE_REGION" and \
                elapsed > float(self.get_parameter("observation_timeout_sec").value):
            self.fail(f"观察 {self.active_region} 超时")

    def status(self):
        total_candidates = len(self.candidate_probabilities)
        searched = len(self.searched_regions)
        return {
            "mission_id": self.spec.get("mission_id", ""),
            "state": self.phase, "state_machine_phase": self.phase,
            "input_text": self.spec.get("input_text", ""),
            "mission_type": self.spec.get("mission_type", ""),
            "task_type": self.spec.get("mission_type", ""),
            "target_person": self.spec.get("target_person"),
            "target_region": self.active_region or self.spec.get("target_region"),
            "current_region": self.visited_regions[-1] if self.visited_regions else "lobby",
            "candidate_regions": list(self.candidate_probabilities),
            "candidate_probabilities": self.candidate_probabilities,
            "search_queue": self.search_queue,
            "searched_regions": self.searched_regions,
            "visited_regions": self.visited_regions,
            "negative_observations": self.negative_observations,
            "search_progress": {"searched": searched, "total": total_candidates,
                                "text": f"已搜索 {searched}/{total_candidates} 个候选区域"},
            "current_action": self.message,
            "message": self.message,
            "reminder_text": self.spec.get("message", ""),
            "visual_triggered": self.visual_triggered,
            "identity_confirmed": any(
                item["knowledge_state"] == "CONFIRMED"
                for item in self.person_knowledge.values()),
            "person_knowledge": self.person_knowledge,
            "patrol_progress": {"completed": len(self.checkpoint_results),
                                "total": len(self.patrol_checkpoints)},
            "patrol_report": self.report,
            "retry_count": self.retry_count, "retry_limit": self.spec.get("retry_limit", 0),
            "fallback_active": self.fallback_active,
            "llm_provider": self.spec.get("llm_provider", "offline"),
            "last_error": self.last_error, "robot_pose": self.robot,
            "planned_path": self.planned_path, "phase": "PHASE_2_MVP",
            "graph_revision": self.graph.graph_revision,
            "plan_gate_accepted": self.gate_result.get("accepted", False),
            "plan_gate_rejected_reason": self.gate_result.get("rejected_reason", ""),
            "semantic_provider": self.llm_debug.get("provider_used",
                                                    self.spec.get("llm_provider", "offline")),
            "truth_access": "NONE",
        }

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.status(), ensure_ascii=False)))


def main(args=None):
    rclpy.init(args=args)
    node = OfficeSearchExecutor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.planner_pool.shutdown(wait=False, cancel_futures=True)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
