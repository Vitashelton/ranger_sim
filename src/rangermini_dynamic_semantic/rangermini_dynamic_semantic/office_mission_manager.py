"""Mission text parser and safety facade for Office RPG Phase 1.5.

This node never consumes NPC truth and never generates chassis commands. It
turns text into a deterministic MissionSpec and routes STOP through the existing
task control safety chain.
"""
import json
import os
import time
import uuid

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .llm_providers import OfflineRuleProvider, provider_for


PUBLIC_REGIONS = {
    "lobby": {"display_name": "Lobby", "x": 1.2, "y": 2.3},
    "room_904": {"display_name": "904门口", "x": 8.0, "y": 4.15},
    "room_906": {"display_name": "906门口", "x": 13.35, "y": 4.15},
    "room_908": {"display_name": "908门口", "x": 18.55, "y": 4.15},
    "corridor_junction": {"display_name": "走廊交叉口", "x": 11.0, "y": 3.75},
}


class OfficeMissionManager(Node):
    def __init__(self):
        super().__init__("office_rpg_mission_manager")
        self.declare_parameter("llm_provider", "offline")
        self.declare_parameter("llm_model", "deepseek-v4-flash")
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.regions_pub = self.create_publisher(
            String, "/office_rpg/semantic_regions", latched)
        self.spec_pub = self.create_publisher(String, "/office_rpg/mission_spec", 10)
        self.control_pub = self.create_publisher(String, "/office_rpg/executor_control", 10)
        self.event_pub = self.create_publisher(String, "/office_rpg/event_log", 50)
        self.task_control_pub = self.create_publisher(String, "/task_control", 10)
        self.reset_schedule_pub = self.create_publisher(
            String, "/office_rpg/schedule_reset", 10)
        self.create_subscription(String, "/office_rpg/mission_text", self.on_mission, 10)
        self.create_subscription(String, "/office_rpg/llm_provider", self.on_provider, 10)
        self.create_service(Trigger, "/office_rpg/stop", self.on_stop)
        self.create_service(Trigger, "/office_rpg/reset", self.on_reset)
        self.provider_name = str(self.get_parameter("llm_provider").value)
        self.provider_model = str(self.get_parameter("llm_model").value)
        self.current_mission_id = ""
        self.regions_pub.publish(String(data=json.dumps(
            {"regions": PUBLIC_REGIONS, "coordinate_scope": "observation_points_only"},
            ensure_ascii=False)))
        self.emit("SYSTEM_READY", "Office RPG Phase 2 MVP MissionSpec 接口已就绪")

    def emit(self, event_type, message, **extra):
        payload = {"event_type": event_type, "message": message,
                   "timestamp": time.time(), "stamp": time.time(),
                   "mission_id": self.current_mission_id,
                   "source": "office_mission_manager"}
        payload.update(extra)
        self.event_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def on_provider(self, msg):
        mapping = {
            "Offline": ("offline", ""), "Local Qwen": ("local_qwen", ""),
            "DeepSeek V4 Flash": ("deepseek_v4_flash", "deepseek-v4-flash"),
            "DeepSeek V4 Pro": ("deepseek_v4_pro", "deepseek-v4-pro"),
        }
        value = msg.data.strip()
        self.provider_name, requested_model = mapping.get(value, (value.lower(), ""))
        if requested_model:
            self.provider_model = requested_model

    def on_mission(self, msg):
        text = msg.data.strip()
        self.current_mission_id = f"mission_{uuid.uuid4().hex[:8]}"
        self.emit("MISSION_RECEIVED", "接收任务", input_text=text)
        try:
            provider = provider_for(self.provider_name, self.provider_model)
            plan = provider.parse_mission(text, {"regions": PUBLIC_REGIONS})
        except Exception as provider_error:
            self.emit("PARSER_FALLBACK", "可选解析器失败，安全回退 Offline",
                      error=str(provider_error))
            try:
                plan = OfflineRuleProvider().parse_mission(
                    text, {"regions": PUBLIC_REGIONS})
                plan.fallback_reason = f"解析器失败，已回退 Offline: {provider_error}"
                self.provider_name = "offline"
            except Exception as offline_error:
                self.emit("MISSION_FAILED", "Offline 规则无法解析任务",
                          error=str(offline_error))
                return
        plan.provider = self.provider_name
        spec = plan.to_dict()
        spec.update({
            "mission_id": self.current_mission_id,
            "llm_provider": self.provider_name,
            "llm_model": plan.model or self.provider_model,
            "llm_api_configured": bool(os.environ.get("LLM_API_KEY")),
            "phase": "PHASE_2_MVP",
        })
        self.emit("MISSION_PARSED", "确定性 MissionSpec 已生成", mission_spec=spec)
        if plan.fallback_reason:
            self.emit("FALLBACK_ENABLED", plan.fallback_reason)
        if plan.mission_type == "STOP_MISSION":
            self.stop_internal("文本停止命令")
            return
        self.spec_pub.publish(String(data=json.dumps(spec, ensure_ascii=False)))

    def stop_internal(self, reason):
        # Immediate safety action is independent of executor scheduling latency.
        self.task_control_pub.publish(String(data="STOP"))
        self.control_pub.publish(String(data=json.dumps(
            {"command": "STOP", "reason": reason, "timestamp": time.time()},
            ensure_ascii=False)))
        self.emit("MISSION_STOP_REQUESTED", reason)

    def on_stop(self, _request, response):
        self.stop_internal("用户停止任务")
        response.success = True
        response.message = "已停止搜索/导航/巡检并进入安全停车链"
        return response

    def on_reset(self, _request, response):
        self.task_control_pub.publish(String(data="STOP"))
        self.control_pub.publish(String(data=json.dumps(
            {"command": "RESET", "timestamp": time.time()})))
        self.reset_schedule_pub.publish(String(data="RESET"))
        self.emit("SCENE_RESET", "任务状态和 NPC 日程已重置")
        response.success = True
        response.message = "Phase 2 MVP 状态机、动态图和 NPC 日程已重置"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = OfficeMissionManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
