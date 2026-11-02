"""Mission-only LLM provider boundary. Providers never emit chassis commands."""
from dataclasses import asdict, dataclass, field
import json
import os
import re
import time
from typing import Any, Dict, List, Optional


SEMANTIC_PLANNER_SYSTEM_PROMPT = """You are the high-level semantic task planner for an indoor mobile robot.

You receive a mission and a dynamic semantic topological graph in JSON.
Select exactly one next semantic action.

Rules:
1. Output one JSON object only.
2. Never invent nodes, people, edges or actions.
3. Use only values listed in allowed_actions.
4. target_region must be an existing graph node.
5. Do not output coordinates, paths, velocities or ROS commands.
6. Prefer unsearched reachable regions with stronger target belief.
7. If the previous region returned TARGET_NOT_FOUND, choose another region.
8. If no valid region exists, output ABORT_TASK.
9. A* and the robot navigation system will calculate the physical route.
10. Use exactly the six fields shown below. Do not add reasoning, coordinates,
    paths, scores, ROS commands, or any other field.

JSON output:
{
  "graph_revision": 0,
  "action": "SEARCH_REGION",
  "target_region": "room_904",
  "target_entity": "teacher_zhang",
  "reason_code": "HIGHEST_VALID_BELIEF",
  "fallback_region": "room_906"
}

Copy graph_revision from the input graph without changing it. Use null for a
field that does not apply. Put the explanation into reason_code only."""


@dataclass
class MissionPlan:
    mission_type: str
    input_text: str
    target_region: Optional[str] = None
    target_person: Optional[str] = None
    message: str = ""
    candidate_regions: List[str] = field(default_factory=list)
    primary_steps: List[str] = field(default_factory=list)
    fallback_steps: List[str] = field(default_factory=list)
    retry_limit: int = 0
    checkpoints: List[str] = field(default_factory=list)
    completion_conditions: List[str] = field(default_factory=list)
    failure_conditions: List[str] = field(default_factory=list)
    scheduled_time: str = ""
    watch_for: Optional[str] = None
    notify_person: Optional[str] = None
    provider: str = "offline"
    model: str = ""
    fallback_reason: str = ""

    @property
    def task_type(self):
        """Compatibility alias for Phase 1 status consumers."""
        return self.mission_type

    @property
    def reminder_text(self):
        return self.message

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMProvider:
    name = "base"

    def parse_mission(self, text: str, context: Dict[str, Any]) -> MissionPlan:
        raise NotImplementedError


class OfflineRuleProvider(LLMProvider):
    name = "offline"
    PEOPLE = {
        "张老师": ("teacher_zhang", ["room_904", "room_906"]),
        "teacher_zhang": ("teacher_zhang", ["room_904", "room_906"]),
        "李同学": ("student_li", ["room_908", "corridor_junction"]),
        "student_li": ("student_li", ["room_908", "corridor_junction"]),
        "访客": ("visitor", ["lobby"]),
        "visitor": ("visitor", ["lobby"]),
    }

    def parse_mission(self, text: str, context: Dict[str, Any]) -> MissionPlan:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("任务文本不能为空")
        if re.search(r"停止|停下|stop", cleaned, re.I):
            return MissionPlan("STOP_MISSION", cleaned,
                               primary_steps=["STOP"],
                               completion_conditions=["state=STOPPED"])
        conditional = (
            ("李同学" in cleaned or "student_li" in cleaned)
            and re.search(r"第一次.*没找到|未找到.*巡检908|找不到.*908|再找一次", cleaned))
        if conditional:
            reminder = ""
            if "提醒" in cleaned:
                reminder = cleaned.split("提醒", 1)[1]
                reminder = re.split(r"[；;]?[如若]果", reminder, maxsplit=1)[0]
                reminder = reminder.strip("，,。；; ")
                reminder = re.sub(r"^[他她]\s*", "", reminder)
            return MissionPlan(
                "CONDITIONAL_PATROL", cleaned, target_person="student_li",
                message=reminder or "提交报告",
                candidate_regions=["room_908", "corridor_junction"],
                primary_steps=["SEARCH_PERSON", "DELIVER_MESSAGE"],
                fallback_steps=["PATROL:room_908", "SEARCH_PERSON", "DELIVER_MESSAGE"],
                retry_limit=1, checkpoints=["room_908"],
                completion_conditions=["target_confirmed", "message_delivered"],
                failure_conditions=["retry_limit_exceeded", "navigation_failed"])
        if "巡检" in cleaned:
            checkpoints = ["lobby", "room_904", "room_906"]
            scheduled = "08:00" if re.search(r"(?:每天)?\s*8点|八点", cleaned) else "IMMEDIATE"
            return MissionPlan(
                "PATROL_AND_REPORT", cleaned, checkpoints=checkpoints,
                primary_steps=["PATROL_CHECKPOINT", "OBSERVE_REGION", "GENERATE_REPORT"],
                scheduled_time=scheduled, watch_for="visitor",
                notify_person="teacher_zhang",
                completion_conditions=["all_reachable_checkpoints_visited", "report_generated"],
                failure_conditions=["all_checkpoints_failed"])
        person_id = None
        candidates: List[str] = []
        matched_name = ""
        for name, (identity, regions) in self.PEOPLE.items():
            if name in cleaned:
                matched_name = name
                person_id, candidates = identity, list(regions)
                break
        if person_id:
            reminder = ""
            if "提醒" in cleaned:
                reminder = cleaned.split("提醒", 1)[1].strip("，,。 ")
                reminder = re.sub(r"^[他她]\s*", "", reminder)
            mission_type = "SEARCH_AND_NOTIFY" if reminder else "SEARCH_PERSON"
            steps = ["PLAN_SEARCH", "NAVIGATE_TO_REGION", "OBSERVE_REGION",
                     "IDENTIFY_TARGET"]
            if reminder:
                steps.append("DELIVER_MESSAGE")
            return MissionPlan(
                mission_type, cleaned, target_person=person_id, message=reminder,
                candidate_regions=candidates, primary_steps=steps,
                completion_conditions=["target_confirmed"] +
                                      (["message_delivered"] if reminder else []),
                failure_conditions=["candidate_regions_exhausted", "navigation_failed"])
        room = re.search(r"(?:去|前往|导航到)?\s*(904|906|908)", cleaned)
        if room:
            return MissionPlan("NAVIGATE_TO_REGION", cleaned,
                               target_region=f"room_{room.group(1)}",
                               primary_steps=["NAVIGATE_TO_REGION"],
                               completion_conditions=["target_region_arrived"])
        raise ValueError(f"离线规则暂时无法解析：{cleaned}")


class LocalQwenProvider(LLMProvider):
    name = "local_qwen"

    def parse_mission(self, text: str, context: Dict[str, Any]) -> MissionPlan:
        plan = OfflineRuleProvider().parse_mission(text, context)
        plan.fallback_reason = "Phase 1.5 未接入 Local Qwen，已回退 Offline"
        return plan


class DeepSeekProvider(LLMProvider):
    name = "deepseek"
    ALLOWED_MODELS = {"deepseek-v4-flash", "deepseek-v4-pro"}

    def __init__(self, model: str):
        if model not in self.ALLOWED_MODELS:
            raise ValueError(f"不支持的 DeepSeek 模型：{model}")
        self.model = model

    @property
    def configured(self) -> bool:
        return bool(os.environ.get("LLM_API_KEY"))

    def parse_mission(self, text: str, context: Dict[str, Any]) -> MissionPlan:
        # Mission parsing remains deterministic.  Phase 2 invokes the remote
        # provider only for one-step semantic planning inside PLAN_SEARCH.
        plan = OfflineRuleProvider().parse_mission(text, context)
        plan.model = self.model
        plan.fallback_reason = "" if self.configured else "API未配置，PLAN_SEARCH 将安全回退 Offline"
        return plan


def provider_for(name: str, model: str = "") -> LLMProvider:
    normalized = name.strip().lower().replace(" ", "_")
    if normalized == "local_qwen":
        return LocalQwenProvider()
    if normalized in {"deepseek_v4_flash", "deepseek_v4_pro", "deepseek"}:
        chosen = model or normalized.replace("_", "-")
        if chosen == "deepseek":
            chosen = "deepseek-v4-flash"
        return DeepSeekProvider(chosen)
    return OfflineRuleProvider()


class OfflineSemanticProvider:
    """Deterministic one-step planner used for fallback and no-key operation."""
    name = "offline"

    def plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        graph = request["graph"]
        mission = request["mission"]
        target = mission.get("target_person")
        beliefs = next((item.get("candidates", {})
                        for item in graph.get("entity_beliefs", [])
                        if item.get("entity") == target), {})
        nodes = {item["id"]: item for item in graph.get("nodes", [])}
        ranked = sorted(
            ((float(probability), region) for region, probability in beliefs.items()
             if region in nodes and nodes[region].get("reachable", False)
             and not nodes[region].get("searched", False)),
            reverse=True)
        if not ranked:
            return {
                "graph_revision": graph["graph_revision"],
                "action": "ABORT_TASK", "target_region": None,
                "target_entity": target, "reason_code": "NO_VALID_REGION",
                "fallback_region": None,
            }
        fallback = ranked[1][1] if len(ranked) > 1 else None
        return {
            "graph_revision": graph["graph_revision"],
            "action": "SEARCH_REGION", "target_region": ranked[0][1],
            "target_entity": target,
            "reason_code": "HIGHEST_VALID_BELIEF",
            "fallback_region": fallback,
        }


class OpenAICompatibleProvider:
    """Remote semantic planner using the OpenAI-compatible JSON API only."""
    name = "openai_compatible"

    def __init__(self):
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
        self.model = os.environ.get("LLM_MODEL", "deepseek-v4-flash")
        try:
            self.timeout_s = max(0.1, float(os.environ.get("LLM_TIMEOUT_S", "15")))
        except ValueError:
            self.timeout_s = 15.0
        self.last_metadata = {}

    @property
    def configured(self):
        return bool(self.api_key)

    def plan(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if not self.configured:
            raise RuntimeError("LLM_API_KEY is not configured")
        # Import lazily so Offline mode never depends on client initialization.
        from openai import OpenAI
        started = time.monotonic()
        self.last_metadata = {
            "api_called": True, "provider": self.name,
            "base_url": self.base_url, "model": self.model,
            "timeout_s": self.timeout_s,
        }
        client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                        timeout=self.timeout_s, max_retries=0)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SEMANTIC_PLANNER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    request, ensure_ascii=False, separators=(",", ":"))},
            ],
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            stream=False,
        )
        self.last_metadata["latency_s"] = round(time.monotonic() - started, 3)
        if not response.choices:
            raise ValueError("Remote planner returned no choices")
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ValueError("Remote planner returned empty content")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("Remote planner JSON is not an object")
        self.last_metadata["response_id"] = getattr(response, "id", "")
        self.last_metadata["remote_success"] = True
        return parsed


def semantic_provider_for(name: str):
    normalized = (name or "offline").strip().lower().replace(" ", "_")
    if normalized in {
            "deepseek", "deepseek_v4_flash", "deepseek_v4_pro",
            "openai_compatible", "remote"}:
        return OpenAICompatibleProvider()
    return OfflineSemanticProvider()
