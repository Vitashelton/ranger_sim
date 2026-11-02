"""Pure helpers for tolerant ROS JSON state handling."""
import json


EMPTY_STATUS = {
    "state": "WAITING_FOR_ROS", "input_text": "", "target_person": None,
    "current_region": "lobby", "candidate_regions": [], "visited_regions": [],
    "current_action": "idle", "llm_provider": "offline", "last_error": "",
    "robot_pose": {"x": 1.2, "y": 2.3}, "planned_path": [],
    "visual_triggered": False, "identity_confirmed": False, "reminder_text": "",
    "state_machine_phase": "WAITING_FOR_ROS", "search_queue": [],
    "candidate_probabilities": {}, "search_progress": {"searched": 0, "total": 0},
    "person_knowledge": {}, "patrol_report": None,
    "graph_revision": 0, "plan_gate_accepted": False,
    "plan_gate_rejected_reason": "",
}


def decode(payload, fallback):
    if not payload:
        return dict(fallback)
    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        result = dict(fallback)
        result["last_error"] = "收到无法解析的 ROS JSON"
        return result


PERSON_LABELS = {"teacher_zhang": "张老师", "student_li": "李同学",
                 "visitor": "访客", None: "—"}


def person_label(identity):
    return PERSON_LABELS.get(identity, identity or "—")
