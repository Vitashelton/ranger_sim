"""Minimal evidence gate for one-step semantic LLM plans."""
from copy import deepcopy


PLAN_FIELDS = {
    "graph_revision", "action", "target_region", "target_entity",
    "reason_code", "fallback_region",
}
REGION_ACTIONS = {"SEARCH_REGION", "PATROL_REGION"}
FORBIDDEN_KEYS = {
    "x", "y", "z", "coordinates", "pose", "path", "velocity", "cmd_vel",
    "linear", "angular", "ros_command",
}


class PlanGate:
    @staticmethod
    def _contains_forbidden(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in FORBIDDEN_KEYS:
                    return True
                if PlanGate._contains_forbidden(child):
                    return True
        elif isinstance(value, list):
            return any(PlanGate._contains_forbidden(item) for item in value)
        return False

    def validate(self, plan, graph, known_entities):
        result = {
            "accepted": False,
            "graph_revision": graph.get("graph_revision"),
            "rejected_reason": "",
            "plan": deepcopy(plan) if isinstance(plan, dict) else plan,
        }
        if not isinstance(plan, dict):
            result["rejected_reason"] = "PLAN_NOT_JSON_OBJECT"
            return result
        unknown = set(plan) - PLAN_FIELDS
        if unknown:
            result["rejected_reason"] = "UNKNOWN_FIELDS:" + ",".join(sorted(unknown))
            return result
        if self._contains_forbidden(plan):
            result["rejected_reason"] = "FORBIDDEN_CONTROL_OR_COORDINATE_FIELD"
            return result
        if plan.get("graph_revision") != graph.get("graph_revision"):
            result["rejected_reason"] = "STALE_GRAPH_REVISION"
            return result
        allowed = set(graph.get("allowed_actions", []))
        action = plan.get("action")
        if action not in allowed:
            result["rejected_reason"] = "ACTION_NOT_ALLOWED"
            return result
        nodes = {item.get("id"): item for item in graph.get("nodes", [])}
        region = plan.get("target_region")
        if action in REGION_ACTIONS and region not in nodes:
            result["rejected_reason"] = "TARGET_REGION_UNKNOWN"
            return result
        if region is not None and region not in nodes:
            result["rejected_reason"] = "TARGET_REGION_UNKNOWN"
            return result
        if region is not None and not bool(nodes[region].get("reachable", False)):
            result["rejected_reason"] = "TARGET_REGION_UNREACHABLE"
            return result
        fallback = plan.get("fallback_region")
        if fallback is not None and fallback not in nodes:
            result["rejected_reason"] = "FALLBACK_REGION_UNKNOWN"
            return result
        entity = plan.get("target_entity")
        if entity is not None and entity not in set(known_entities):
            result["rejected_reason"] = "TARGET_ENTITY_UNKNOWN"
            return result
        result["accepted"] = True
        return result
