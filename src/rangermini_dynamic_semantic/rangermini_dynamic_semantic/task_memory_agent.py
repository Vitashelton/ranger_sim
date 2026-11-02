"""Risk-based decisions for reusing or revalidating task-relevant memory.

This module is ROS- and simulator-independent so the decision can be audited
before it is connected to a Gazebo executor. It does not perform perception:
the caller supplies scoped facts, evidence quality, geometric conflict, and
cost estimates for feasible actions.
"""
from dataclasses import dataclass
import math
import json
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class TaskFact:
    fact_id: str
    map_id: str
    region_id: str
    predicate: str
    value: str
    captured_at: float
    reliability: float
    impact: float
    dynamic: bool = True


@dataclass(frozen=True)
class ActionCost:
    """Estimated costs and attainable evidence gain for one feasible action."""
    action: str
    movement_s: float = 0.0
    wait_s: float = 0.0
    query_cost: float = 0.0
    residual_risk_fraction: float = 1.0
    affected_nodes: int = 0
    feasible: bool = True
    mode: str = "NONE"
    viewpoint: tuple = ()


@dataclass(frozen=True)
class Decision:
    action: str
    fact_id: str
    risk: float
    score: float
    scores: dict
    rationale: str
    mode: str
    viewpoint: tuple


class TaskMemoryAgent:
    """Choose continue / inspect / wait / repair / report-uncertain by expected cost.

    The objective is an explicit engineering prototype, not a learned policy.
    ``risk`` combines age-dependent reliability, current conflict, and the
    task impact of making a wrong decision. Stable facts decay more slowly;
    age alone never converts a fact into evidence of absence.
    """

    ACTIONS = {"continue", "inspect", "wait", "local_repair",
               "report_uncertain"}

    def __init__(self, dynamic_half_life_s=900.0, stable_half_life_s=86400.0,
                 time_cost_per_s=0.01, query_cost_weight=1.0,
                 repair_node_cost=0.4, stop_cost=3.0):
        vals = (dynamic_half_life_s, stable_half_life_s, time_cost_per_s,
                query_cost_weight, repair_node_cost, stop_cost)
        if not all(math.isfinite(x) and x >= 0 for x in vals):
            raise ValueError("agent parameters must be finite and nonnegative")
        if dynamic_half_life_s == 0 or stable_half_life_s == 0:
            raise ValueError("memory half-lives must be positive")
        self.dynamic_half_life_s = float(dynamic_half_life_s)
        self.stable_half_life_s = float(stable_half_life_s)
        self.time_cost_per_s = float(time_cost_per_s)
        self.query_cost_weight = float(query_cost_weight)
        self.repair_node_cost = float(repair_node_cost)
        self.stop_cost = float(stop_cost)

    @staticmethod
    def _unit(value, name):
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return value

    def fact_risk(self, fact, now, conflict=0.0):
        if not isinstance(fact, TaskFact):
            raise TypeError("fact must be TaskFact")
        if not math.isfinite(now) or not math.isfinite(fact.captured_at):
            raise ValueError("timestamps must be finite")
        if now < fact.captured_at:
            raise ValueError("now precedes fact capture")
        base = self._unit(fact.reliability, "reliability")
        impact = self._unit(fact.impact, "impact")
        conflict = self._unit(conflict, "conflict")
        half_life = (self.dynamic_half_life_s if fact.dynamic
                     else self.stable_half_life_s)
        aged_reliability = base * math.exp(-math.log(2.0) *
                                            (now-fact.captured_at) / half_life)
        uncertainty = 1.0 - aged_reliability
        # Current geometric/evidence conflict raises the probability that
        # reusing the old fact changes the task decision.
        return min(1.0, impact * (uncertainty + (1.0-uncertainty)*conflict))

    def decide(self, fact, now, conflict=0.0, action_costs=(),
               unresolved_task_nodes=1, no_action_budget=False):
        risk = self.fact_risk(fact, now, conflict)
        if unresolved_task_nodes < 0:
            raise ValueError("unresolved_task_nodes must be nonnegative")
        candidates = {}
        for candidate in action_costs:
            if isinstance(candidate, ActionCost) and candidate.action in self.ACTIONS:
                candidates.setdefault(candidate.action, []).append(candidate)
        scores = {"continue": risk}
        selected_candidates = {}
        for name, alternatives in candidates.items():
            if name == "continue":
                continue
            scored = []
            for candidate in alternatives:
                if not candidate.feasible:
                    continue
                movement = float(candidate.movement_s)
                waiting = float(candidate.wait_s)
                query = float(candidate.query_cost)
                if (not all(math.isfinite(v) and v >= 0
                            for v in (movement, waiting, query))):
                    raise ValueError("action costs must be finite and nonnegative")
                residual = self._unit(candidate.residual_risk_fraction,
                                      "residual_risk_fraction")
                if name == "inspect":
                    score = risk * residual + self.time_cost_per_s * movement + \
                        self.query_cost_weight * query
                elif name == "wait":
                    score = risk * residual + self.time_cost_per_s * waiting
                elif name == "local_repair":
                    nodes = max(0, int(candidate.affected_nodes))
                    score = risk * residual + self.repair_node_cost * nodes + \
                        self.time_cost_per_s * movement + \
                        self.query_cost_weight * query
                elif name == "report_uncertain":
                    score = self.stop_cost * max(1, unresolved_task_nodes)
                else:
                    continue
                scored.append((score, candidate))
            if scored:
                scores[name], selected_candidates[name] = min(
                    scored, key=lambda pair: (pair[0], pair[1].mode,
                                               pair[1].viewpoint))
        if no_action_budget:
            scores.pop("inspect", None)
        # Stable ordering makes policy replay deterministic under ties.
        rank = {"continue": 0, "inspect": 1, "wait": 2,
                "local_repair": 3, "report_uncertain": 4}
        action = min(scores, key=lambda name: (scores[name], rank[name]))
        selected = selected_candidates.get(action)
        rationale = ("memory_risk_below_alternative_cost" if action == "continue"
                     else "minimum_expected_task_and_action_cost")
        return Decision(action, fact.fact_id, risk, scores[action], scores,
                        rationale,
                        selected.mode if selected else "NONE",
                        selected.viewpoint if selected else ())


class PersistentTaskMemory:
    """Small JSON store for cross-episode facts, keyed by registered scope.

    Dynamic facts are retained as historical evidence and aged by the agent;
    they are not deleted or rewritten as absent merely because they are old.
    """

    def __init__(self, facts=()):
        self._facts = {}
        for fact in facts:
            self.upsert(fact)

    @staticmethod
    def _key(fact):
        return (fact.map_id, fact.region_id, fact.predicate, fact.fact_id)

    def upsert(self, fact):
        if not isinstance(fact, TaskFact):
            raise TypeError("fact must be TaskFact")
        TaskMemoryAgent._unit(fact.reliability, "reliability")
        TaskMemoryAgent._unit(fact.impact, "impact")
        if not math.isfinite(fact.captured_at):
            raise ValueError("captured_at must be finite")
        self._facts[self._key(fact)] = fact

    def retrieve(self, map_id, region_id, predicates=None):
        allowed = None if predicates is None else set(predicates)
        return tuple(sorted(
            (fact for fact in self._facts.values()
             if fact.map_id == map_id and fact.region_id == region_id and
             (allowed is None or fact.predicate in allowed)),
            key=lambda fact: (fact.predicate, fact.fact_id)))

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = [fact.__dict__ for fact in sorted(
            self._facts.values(), key=self._key)]
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.",
                                        dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"schema": "ranger_task_memory_v1", "facts": payload},
                          stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "ranger_task_memory_v1":
            raise ValueError("unsupported task-memory schema")
        facts = []
        for item in payload.get("facts", []):
            facts.append(TaskFact(**item))
        return cls(facts)


def action_costs_from_view_candidates(candidates, query_cost=0.0,
                                      evidence_gain_scale=1.0):
    """Convert evaluated feasible viewpoint/mode candidates to inspect costs.

    Expected inputs have ``viewpoint``, ``mode``, ``movement_s``, and
    ``observation_gain`` fields. The caller should derive them from sensor
    geometry and the existing Ranger rollout planner, never simulator labels.
    """
    if not math.isfinite(query_cost) or query_cost < 0:
        raise ValueError("query_cost must be finite and nonnegative")
    if not math.isfinite(evidence_gain_scale) or evidence_gain_scale < 0:
        raise ValueError("evidence_gain_scale must be finite and nonnegative")
    result = []
    for item in candidates:
        gain = max(0.0, min(1.0,
                           float(item.get("observation_gain", 0.0)) *
                           evidence_gain_scale))
        result.append(ActionCost(
            action="inspect", movement_s=float(item["movement_s"]),
            query_cost=query_cost,
            residual_risk_fraction=1.0-gain,
            feasible=bool(item.get("feasible", True)),
            mode=str(item.get("mode", "UNKNOWN")),
            viewpoint=tuple(item.get("viewpoint", ())),
        ))
    return result
