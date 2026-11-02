"""Versioned evidence-grounded missions for the existing dynamic semantic graph.

This ROS-free core deliberately does not consume Gazebo ground truth.  The only
inputs are place metadata, semantic observations, graph snapshots, navigation
results, and explicit human acknowledgements.

Publication caveat: this is a proposed engineering method, not proof of novelty
or a pretrained embodied model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import heapq
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Optional

from .semantic_state_consistency import SemanticCommitGate


class MissionError(ValueError):
    pass


VALID_OPS = {"NAVIGATE", "VERIFY_OBJECT", "WAIT_HUMAN", "SEARCH_OBJECT"}
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED", "ABORTED", "NO_ROUTE", "REJECTED"}


@dataclass
class Step:
    op: str
    target: str
    condition: str
    status: str = "PENDING"


class PlaceCatalog:
    """Simulated semantic identity/prior catalog, not an observed object map."""

    def __init__(self, document: dict):
        if document.get("schema") != "rangermini.places.v1":
            raise MissionError("invalid place-catalog schema")
        self.mode = document.get("scene")
        self.places: dict[str, dict] = {}
        for place in document.get("places", []):
            ident = place.get("id", "")
            if not isinstance(ident, str) or not ident or ident in self.places:
                raise MissionError("duplicate or missing place identity")
            for key in ("name", "side", "door_sign", "features", "neighbors", "object_priors"):
                if key not in place:
                    raise MissionError("place missing " + key)
            for value in place["object_priors"].values():
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    raise MissionError("invalid object prior")
            self.places[ident] = copy.deepcopy(place)
        if not self.places:
            raise MissionError("catalog must contain at least one place")
        for p in self.places.values():
            if any(neighbor not in self.places for neighbor in p["neighbors"]):
                raise MissionError("unknown neighbor")

    @classmethod
    def load(cls, path: str | Path) -> "PlaceCatalog":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def resolve(self, term: str) -> str:
        term = term.strip()
        if not term:
            raise MissionError("empty place name")
        matches = {ident for ident, place in self.places.items()
                   if term == ident or term == place["name"]
                   or term in place.get("aliases", [])}
        if len(matches) == 1:
            return matches.pop()
        raise MissionError("missing/ambiguous place: " + term)

    def prior(self, place: str, category: str) -> float:
        return float(self.places[place]["object_priors"].get(category, 0.02))


class BoundedTaskCompiler:
    """Validate LLM-produced JSON or restricted human speech into task steps."""

    def __init__(self, catalog: PlaceCatalog):
        self.catalog = catalog

    def compile(self, request: dict) -> list[Step]:
        if not isinstance(request, dict):
            raise MissionError("LLM output must be a JSON object")
        forbidden = {"cmd_vel", "coordinates", "velocity", "robot_pickup",
                     "robot_fill", "robot_deliver", "motor_command"}
        if forbidden.intersection(request):
            raise MissionError("LLM must not issue chassis or manipulator commands")
        kind = request.get("type")
        steps: list[Step] = []
        if kind == "visit_sequence":
            places = request.get("places")
            if not isinstance(places, list) or not 1 <= len(places) <= 12:
                raise MissionError("visit sequence requires 1-12 places")
            for place in places:
                resolved = self.catalog.resolve(str(place))
                steps.append(Step("NAVIGATE", resolved, "navigation result SUCCEEDED"))
        elif kind == "human_assisted_service":
            source = self.catalog.resolve(str(request.get("source", "")))
            service = self.catalog.resolve(str(request.get("service", "")))
            destination = self.catalog.resolve(str(request.get("destination", "")))
            object_class = str(request.get("object_category", ""))
            # Actions are human-side contracts, not mobile robot actuations.
            if object_class not in {"cup", "parcel_red", "parcel_blue"}:
                raise MissionError("unknown demonstration object class")
            steps = [
                Step("NAVIGATE", source, "navigation result SUCCEEDED"),
                Step("WAIT_HUMAN", "pickup:" + object_class,
                     "explicit named operator pickup acknowledgement"),
                Step("NAVIGATE", service, "navigation result SUCCEEDED"),
                Step("WAIT_HUMAN", "service:" + object_class,
                     "explicit named operator service acknowledgement"),
                Step("NAVIGATE", destination, "navigation result SUCCEEDED"),
                Step("WAIT_HUMAN", "handoff:" + object_class,
                     "explicit named operator handoff acknowledgement"),
            ]
        elif kind == "objectnav":
            category = str(request.get("category", ""))
            if not category or category not in {
                c for p in self.catalog.places.values()
                for c in p["object_priors"]
            }:
                raise MissionError("object category not in scene catalog")
            steps = [Step("SEARCH_OBJECT", category,
                          "fresh rendered RGB-D observation, <=1m detection")]
        else:
            raise MissionError("unsupported task schema")
        return steps

    def parse_short_visit(self, text: str) -> list[Step]:
        """Offline deterministic fallback. Not called an LLM inference."""
        text = re.sub(r"^(?:不对|等等|不是|改成|改为)[，,：: ]*", "", text.strip())
        text = re.sub(r"^(?:请|麻烦)?(?:带我|帮我)?(?:去|前往|导航到)", "", text)
        chunks = re.split(r"(?:先去|然后去|再去|最后去|接着去|、|，|,|→|->|到)", text)
        places = [c.strip("。 ；;。") for c in chunks if c.strip("。 ；;。")]
        if not places:
            raise MissionError("could not parse visit sequence")
        return self.compile({"type": "visit_sequence", "places": places})


class EvidenceRouteSelector:
    """Task-conditioned routing with temporal object evidence and edge costs.

    Relative scoring is a transparent experimental method; LLM candidates may
    be accepted only if present in the returned reachable list.
    """

    def __init__(self, catalog: PlaceCatalog):
        self.catalog = catalog

    def route_costs(self, graph: dict, start: str) -> dict[str, float]:
        adjacency = {p: [] for p in self.catalog.places}
        for edge in graph.get("edges", []):
            a, b = edge.get("from"), edge.get("to")
            if a not in adjacency or b not in adjacency:
                continue
            if edge.get("state") == "TEMP_BLOCKED":
                continue
            cost = edge.get("effective_cost", edge.get("base_cost", edge.get("cost", 1)))
            if not isinstance(cost, (float, int)) or not math.isfinite(cost) or cost <= 0:
                continue
            adjacency[a].append((b, float(cost)))
            adjacency[b].append((a, float(cost)))
        if start not in adjacency:
            return {}
        distances = {start: 0.0}
        heap = [(0.0, start)]
        while heap:
            cost, place = heapq.heappop(heap)
            if cost > distances[place] + 1e-9:
                continue
            for neighbor, edge_cost in adjacency[place]:
                alternative = cost + edge_cost
                if alternative < distances.get(neighbor, float("inf")):
                    distances[neighbor] = alternative
                    heapq.heappush(heap, (alternative, neighbor))
        return distances

    def ranked(self, graph: dict, current: str, category: str,
               visited: set[str], now: float, *, mode="full") -> list[dict]:
        if mode not in ("full", "no_memory", "no_risk", "prior_only"):
            raise MissionError("unknown selector ablation")
        costs = self.route_costs(graph, current)
        if mode == "no_risk":
            relaxed = copy.deepcopy(graph)
            for edge in relaxed.get("edges", []):
                edge["state"] = "FREE"
                edge["effective_cost"] = edge.get("base_cost", edge.get("cost", 1))
            costs = self.route_costs(relaxed, current)
        observations = {p: 0.0 for p in self.catalog.places}
        if mode not in ("no_memory", "prior_only"):
            for node in graph.get("semantic_nodes", []):
                if node.get("category") != category or node.get("region") not in observations:
                    continue
                if node.get("state") not in ("CONFIRMED", "UNCERTAIN"):
                    continue
                try:
                    confidence = float(node.get("confidence", 0))
                    seen = float(node.get("last_seen", -1e9))
                except (ValueError, TypeError):
                    continue
                if not math.isfinite(confidence) or not math.isfinite(seen) or seen > now + 1:
                    continue
                # State and age gate: do not treat stale observations as current truth.
                freshness = math.exp(-max(0.0, now - seen) / 30.0)
                observations[node["region"]] = max(
                    observations[node["region"]], max(0.0, min(1.0, confidence)) * freshness)
        scored = []
        for place in sorted(self.catalog.places):
            if place in visited or place not in costs:
                continue
            travel = costs[place]
            prior = self.catalog.prior(place, category)
            evidence = observations[place]
            # Prior helps ranking, but never declares a target found.
            if mode == "prior_only":
                utility = prior - 0.012 * travel
            else:
                utility = 0.75 * prior + 1.55 * evidence + 0.18 * (1.0 - prior) - 0.012 * travel
            scored.append({"region": place, "utility": round(utility, 6),
                           "prior": prior, "evidence": round(evidence, 5),
                           "route_cost": round(travel, 3),
                           "graph_revision": int(graph.get("revision", 0))})
        scored.sort(key=lambda v: (-v["utility"], v["route_cost"], v["region"]))
        return scored

    @staticmethod
    def gated_llm_choice(candidate: dict, allowed: list[dict], revision: int) -> str:
        if set(candidate) != {"action", "target_region", "graph_revision"}:
            raise MissionError("LLM proposed unexpected fields")
        if candidate["action"] != "SELECT_REGION" or candidate["graph_revision"] != revision:
            raise MissionError("LLM action or graph revision invalid/stale")
        authorized = {c["region"] for c in allowed}
        if candidate["target_region"] not in authorized:
            raise MissionError("LLM chose an unreachable/unlisted region")
        return candidate["target_region"]


class SQLiteMissionLog:
    """Atomic state + event journal; only event facts can substantiate answers."""

    def __init__(self, filename: str | Path):
        self.conn = sqlite3.connect(str(filename))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                          "mission TEXT NOT NULL, kind TEXT NOT NULL, data TEXT NOT NULL)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS snapshots (name TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self.conn.commit()

    def append(self, mission: str, kind: str, data: dict, state: dict) -> None:
        with self.conn:
            self.conn.execute("INSERT INTO events(mission,kind,data) VALUES (?,?,?)",
                              (mission, kind, json.dumps(data, ensure_ascii=False)))
            self.conn.execute("INSERT OR REPLACE INTO snapshots(name,data) VALUES (?,?)",
                              ("engine", json.dumps(state, ensure_ascii=False)))

    def state(self) -> Optional[dict]:
        row = self.conn.execute("SELECT data FROM snapshots WHERE name='engine'").fetchone()
        return json.loads(row[0]) if row else None

    def events(self, mission: str) -> list[dict]:
        rows = self.conn.execute("SELECT seq,kind,data FROM events WHERE mission=? ORDER BY seq",
                                 (mission,)).fetchall()
        return [{"sequence": row[0], "kind": row[1], **json.loads(row[2])} for row in rows]

    def close(self) -> None:
        self.conn.close()


class GroundedMissionEngine:
    """Deterministic controller with explicit effects dispatched by a ROS adapter.

    A cancellation request/acknowledgement is not a terminal state.  The next
    target is published *only after* the previous goal terminal result arrives.
    """

    def __init__(self, catalog: PlaceCatalog, ledger: SQLiteMissionLog,
                 *, selector_mode: str = "full"):
        self.catalog, self.log = catalog, ledger
        self.selector = EvidenceRouteSelector(catalog)
        self.selector_mode = selector_mode
        self.steps: list[Step] = []
        self.index = 0
        self.mission_id = ""
        self.seq = 0
        self.revision = 0
        self.active_goal: Optional[dict] = None
        self.pending_revision: Optional[list[Step]] = None
        self.waiting_cancel = False
        self.current_region = "lobby" if "lobby" in catalog.places else next(iter(catalog.places))
        self.last_graph: dict = {}
        self.visited_search: set[str] = set()
        self.seen_observations: set[tuple] = set()
        self.positive_hits: list[tuple] = []  # fresh independent camera frames
        self.last_arrival_stamp = -1.0
        self.blocked = ""
        self.effects: list[dict] = []
        self.semantic_gate = SemanticCommitGate()
        previous = self.log.state()
        if previous:
            self._restore(previous)
            if self.active_goal or self.pending_revision is not None:
                self.blocked = "INFLIGHT_STATE_REQUIRES_RECONCILIATION"

    def _snapshot(self) -> dict:
        return {"steps": [asdict(s) for s in self.steps], "index": self.index,
                "mission_id": self.mission_id, "seq": self.seq, "revision": self.revision,
                "active_goal": self.active_goal,
                "pending_revision": [asdict(s) for s in self.pending_revision]
                if self.pending_revision is not None else None,
                "waiting_cancel": self.waiting_cancel, "current_region": self.current_region,
                "visited_search": sorted(self.visited_search),
                "last_arrival_stamp": self.last_arrival_stamp, "blocked": self.blocked}

    def _restore(self, previous: dict):
        self.steps = [Step(**step) for step in previous["steps"]]
        self.index = previous["index"]
        self.mission_id = previous["mission_id"]
        self.seq = previous["seq"]
        self.revision = previous["revision"]
        self.active_goal = previous["active_goal"]
        pending = previous["pending_revision"]
        self.pending_revision = [Step(**step) for step in pending] if pending is not None else None
        self.waiting_cancel = previous["waiting_cancel"]
        self.current_region = previous["current_region"]
        self.visited_search = set(previous["visited_search"])
        self.last_arrival_stamp = previous["last_arrival_stamp"]
        self.blocked = previous["blocked"]

    def _record(self, kind: str, **data):
        self.log.append(self.mission_id, kind, data, self._snapshot())

    def _effect(self, action: str, **data):
        self.effects.append({"action": action, "mission_id": self.mission_id, **data})

    def take_effects(self) -> list[dict]:
        result = list(self.effects)
        self.effects.clear()
        return result

    def current(self) -> Optional[Step]:
        return self.steps[self.index] if self.index < len(self.steps) else None

    def status(self) -> dict:
        return {"mission_id": self.mission_id, "plan_revision": self.revision,
                "current_index": self.index, "total_steps": len(self.steps),
                "current_step": asdict(self.current()) if self.current() else None,
                "active_goal": copy.deepcopy(self.active_goal),
                "waiting_cancel_terminal": self.waiting_cancel,
                "blocked": self.blocked,
                "finished": bool(self.steps) and self.index == len(self.steps)}

    def update_graph(self, graph: dict):
        if not isinstance(graph, dict) or "place_nodes" not in graph or "edges" not in graph:
            raise MissionError("invalid online graph snapshot")
        # Ground-truth topics are intentionally not part of this schema.
        if any(k in graph for k in ("ground_truth", "truth_entities", "oracle_poses")):
            raise MissionError("oracle content forbidden in online graph")
        self.last_graph = copy.deepcopy(graph)
        if not self.active_goal and not self.blocked and self.current() and self.current().status == "PENDING":
            self.dispatch(now=float(graph.get("timestamp", 0)))

    def capture_semantic_context(self, *, request_id: str, read_fact_versions: dict):
        """Bind the exact outgoing semantic memory snapshot to current ownership."""
        if self.current() is None or self.blocked:
            raise MissionError("no executable task for semantic decision")
        context = self.semantic_gate.capture(
            request_id=request_id, mission_id=self.mission_id,
            task_revision=self.revision, task_index=self.index,
            goal_id=self.active_goal["goal_id"] if self.active_goal else None,
            read_fact_versions=read_fact_versions)
        self._record("SEMANTIC_REQUEST_CONTEXT_CAPTURED", context=asdict(context))
        return context

    def commit_semantic_revision(self, context, steps: list[Step], *,
                                 current_fact_versions: dict, now: float):
        """Validate an async proposal before invoking the existing revision path.

        A valid context is necessary, not sufficient: revise_remaining still
        validates operations, destinations, and execution state. Navigation
        preemption/cancellation is delegated to the existing adapter.
        """
        accepted, reason = self.semantic_gate.validate(
            context, mission_id=self.mission_id, task_revision=self.revision,
            task_index=self.index,
            goal_id=self.active_goal["goal_id"] if self.active_goal else None,
            current_fact_versions=current_fact_versions,
            update_pending=self.pending_revision is not None or self.waiting_cancel)
        if accepted:
            try:
                self.revise_remaining(steps, now=now)
            except MissionError as exc:
                accepted, reason = False, "proposal_not_executable: " + str(exc)
            else:
                self.semantic_gate.consume(context)
        self._record("SEMANTIC_REVISION_ACCEPTED" if accepted else "SEMANTIC_REVISION_REJECTED",
                     request_id=getattr(context, "request_id", None), reason=reason, at=now)
        return accepted, reason

    def start(self, steps: list[Step], *, mission_id: str):
        if not mission_id or not re.fullmatch(r"[a-zA-Z0-9_.-]{1,90}", mission_id):
            raise MissionError("invalid mission ID")
        if self.active_goal or (self.current() is not None and not self.status()["finished"]):
            raise MissionError("active mission cannot be overwritten")
        if not steps or len(steps) > 24 or any(s.op not in VALID_OPS for s in steps):
            raise MissionError("unsupported mission")
        for s in steps:
            if s.op == "NAVIGATE" and s.target not in self.catalog.places:
                raise MissionError("nonexistent destination")
        self.mission_id = mission_id
        self.steps = copy.deepcopy(steps)
        self.index, self.seq, self.revision = 0, 0, 1
        self.active_goal, self.pending_revision = None, None
        self.waiting_cancel = False
        self.visited_search.clear()
        self.last_arrival_stamp = -1
        self.seen_observations.clear()
        self.positive_hits.clear()
        self.blocked = ""
        self._record("MISSION_COMPILED", plan=[asdict(s) for s in steps])
        self.dispatch(now=float(self.last_graph.get("timestamp", 0)))

    def _nav_goal(self, region: str, *, search_category="", now=0.0):
        self.seq += 1
        self.positive_hits.clear()
        self.active_goal = {"goal_id": f"{self.mission_id}.g{self.seq}",
                            "target_region": region,
                            "search_category": search_category,
                            "plan_revision": self.revision}
        self.current().status = "NAVIGATING"
        self._record("NAV_GOAL_REQUESTED", **self.active_goal, at=now)
        self._effect("SEND_GOAL", **self.active_goal)
        self._effect("SET_TASK_CONTEXT", category=search_category, target_region=region)

    def dispatch(self, *, now: float):
        if self.blocked or self.active_goal or self.waiting_cancel or self.pending_revision is not None:
            return
        step = self.current()
        if step is None:
            self._record("MISSION_FINISHED", at=now)
            self._effect("MISSION_FINISHED")
            return
        if step.op == "NAVIGATE":
            self._nav_goal(step.target, now=now)
        elif step.op == "WAIT_HUMAN":
            if step.status == "PENDING":
                step.status = "WAITING_HUMAN"
                self._record("WAITING_HUMAN", action=step.target)
        elif step.op == "VERIFY_OBJECT":
            if step.status == "PENDING":
                step.status = "WAITING_SENSOR_EVIDENCE"
                self._record("WAITING_OBJECT", category=step.target)
                self._effect("REQUEST_PERCEPTION", category=step.target)
        elif step.op == "SEARCH_OBJECT":
            if not self.last_graph:
                self._record("AWAIT_GRAPH", category=step.target)
                return
            candidates = self.selector.ranked(
                self.last_graph, self.current_region, step.target,
                self.visited_search, now, mode=self.selector_mode)
            if not candidates:
                self.blocked = "OBJECT_SEARCH_NO_REACHABLE_UNVISITED_REGION"
                self._record("OBJECT_SEARCH_BLOCKED", category=step.target)
                return
            self._record("OBJECT_SEARCH_RANKED", target=step.target,
                         graph_revision=self.last_graph.get("revision"),
                         candidates=candidates)
            self._nav_goal(candidates[0]["region"], search_category=step.target, now=now)

    def navigation_result(self, *, goal_id: str, status: str, stamp: float):
        if status not in TERMINAL:
            raise MissionError("nonterminal navigation result")
        if not self.active_goal or self.active_goal["goal_id"] != goal_id:
            self._record("STALE_NAV_RESULT", goal_id=goal_id, status=status)
            return
        goal = self.active_goal
        self.active_goal = None
        step = self.current()
        was_canceling = self.waiting_cancel
        self.waiting_cancel = False
        self._record("NAV_GOAL_TERMINAL", goal_id=goal_id,
                     target_region=goal["target_region"], status=status,
                     at=stamp, cancel_race=was_canceling)
        if status == "SUCCEEDED":
            self.current_region = goal["target_region"]
            self.last_arrival_stamp = stamp
            self._record("NAV_REGION_ARRIVED", region=self.current_region, goal_id=goal_id)
        if self.pending_revision is not None:
            # If an action succeeded during cancellation, its physical arrival
            # remains a real fact; but the future plan must be the new suffix.
            if status == "SUCCEEDED" and step.op == "NAVIGATE":
                step.status = "COMPLETED"
                self._record("STEP_COMPLETED", op=step.op, target=goal["target_region"],
                             source="goal_success_during_replan")
                self.index += 1
            else:
                step.status = "SUPERSEDED"
                self._record("STEP_SUPERSEDED", op=step.op, target=step.target,
                             navigation_target=goal["target_region"], status=status,
                             semantic_completion_claimed=False)
            self.steps = self.steps[:self.index] + self.pending_revision
            self.pending_revision = None
            self.revision += 1
            self._record("PLAN_REVISION_COMMITTED", revision=self.revision,
                         remaining=[asdict(s) for s in self.steps[self.index:]])
            self.dispatch(now=stamp)
            return
        if status != "SUCCEEDED":
            self.blocked = "NAV_" + status
            self._record("MISSION_BLOCKED", reason=self.blocked)
            return
        if step.op == "SEARCH_OBJECT":
            self.visited_search.add(self.current_region)
            step.status = "WAITING_SENSOR_EVIDENCE"
            self.positive_hits.clear()
            self._record("AWAIT_OBJECT_AT_REGION", category=step.target,
                         region=self.current_region)
            self._effect("REQUEST_PERCEPTION", category=step.target)
        else:
            self._complete_step("nav_success", stamp)

    def _complete_step(self, source: str, stamp: float):
        step = self.current()
        if not step:
            return
        step.status = "COMPLETED"
        self._record("STEP_COMPLETED", op=step.op, target=step.target, source=source)
        self.index += 1
        self.dispatch(now=stamp)

    def observe(self, observation: dict, *, now: float) -> bool:
        if observation.get("uses_sim_ground_truth") is not False:
            raise MissionError("cannot confirm targets using sim oracle data")
        if observation.get("source") not in ("rendered_rgbd", "robot_rgbd"):
            raise MissionError("unknown sensor evidence origin")
        obs_id = (observation.get("source"), observation.get("observation_id"))
        if obs_id[1] is None or obs_id in self.seen_observations:
            return False
        self.seen_observations.add(obs_id)
        if len(self.seen_observations) > 5000:
            self.seen_observations = set(list(self.seen_observations)[-3000:])
        step = self.current()
        if step is None or step.op not in ("VERIFY_OBJECT", "SEARCH_OBJECT"):
            return False
        if step.status != "WAITING_SENSOR_EVIDENCE":
            return False
        try:
            stamp = float(observation.get("sensor_stamp", -1))
        except (TypeError, ValueError):
            return False
        if stamp < self.last_arrival_stamp or stamp > now + 0.5 or now - stamp > 2.0:
            return False
        for hit in observation.get("detections", []):
            if hit.get("category") != step.target:
                continue
            try:
                conf = float(hit.get("confidence", 0))
                range_m = float(hit.get("range_m", math.inf))
            except (ValueError, TypeError):
                continue
            if 0.7 <= conf <= 1.0 and 0.15 < range_m <= 1.0:
                point = hit.get("position", {})
                try:
                    px, py = float(point["x"]), float(point["y"])
                except (ValueError, KeyError, TypeError):
                    continue
                if not math.isfinite(px) or not math.isfinite(py):
                    continue
                self.positive_hits = [x for x in self.positive_hits
                                      if now - x[0] <= 2.5
                                      and math.hypot(px - x[1], py - x[2]) <= 0.6]
                self.positive_hits.append((now, px, py, obs_id))
                self._record("OBJECT_CANDIDATE_SEEN", category=step.target,
                             observation_id=obs_id[1], range_m=range_m,
                             independent_frames=len({x[3] for x in self.positive_hits}))
                if len({x[3] for x in self.positive_hits}) < 2:
                    return False
                self._record("OBJECT_CONFIRMED_BY_SENSOR", category=step.target,
                             region=self.current_region,
                             observation_id=obs_id[1], range_m=range_m,
                             independent_frames=len({x[3] for x in self.positive_hits}),
                             confidence=conf, at=now)
                self.positive_hits.clear()
                self._complete_step("two_fresh_rgbd_views_near_stop", now)
                return True
        return False

    def continue_search(self, *, now: float):
        step = self.current()
        if not step or step.op != "SEARCH_OBJECT" or step.status != "WAITING_SENSOR_EVIDENCE":
            raise MissionError("not waiting for object search evidence")
        # A timeout is *not* an observed negative, hence no false absence update.
        self.positive_hits.clear()
        self._record("OBJECT_NOT_VERIFIED", category=step.target, region=self.current_region,
                     negative_evidence_claimed=False)
        step.status = "PENDING"
        self.dispatch(now=now)

    def human_ack(self, *, action: str, operator_id: str, note: str, now: float):
        step = self.current()
        if step is None or step.op != "WAIT_HUMAN" or step.status != "WAITING_HUMAN":
            raise MissionError("no human operation pending")
        if action != step.target or not operator_id.strip() or not note.strip():
            raise MissionError("invalid operator confirmation")
        if operator_id.strip().lower() in ("robot", "llm", "agent"):
            raise MissionError("robot/LLM cannot certify human manipulation")
        self._record("HUMAN_ACTION_ACKNOWLEDGED", action=action,
                     operator_id=operator_id, note=note,
                     robot_actuated_manipulation=False, at=now)
        self._complete_step("human_ack", now)

    def revise_remaining(self, steps: list[Step], *, now: float):
        if not self.steps or self.status()["finished"]:
            raise MissionError("no active mission to revise")
        if self.blocked or self.pending_revision is not None or len(steps) > 12:
            raise MissionError("unsafe or duplicate revision")
        if any(s.op != "NAVIGATE" or s.target not in self.catalog.places for s in steps):
            raise MissionError("revisions accept only safe visit suffixes")
        if not steps:
            raise MissionError("empty replacement plan")
        self.pending_revision = copy.deepcopy(steps)
        self._record("PLAN_REVISION_REQUESTED", new_steps=[asdict(s) for s in steps], at=now)
        if self.active_goal:
            self.waiting_cancel = True
            self._record("NAV_CANCEL_REQUESTED", goal_id=self.active_goal["goal_id"])
            self._effect("CANCEL_GOAL", goal_id=self.active_goal["goal_id"])
        else:
            old = self.current()
            if old:
                old.status = "SUPERSEDED"
                self._record("STEP_SUPERSEDED", target=old.target, status="NO_ACTIVE_GOAL")
            self.steps = self.steps[:self.index] + self.pending_revision
            self.pending_revision = None
            self.revision += 1
            self._record("PLAN_REVISION_COMMITTED", revision=self.revision)
            self.dispatch(now=now)

    def cancel_ack(self, *, goal_id: str, accepted: bool):
        if self.active_goal and goal_id == self.active_goal["goal_id"]:
            self._record("NAV_CANCEL_ACKNOWLEDGED", goal_id=goal_id, accepted=accepted)
            # Crucially, ack alone must not unlock publication of a new goal.

    def answers_from_events(self) -> dict:
        events = self.log.events(self.mission_id)
        return {
            "visited_regions": [e["region"] for e in events if e["kind"] == "NAV_REGION_ARRIVED"],
            "confirmed_objects": [e["category"] for e in events
                                  if e["kind"] == "OBJECT_CONFIRMED_BY_SENSOR"],
            "human_actions": [e["action"] for e in events
                              if e["kind"] == "HUMAN_ACTION_ACKNOWLEDGED"],
            "robot_manipulation": [],
            "pending_steps": [asdict(s) for s in self.steps[self.index:]],
        }
