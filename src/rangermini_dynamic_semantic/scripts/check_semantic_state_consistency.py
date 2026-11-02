#!/usr/bin/env python3
"""Replay cross-layer consistency cases; no Gazebo or navigation backend."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from rangermini_dynamic_semantic.grounded_mission_core import (
    GroundedMissionEngine, PlaceCatalog, SQLiteMissionLog, Step)
from rangermini_dynamic_semantic.semantic_state_consistency import SemanticCommitGate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gate = SemanticCommitGate()
    captured = dict(request_id="vision_1", mission_id="assist_1", task_revision=1,
                    task_index=0, goal_id="goal_1", read_fact_versions={"target_location": 4})
    context = gate.capture(**captured)
    current = dict(mission_id="assist_1", task_revision=1, task_index=0,
                   goal_id="goal_1", current_fact_versions={"target_location": 4})
    checks = []

    def check(name, state, expected, reason, proposal=context):
        result = gate.validate(proposal, **state)
        assert result == (expected, reason), (name, result)
        checks.append({"name": name, "accepted": result[0], "reason": result[1]})

    check("unchanged_context", current, True, "snapshot_valid")
    check("unrelated_memory_update", dict(current, current_fact_versions={
        "target_location": 4, "distant_chair": 99}), True, "snapshot_valid")
    check("relevant_memory_update", dict(current, current_fact_versions={
        "target_location": 5}), False, "dependency_changed")
    check("missing_dependency", dict(current, current_fact_versions={}), False, "dependency_missing")
    check("task_update", dict(current, task_revision=2), False, "task_changed")
    check("task_progress", dict(current, task_index=1), False, "task_changed")
    check("goal_owner_update", dict(current, goal_id="goal_2"), False, "goal_owner_changed")
    check("pending_update", dict(current, update_pending=True), False, "task_update_pending")
    check("tampered_context", current, False, "unissued_or_modified_context",
          replace(context, read_versions=()))
    gate.consume(context)
    check("duplicate_response", current, False, "duplicate_result")

    ids = ["lobby", "lab_614", "lab_615"]
    catalog = PlaceCatalog({"schema": "rangermini.places.v1", "scene": "synthetic_core_replay",
        "places": [{"id": ident, "name": ident, "side": "test", "door_sign": "",
                    "features": [], "neighbors": [v for v in ids if v != ident],
                    "object_priors": {"cup": 1.0 if ident == "lab_614" else 0.01}}
                   for ident in ids]})
    store = SQLiteMissionLog(args.output / "events.sqlite3")
    engine = GroundedMissionEngine(catalog, store)
    engine.update_graph({"place_nodes": ids, "edges": [
        {"from": "lobby", "to": "lab_614", "cost": 1.0},
        {"from": "lab_614", "to": "lab_615", "cost": 1.0}], "timestamp": 0})
    engine.start([Step("NAVIGATE", "lab_614", "navigation success"),
                  Step("SEARCH_OBJECT", "cup", "current sensor evidence")], mission_id="assist_replay")
    first_goal = engine.active_goal["goal_id"]
    engine.take_effects()
    engine.navigation_result(goal_id=first_goal, status="SUCCEEDED", stamp=1.0)
    assert engine.index == 1 and engine.steps[0].status == "COMPLETED"
    search_goal = engine.active_goal["goal_id"]
    engine.take_effects()
    stale = engine.capture_semantic_context(request_id="stale", read_fact_versions={"target_location": 4})
    accepted, reason = engine.commit_semantic_revision(
        stale, [Step("NAVIGATE", "lab_615", "navigation success")],
        current_fact_versions={"target_location": 5}, now=1.1)
    assert not accepted and reason == "dependency_changed"
    assert not engine.take_effects() and engine.pending_revision is None
    fresh = engine.capture_semantic_context(request_id="fresh", read_fact_versions={"target_location": 5})
    accepted, reason = engine.commit_semantic_revision(
        fresh, [Step("NAVIGATE", "lab_615", "navigation success")],
        current_fact_versions={"target_location": 5, "distant_chair": 99}, now=1.2)
    assert accepted
    effects = engine.take_effects()
    assert [e["action"] for e in effects] == ["CANCEL_GOAL"]
    engine.cancel_ack(goal_id=search_goal, accepted=True)
    assert not engine.take_effects(), "cancel acknowledgment released a replacement goal"
    # Physical arrival wins the race with cancellation; semantic search was
    # never verified and must not become a completed service prefix.
    engine.navigation_result(goal_id=search_goal, status="SUCCEEDED", stamp=1.3)
    assert engine.current_region == "lab_614"
    assert engine.index == 1 and engine.steps[0].status == "COMPLETED"
    assert engine.current().op == "NAVIGATE" and engine.current().target == "lab_615"
    assert engine.active_goal["goal_id"] != search_goal
    replacement_effects = engine.take_effects()
    assert sum(e["action"] == "SEND_GOAL" for e in replacement_effects) == 1
    replacement_goal = engine.active_goal["goal_id"]
    engine.navigation_result(goal_id=search_goal, status="SUCCEEDED", stamp=1.4)
    assert engine.active_goal["goal_id"] == replacement_goal and engine.index == 1
    events = store.events(engine.mission_id)
    assert not any(e["kind"] == "STEP_COMPLETED" and e.get("op") == "SEARCH_OBJECT" for e in events)
    assert any(e["kind"] == "STEP_SUPERSEDED" and e.get("op") == "SEARCH_OBJECT" for e in events)
    report = {"evidence_type": "synthetic_event_replay", "gazebo_episode": False,
              "nav2_used": False, "deepseek_api_called": False,
              "context_checks": checks, "cancel_success_race_passed": True,
              "completed_prefix_preserved": True,
              "navigation_arrival_is_not_semantic_completion": True,
              "stale_navigation_result_ignored": True, "events": events,
              "checks_passed": True}
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    store.close()
    print(f"PASS: {len(checks)} semantic-context cases and cross-layer execution race")
    print(f"Evidence: {args.output / 'report.json'} (synthetic replay only)")


if __name__ == "__main__":
    main()
