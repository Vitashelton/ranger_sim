#!/usr/bin/env python3
"""Matched offline A/B/C fixture for the actual dynamic facility region IDs.

Intentionally does not call ROS 2, Gazebo, a camera or a remote LLM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rangermini_dynamic_semantic.grounded_mission_core import (
    BoundedTaskCompiler, GroundedMissionEngine, PlaceCatalog, SQLiteMissionLog,
)


def graph_snapshot(cfg, stamp=0):
    return {"revision": 1, "timestamp": stamp,
            "place_nodes": [{"id": p} for p in cfg["regions"]],
            "semantic_nodes": [],
            "edges": [{"from": e["from"], "to": e["to"],
                       "state": "FREE", "base_cost": e["cost"],
                       "effective_cost": e["cost"]}
                      for e in cfg["topology_edges"]]}


def replay(path: Path | None = None) -> dict:
    catalog = PlaceCatalog.load(ROOT / "config/grounded_places_dynamic.json")
    cfg = yaml.safe_load((ROOT / "config/dynamic_benchmark.yaml").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        log = SQLiteMissionLog(Path(tmp) / "events.sqlite3")
        agent = GroundedMissionEngine(catalog, log)
        agent.update_graph(graph_snapshot(cfg))
        agent.take_effects()
        compiler = BoundedTaskCompiler(catalog)
        itinerary = compiler.parse_short_visit("先去检修区，再去仓储区")
        agent.start(itinerary, mission_id="ABC_dynamic_001")
        initial_effects = agent.take_effects()
        assert len([e for e in initial_effects if e["action"] == "SEND_GOAL"]) == 1
        first_goal = agent.active_goal["goal_id"]
        agent.navigation_result(goal_id=first_goal, status="SUCCEEDED", stamp=10.0)
        started_storage_effects = agent.take_effects()
        storage_goal = agent.active_goal["goal_id"]
        assert agent.active_goal["target_region"] == "storage"

        # While moving to storage, the user corrects the remaining suffix.
        agent.revise_remaining(compiler.parse_short_visit("不对，最后去办公区"), now=12.0)
        cancel_request = agent.take_effects()
        assert [x["action"] for x in cancel_request] == ["CANCEL_GOAL"]
        agent.cancel_ack(goal_id=storage_goal, accepted=True)
        after_ack_effects = agent.take_effects()
        assert after_ack_effects == []  # ack is NOT terminal
        agent.navigation_result(goal_id=storage_goal, status="CANCELED", stamp=13.0)
        new_effects = agent.take_effects()
        assert any(x["action"] == "SEND_GOAL" and x["target_region"] == "office"
                   for x in new_effects)
        office_goal = agent.active_goal["goal_id"]
        agent.navigation_result(goal_id=office_goal, status="SUCCEEDED", stamp=25.0)
        final_effects = agent.take_effects()
        answers = agent.answers_from_events()
        assert answers["visited_regions"] == ["inspection", "office"]
        assert not answers["pending_steps"]
        event_rows = log.events("ABC_dynamic_001")

        # Separate fixture: the Ranger can navigate, NOT physically manipulate.
        human_plan = compiler.compile({
            "type": "human_assisted_service", "source": "office",
            "service": "receiving", "destination": "office",
            "object_category": "parcel_red"})
        agent.start(human_plan, mission_id="HUMAN_SERVICE_dynamic_001")
        human_effects = agent.take_effects()
        clock = 30.0
        while not agent.status()["finished"]:
            step = agent.current()
            if step.op == "NAVIGATE":
                gid = agent.active_goal["goal_id"]
                clock += 4
                agent.navigation_result(goal_id=gid, status="SUCCEEDED", stamp=clock)
            else:
                agent.human_ack(action=step.target, operator_id="demo_operator",
                                note="Synthetic human acknowledgement in offline fixture", now=clock)
            agent.take_effects()
        human_answer = agent.answers_from_events()
        assert human_answer["robot_manipulation"] == []
        assert len(human_answer["human_actions"]) == 3
        output = {
            "provenance": {"simulation": "SCRIPTED_EVENT_REPLAY_ONLY",
                           "ran_Gazebo": False, "ran_ROS2": False,
                           "called_remote_LLM": False,
                           "used_real_RGBD": False,
                           "scene": "dynamic_indoor_benchmark.sdf"},
            "scene_A": {"input": "先去检修区，再去仓储区",
                        "compiled": [s.__dict__ for s in itinerary],
                        "first_goal": first_goal,
                        "after_first_arrival": started_storage_effects},
            "scene_B": {"input": "不对，最后去办公区",
                        "cancel_request": cancel_request,
                        "after_cancel_ack": after_ack_effects,
                        "replacement_effects_after_terminal": new_effects,
                        "after_final_arrival": final_effects},
            "scene_C": {"question": "刚才去过哪里，还有什么任务？",
                        "answer_from_sqlite_events": answers},
            "human_assisted_contract": {
                "compiled": [s.__dict__ for s in human_plan],
                "initial_effects": human_effects,
                "answer_from_sqlite_events": human_answer},
            "verified_event_log": event_rows,
        }
        log.close()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("/tmp/grounded_abc_replay.json"))
    args = p.parse_args()
    result = replay(args.out)
    print("A: reached inspection, moving to storage")
    print("B: cancel storage ACK did not dispatch new goal; terminal CANCELED released office goal")
    print("C: visited", result["scene_C"]["answer_from_sqlite_events"]["visited_regions"])
    print("Human-assisted action log:", result["human_assisted_contract"]["answer_from_sqlite_events"]["human_actions"])
    print("No robot manipulation logged. Saved:", args.out)


if __name__ == "__main__":
    main()
