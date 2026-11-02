"""ROS-independent research contracts and source integration checks."""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import tempfile

import pytest
import yaml

from rangermini_dynamic_semantic.grounded_mission_core import (
    PlaceCatalog, BoundedTaskCompiler, EvidenceRouteSelector,
    GroundedMissionEngine, SQLiteMissionLog, MissionError, Step)
from rangermini_dynamic_semantic.grounded_objectnav_proxy import (
    METHODS, create_episode, run_episode, summary, paired_bootstrap)

ROOT = Path(__file__).resolve().parents[1]
DYNAMIC = PlaceCatalog.load(ROOT / "config/grounded_places_dynamic.json")
CORRIDOR = PlaceCatalog.load(ROOT / "config/grounded_places_corridor.json")
CONFIG = yaml.safe_load((ROOT / "config/dynamic_benchmark.yaml").read_text(encoding="utf-8"))


def graph(*, blocked=None, obj=None):
    edges = []
    for e in CONFIG["topology_edges"]:
        row = dict(e, base_cost=e["cost"], effective_cost=e["cost"], state="FREE")
        if blocked and {e["from"], e["to"]} == set(blocked):
            row["state"] = "TEMP_BLOCKED"
        edges.append(row)
    return {"revision": 3, "timestamp": 100, "place_nodes": [{"id": r} for r in CONFIG["regions"]],
            "edges": edges, "semantic_nodes": [obj] if obj else []}


@pytest.fixture
def engine(tmp_path):
    store = SQLiteMissionLog(tmp_path / "session.sqlite3")
    agent = GroundedMissionEngine(DYNAMIC, store)
    agent.update_graph(graph())
    yield agent
    store.close()


def start_visit(agent, destinations=("inspection", "storage")):
    steps = BoundedTaskCompiler(DYNAMIC).compile({"type": "visit_sequence", "places": list(destinations)})
    agent.start(steps, mission_id="t_r1")
    return agent.take_effects()


def test_catalog_mirrors_actual_dynamic_world_regions_and_topology():
    assert set(DYNAMIC.places) == set(CONFIG["regions"])
    assert len(DYNAMIC.places) == 8
    for p in DYNAMIC.places.values():
        assert p["pose_source"] == "simulated_yaml_region_center"
        assert p["door_sign"] == ""
        assert len(p["features"]) > 0
    assert len(CORRIDOR.places) == 4
    assert CORRIDOR.places["room_906"]["name"] == "茶水间"


@pytest.mark.parametrize("term,expected", [
    ("检修区", "inspection"), ("实验室", "inspection"),
    ("仓库", "storage"), ("办公室", "office")])
def test_dynamic_name_resolution(term, expected):
    assert DYNAMIC.resolve(term) == expected


def test_reject_invalid_catalog_duplicate_and_unknown_neighbor():
    raw = json.loads((ROOT / "config/grounded_places_dynamic.json").read_text())
    raw["places"].append(copy.deepcopy(raw["places"][0]))
    with pytest.raises(MissionError):
        PlaceCatalog(raw)
    raw["places"].pop()
    raw["places"][0]["neighbors"].append("invalid_place")
    with pytest.raises(MissionError):
        PlaceCatalog(raw)


def test_compiler_produces_capability_safe_workflow():
    steps = BoundedTaskCompiler(DYNAMIC).compile({
        "type": "human_assisted_service", "source": "office", "service": "receiving",
        "destination": "office", "object_category": "parcel_red"})
    assert len(steps) == 6
    assert [x.op for x in steps].count("WAIT_HUMAN") == 3
    assert not any(x.op.startswith("ROBOT_") for x in steps)
    assert steps[-1].target == "handoff:parcel_red"


@pytest.mark.parametrize("payload", [
    {"type": "goto", "places": ["office"]},
    {"type": "visit_sequence", "places": []},
    {"type": "visit_sequence", "places": ["fake"]},
    {"type": "visit_sequence", "places": ["office"], "cmd_vel": [2, 2]},
    {"type": "human_assisted_service", "source": "office", "service": "packing",
     "destination": "office", "object_category": "cup", "robot_pickup": True},
    {"type": "objectnav", "category": "rainbow_dragon"},
])
def test_llm_output_cannot_escape_action_catalog(payload):
    with pytest.raises(MissionError):
        BoundedTaskCompiler(DYNAMIC).compile(payload)


def test_offline_parser_handles_user_correction_and_preserves_order():
    compiler = BoundedTaskCompiler(DYNAMIC)
    assert [s.target for s in compiler.parse_short_visit("先去检修区，再去仓储区")] == ["inspection", "storage"]
    assert [s.target for s in compiler.parse_short_visit("不对，最后去办公区")] == ["office"]


def test_route_selector_handles_one_blocked_edge_with_other_route():
    selector = EvidenceRouteSelector(DYNAMIC)
    before = selector.route_costs(graph(), "inspection")
    after = selector.route_costs(graph(blocked={"inspection", "storage"}), "inspection")
    assert math.isfinite(after["storage"])
    assert after["storage"] > before["storage"]


def test_stale_semantic_memory_does_not_dominate_fresh_evidence():
    selector = EvidenceRouteSelector(DYNAMIC)
    state = graph(obj={"category": "parcel_red", "region": "receiving",
                       "confidence": .99, "last_seen": 0, "state": "STALE"})
    stale = selector.ranked(state, "lobby", "parcel_red", set(), 100)
    assert all(x["evidence"] == 0 for x in stale)
    state["semantic_nodes"] = [{"category": "parcel_red", "region": "storage",
                                "confidence": .92, "last_seen": 99, "state": "CONFIRMED"}]
    fresh = selector.ranked(state, "lobby", "parcel_red", set(), 100)
    assert any(x["region"] == "storage" and x["evidence"] > .8 for x in fresh)


@pytest.mark.parametrize("fake", [
    {"action": "SELECT_REGION", "target_region": "fake", "graph_revision": 3},
    {"action": "SEND_VELOCITY", "target_region": "office", "graph_revision": 3},
    {"action": "SELECT_REGION", "target_region": "office", "graph_revision": 2},
    {"action": "SELECT_REGION", "target_region": "office", "graph_revision": 3,
     "cmd_vel": [9, 0]},
])
def test_graph_revision_and_candidate_gate_rejects_invalid_llm_choices(fake):
    ranked = EvidenceRouteSelector(DYNAMIC).ranked(graph(), "lobby", "parcel_red", set(), 100)
    with pytest.raises(MissionError):
        EvidenceRouteSelector.gated_llm_choice(fake, ranked, 3)


def test_goal_result_bound_to_current_goal_id_and_not_chat(engine):
    effects = start_visit(engine)
    assert [e["action"] for e in effects] == ["SEND_GOAL", "SET_TASK_CONTEXT"]
    token = engine.active_goal["goal_id"]
    engine.navigation_result(goal_id="wrong", status="SUCCEEDED", stamp=10)
    assert engine.index == 0 and engine.active_goal["goal_id"] == token
    engine.navigation_result(goal_id=token, status="SUCCEEDED", stamp=11)
    assert engine.current().target == "storage"
    assert engine.answers_from_events()["visited_regions"] == ["inspection"]
    assert engine.active_goal["goal_id"] != token


def test_cancel_ack_not_terminal_no_new_goal_then_terminal_releases(engine):
    start_visit(engine)
    engine.navigation_result(goal_id=engine.active_goal["goal_id"], status="SUCCEEDED", stamp=10)
    engine.take_effects()
    old_token = engine.active_goal["goal_id"]
    engine.revise_remaining([Step("NAVIGATE", "office", "success")], now=11)
    assert [e["action"] for e in engine.take_effects()] == ["CANCEL_GOAL"]
    engine.cancel_ack(goal_id=old_token, accepted=True)
    assert engine.take_effects() == []
    engine.navigation_result(goal_id=old_token, status="CANCELED", stamp=12)
    replacements = engine.take_effects()
    assert any(e["action"] == "SEND_GOAL" and e["target_region"] == "office"
               for e in replacements)
    engine.navigation_result(goal_id=engine.active_goal["goal_id"], status="SUCCEEDED", stamp=20)
    assert engine.answers_from_events()["visited_regions"] == ["inspection", "office"]
    assert engine.status()["finished"]


def test_cancel_race_result_succeeded_preserves_real_visit(engine):
    start_visit(engine, ("inspection",))
    old_id = engine.active_goal["goal_id"]
    engine.revise_remaining([Step("NAVIGATE", "office", "success")], now=1)
    engine.take_effects()
    engine.navigation_result(goal_id=old_id, status="SUCCEEDED", stamp=3)
    assert engine.answers_from_events()["visited_regions"] == ["inspection"]
    assert engine.active_goal["target_region"] == "office"


def test_failed_goal_does_not_fabricate_arrival(engine):
    start_visit(engine)
    engine.navigation_result(goal_id=engine.active_goal["goal_id"], status="FAILED", stamp=2)
    assert engine.answers_from_events()["visited_regions"] == []
    assert engine.status()["blocked"] == "NAV_FAILED"


def test_crash_restart_with_active_goal_requires_reconciliation(tmp_path):
    file = tmp_path / "persist.sqlite3"
    ledger = SQLiteMissionLog(file)
    first = GroundedMissionEngine(DYNAMIC, ledger)
    first.start(BoundedTaskCompiler(DYNAMIC).compile({"type": "visit_sequence", "places": ["office"]}),
                mission_id="run_r")
    ledger.close()
    store = SQLiteMissionLog(file)
    resumed = GroundedMissionEngine(DYNAMIC, store)
    assert resumed.status()["blocked"] == "INFLIGHT_STATE_REQUIRES_RECONCILIATION"
    assert resumed.take_effects() == []
    store.close()


def test_human_ack_cannot_be_auto_forged(engine):
    plan = BoundedTaskCompiler(DYNAMIC).compile({
        "type": "human_assisted_service", "source": "office", "service": "receiving",
        "destination": "office", "object_category": "parcel_red"})
    engine.start(plan, mission_id="human1")
    engine.navigation_result(goal_id=engine.active_goal["goal_id"], status="SUCCEEDED", stamp=1)
    with pytest.raises(MissionError):
        engine.human_ack(action="pickup:parcel_red", operator_id="robot",
                         note="autonomous", now=2)
    assert engine.current().op == "WAIT_HUMAN"
    engine.human_ack(action="pickup:parcel_red", operator_id="staff_01",
                     note="operator physically picked it up", now=3)
    assert engine.current().op == "NAVIGATE"
    assert engine.answers_from_events()["robot_manipulation"] == []


def test_object_stop_requires_new_stamped_corroborating_RGBD_frames(engine):
    compiler = BoundedTaskCompiler(DYNAMIC)
    engine.start(compiler.compile({"type":"objectnav", "category":"parcel_red"}),
                 mission_id="object001")
    region = engine.active_goal["target_region"]
    engine.navigation_result(goal_id=engine.active_goal["goal_id"],
                             status="SUCCEEDED", stamp=10)
    def report(obs_id, stamp, rng=.78, conf=.95, source="rendered_rgbd"):
        return {"source": source, "uses_sim_ground_truth": False,
                "observation_id": obs_id, "sensor_stamp": stamp,
                "detections": [{"category": "parcel_red", "confidence": conf,
                                "range_m": rng, "position": {"x": 5, "y": 3}}]}
    assert not engine.observe(report(1, 9), now=11)  # stale before arrival
    assert not engine.observe(report(2, 10.5, 2.0), now=11)  # too far
    assert not engine.observe(report(3, 10.7), now=11)  # first positive frame
    assert not engine.observe(report(3, 10.9), now=11)  # duplicate frame
    assert not engine.observe(report(4, 10.9, conf=.6), now=11)
    assert engine.observe(report(5, 11.0), now=11)  # independent corroboration
    assert engine.status()["finished"]
    assert engine.answers_from_events()["confirmed_objects"] == ["parcel_red"]
    assert region in DYNAMIC.places


def test_object_oracle_sensor_data_rejected(engine):
    engine.start(BoundedTaskCompiler(DYNAMIC).compile({"type":"objectnav", "category":"parcel_red"}),
                 mission_id="o1")
    with pytest.raises(MissionError):
        engine.observe({"uses_sim_ground_truth": True}, now=1)


def test_search_without_confirming_object_does_not_claim_absence(engine):
    engine.start(BoundedTaskCompiler(DYNAMIC).compile({"type":"objectnav", "category":"parcel_red"}),
                 mission_id="o1")
    goal = engine.active_goal["goal_id"]
    engine.navigation_result(goal_id=goal, status="SUCCEEDED", stamp=10)
    engine.continue_search(now=20)
    assert engine.answers_from_events()["confirmed_objects"] == []
    assert any(e["kind"] == "OBJECT_NOT_VERIFIED" and not e["negative_evidence_claimed"]
               for e in engine.log.events(engine.mission_id))


def test_offline_abc_trace_preserves_cancellation_contract():
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_grounded_abc_replay import replay
    trace = replay()
    assert trace["scene_B"]["after_cancel_ack"] == []
    assert trace["scene_C"]["answer_from_sqlite_events"]["visited_regions"] == ["inspection", "office"]
    assert trace["provenance"]["ran_Gazebo"] is False


def test_seeded_proxy_replay_is_deterministic_and_bounded():
    rows = []
    for seed in range(32):
        episode = create_episode(DYNAMIC, CONFIG, seed)
        for method in METHODS:
            first = run_episode(DYNAMIC, episode, method)
            second = run_episode(DYNAMIC, episode, method)
            assert first == second
            assert 0 <= first.proxy_SR <= 1
            assert 0 <= first.proxy_SPL <= 1
            assert 0 <= first.proxy_SoftSPL <= 1
            rows.append(first)
    assert len(summary(rows)) == len(METHODS)
    interval = paired_bootstrap(rows, "no_memory")
    assert interval["ci95_low"] <= interval["mean_difference"] <= interval["ci95_high"]


def test_proxy_oracle_upper_bound_per_matched_episode():
    for seed in range(32):
        ep = create_episode(DYNAMIC, CONFIG, seed)
        oracle = run_episode(DYNAMIC, ep, "oracle")
        assert oracle.proxy_SR == 1 and oracle.proxy_SPL == 1
        for method in METHODS:
            assert run_episode(DYNAMIC, ep, method).proxy_SPL <= oracle.proxy_SPL


def test_proxy_rejects_bad_vision_probabilities():
    with pytest.raises(ValueError):
        create_episode(DYNAMIC, CONFIG, 0, tpr=1.4)
    with pytest.raises(ValueError):
        create_episode(DYNAMIC, CONFIG, 0, fpr=-.01)


def test_agent_source_integrates_topics_but_not_ground_truth():
    node_file = ROOT / "rangermini_dynamic_semantic/grounded_mission_node.py"
    node = node_file.read_text(encoding="utf-8")
    ast.parse(node)
    for topic in ("/benchmark/goal", "/benchmark/cancel_goal",
                  "/benchmark/navigation_result", "/semantic_observations",
                  "/dynamic_semantic_graph"):
        assert topic in node
    assert 'create_subscription(String, "/benchmark/ground_truth' not in node
    launch = (ROOT / "launch/dynamic_semantic_benchmark.launch.py").read_text()
    ast.parse(launch)
    assert 'enable_grounded_agent' in launch
    assert 'default_value="false"' in launch
    controller = (ROOT / "rangermini_dynamic_semantic/benchmark_navigation_controller.py").read_text()
    ast.parse(controller)
    assert '"/benchmark/cancel_ack"' in controller
    assert '"goal_id": goal_id' in controller
