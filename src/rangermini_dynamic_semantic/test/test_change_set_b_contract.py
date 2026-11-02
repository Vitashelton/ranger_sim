import ast
import hashlib
from pathlib import Path

import yaml

from rangermini_dynamic_semantic.reset_contract import canonical_hash


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "rangermini_dynamic_semantic"
PARTICIPANTS = {
    "dynamic_scene_manager.py": "dynamic_scene_manager",
    "streaming_perception_scheduler.py": "streaming_perception_scheduler",
    "rgbd_semantic_perception.py": "rgbd_semantic_perception",
    "temporal_semantic_memory.py": "temporal_semantic_memory",
    "dynamic_topology_maintenance.py": "dynamic_topology_maintenance",
    "benchmark_navigation_controller.py": "benchmark_navigation_controller",
    "dynamic_benchmark_metrics.py": "dynamic_benchmark_metrics",
}


def text(path):
    return path.read_text(encoding="utf-8")


def test_reset_barrier_has_request_ack_ready_phases():
    protocol = text(PKG / "reset_protocol.py")
    coordinator = text(PKG / "trial_reset_coordinator.py")
    assert 'RESET_REQUEST_TOPIC = "/benchmark/reset/request"' in protocol
    assert 'RESET_ACK_TOPIC = "/benchmark/reset/ack"' in protocol
    assert 'RESET_READY_TOPIC = "/benchmark/reset/ready"' in protocol
    assert "TRANSIENT_LOCAL" in protocol
    assert "set(self.acks) == set(self.expected)" in coordinator
    assert '"status": "READY"' in coordinator
    assert '"ack_states": ack_states' in coordinator


def test_every_runtime_participant_resets_and_acknowledges():
    for filename, node_name in PARTICIPANTS.items():
        source = text(PKG / filename)
        assert "ResetParticipant" in source, filename
        assert f'"{node_name}"' in source, filename
        assert "def on_reset_request" in source, filename
        assert "def on_reset_ready" in source, filename
        assert "self.reset.acknowledge(" in source, filename


def test_reset_contract_clears_all_mutable_trial_state():
    perception = text(PKG / "rgbd_semantic_perception.py")
    memory = text(PKG / "temporal_semantic_memory.py")
    graph = text(PKG / "dynamic_topology_maintenance.py")
    scheduler = text(PKG / "streaming_perception_scheduler.py")
    controller = text(PKG / "benchmark_navigation_controller.py")
    scene = text(PKG / "dynamic_scene_manager.py")
    metrics = text(PKG / "dynamic_benchmark_metrics.py")
    for assignment in ("self.rgb = None", "self.depth = None", "self.info = None",
                       "self.odom = None", "self.frame_counter = 0",
                       "self.processed_counter = 0"):
        assert assignment in perception
    for assignment in ("self.tracks.clear()", "self.next_id = 1", "self.revision = 0"):
        assert assignment in memory
    for assignment in ("self.initialize_edges()", "self.object_nodes.clear()",
                       "self.revision = 0", '"blocked_probability": 0.0'):
        assert assignment in graph
    for assignment in ("self.trigger_count = 0", "self.failure = None",
                       "self.stale_ratio = 0.0", "self.burst_until = 0.0"):
        assert assignment in scheduler
    for assignment in ("self.stop()", "self.goal_region = None", "self.path = []",
                       "self.blocked_since = None", "self.failed_edge = None"):
        assert assignment in controller
    assert "self.initial_entities" in scene and '"robot_initial_pose"' in scene
    assert "self.cmd_pub.publish(Twist())" in scene
    assert "self.open_trial_files(int(request.get(" in metrics
    assert "self.accept_events = False" in metrics


def test_nodes_do_not_advance_while_waiting_for_global_ready():
    guarded = [
        "streaming_perception_scheduler.py",
        "rgbd_semantic_perception.py",
        "temporal_semantic_memory.py",
        "dynamic_topology_maintenance.py",
        "benchmark_navigation_controller.py",
        "dynamic_scene_manager.py",
    ]
    for filename in guarded:
        source = text(PKG / filename)
        assert "if self.reset.resetting:" in source, filename
    metrics = text(PKG / "dynamic_benchmark_metrics.py")
    assert "if not self.accept_events:" in metrics


def test_two_equal_trials_have_equal_deterministic_reset_hashes():
    cfg = yaml.safe_load(text(ROOT / "config" / "dynamic_benchmark.yaml"))
    task = yaml.safe_load(text(ROOT / "config" / "task_context.yaml"))
    trial = {"trial_id": "smoke_S6_seed7", "scenario_id": "S6", "seed": 7,
             "method_mode": "Ours", "task_context": task}
    expected = sorted(PARTICIPANTS.values())
    manifest = {"contract_version": "v0.20.0+ChangeSetB",
                "trial_context": trial, "expected_reset_nodes": expected}
    oracle = cfg["scenarios"]["mixed_shift"]["events"]
    memory = {"revision": 0, "next_id": 1, "tracks": [],
              "update_stats": {"accepted": 0, "task_deferred": 0}}
    graph = {"revision": 0, "semantic_nodes": [], "edges": [{
        "id": f"edge_{index:02d}", "from": edge["from"], "to": edge["to"],
        "state": "FREE", "blocked_probability": 0.0,
        "traversal_successes": 0, "traversal_failures": 0,
    } for index, edge in enumerate(cfg["topology_edges"])]}
    first = tuple(canonical_hash(value) for value in (manifest, oracle, memory, graph))
    second = tuple(canonical_hash(value) for value in (manifest, oracle, memory, graph))
    assert first == second


def test_reset_infrastructure_contains_no_wall_clock_state_logic():
    for filename in ("reset_protocol.py", "trial_reset_coordinator.py"):
        tree = ast.parse(text(PKG / filename))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert not (isinstance(node.func.value, ast.Name) and
                            node.func.value.id == "time" and
                            node.func.attr in {"time", "monotonic", "perf_counter"})


def test_frozen_benchmark_assets_are_unchanged():
    expected = {
        "config/dynamic_benchmark.yaml":
            "e4ce3fe9a63651b7044c4fb85ae6e412cbf2790fa8135465ac4b5f1809e65ea7",
        "worlds/dynamic_indoor_benchmark.sdf":
            "ee3c673ba72ef91f4e695e12e826fec6df0784bc683b96553f13f170bbed762d",
        "docs/PAPER_METHOD_SECTION_ZH.md":
            "c3471e2bf45e8d86d36e07538430a8d94d7b7e9e3e57f5587336da2e94f64349",
        "docs/EXPERIMENT_PROTOCOL_V020_ZH.md":
            "89dfcaa6af29999e30126002ff60f67f8babb726ca20faf64a351331a2937f36",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_launch_wires_coordinator_without_nav2_or_baseline_profiles():
    launch = text(ROOT / "launch" / "dynamic_semantic_benchmark.launch.py")
    assert 'executable="trial_reset_coordinator"' in launch
    assert 'DeclareLaunchArgument("auto_reset_on_start"' in launch
    assert '"expected_reset_nodes"' in launch
    assert "nav2" not in launch.lower()
    assert "semanticupdatepolicy" not in launch.lower()


def test_runtime_smoke_verifier_checks_two_reset_signatures():
    smoke = text(PKG / "atomic_reset_smoke.py")
    setup = text(ROOT / "setup.py")
    assert 'self.create_client(Trigger, "/benchmark/reset_trial")' in smoke
    assert "if len(self.results) == 2" in smoke
    for field in ("manifest_hash", "oracle_event_hash",
                  "initial_semantic_snapshot_hash", "initial_topology_hash",
                  "memory_revision", "graph_revision", "edge_probabilities"):
        assert field in smoke
    assert "atomic_reset_smoke =" in setup
