import ast
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STATE_MODULES = [
    "streaming_perception_scheduler.py",
    "temporal_semantic_memory.py",
    "dynamic_topology_maintenance.py",
    "benchmark_navigation_controller.py",
    "dynamic_scene_manager.py",
    "dynamic_benchmark_metrics.py",
]
EVENT_MODULES = STATE_MODULES + ["rgbd_semantic_perception.py"]


def source(module):
    return (ROOT / "rangermini_dynamic_semantic" / module).read_text(encoding="utf-8")


def test_state_logic_contains_no_wall_clock_calls():
    for module in STATE_MODULES:
        tree = ast.parse(source(module))
        forbidden = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (isinstance(node.func.value, ast.Name) and
                    node.func.value.id == "time" and
                    node.func.attr in {"time", "monotonic", "perf_counter"}):
                forbidden.append(node.func.attr)
        assert not forbidden, (module, forbidden)


def test_perf_counter_is_only_used_for_perception_duration():
    for module in EVENT_MODULES:
        count = source(module).count("perf_counter")
        if module == "rgbd_semantic_perception.py":
            assert count == 2
        else:
            assert count == 0


def test_sim_time_is_propagated_to_every_dynamic_node():
    launch = (ROOT / "launch" / "dynamic_semantic_benchmark.launch.py").read_text()
    assert 'DeclareLaunchArgument("use_sim_time", default_value="true")' in launch
    assert 'DeclareLaunchArgument("require_sim_time", default_value="true")' in launch
    assert launch.count("parameters=[common") == 8
    for module in EVENT_MODULES:
        text = source(module)
        assert "require_sim_time" in text
        if module == "dynamic_scene_manager.py":
            assert '"/clock"' in text
        else:
            assert "get_clock().now()" in text


def test_pause_cannot_advance_lifecycle_or_timeouts_by_wall_clock():
    memory = source("temporal_semantic_memory.py")
    scheduler = source("streaming_perception_scheduler.py")
    topology = source("dynamic_topology_maintenance.py")
    controller = source("benchmark_navigation_controller.py")
    assert "now = self.now_sec()" in memory
    assert "now = self.now_sec()" in scheduler
    assert "now = self.now_sec()" in topology
    assert "now = self.now_sec()" in controller
    for text in (memory, scheduler, topology, controller):
        assert "time.monotonic" not in text and "time.time" not in text


def test_task_context_contract_and_memory_path():
    task = yaml.safe_load((ROOT / "config" / "task_context.yaml").read_text())
    required_task = {"task_id", "task_type", "target", "target_region",
                     "priority_objects", "failure_policy"}
    assert required_task <= set(task)
    assert yaml.safe_load(json.dumps(task)) == task
    scheduler = source("streaming_perception_scheduler.py")
    memory = source("temporal_semantic_memory.py")
    assert '"/task_context/current"' in scheduler
    assert '"/task_context/current"' in memory
    assert 'payload.get("task_context", payload)' in memory
    assert "self.relevance(detection)" in memory


def test_all_dynamic_event_sources_attach_trial_context():
    for module in EVENT_MODULES:
        text = source(module)
        for field in ("trial_id", "scenario_id", "seed", "method_mode"):
            assert field in text, (module, field)
        assert "trial_context" in text, module
