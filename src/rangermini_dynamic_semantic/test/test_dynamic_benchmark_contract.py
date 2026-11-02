import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ONLINE_MODULES = [
    "streaming_perception_scheduler.py",
    "rgbd_semantic_perception.py",
    "temporal_semantic_memory.py",
    "dynamic_topology_maintenance.py",
    "benchmark_navigation_controller.py",
]


def load_config():
    return yaml.safe_load((ROOT / "config" / "dynamic_benchmark.yaml").read_text())


def test_world_and_python_parse():
    ET.parse(ROOT / "worlds" / "dynamic_indoor_benchmark.sdf")
    modules = ONLINE_MODULES + ["dynamic_scene_manager.py", "dynamic_benchmark_metrics.py"]
    for module in modules:
        ast.parse((ROOT / "rangermini_dynamic_semantic" / module).read_text())


def test_topology_and_scenario_edges_are_consistent():
    cfg = load_config()
    regions = set(cfg["regions"])
    edges = set()
    for edge in cfg["topology_edges"]:
        assert edge["from"] in regions and edge["to"] in regions
        edges.add(tuple(sorted((edge["from"], edge["to"]))))
    assert len(edges) >= 8
    assert len(edges) == len(regions)  # one cycle gives two route choices
    for scenario in cfg["scenarios"].values():
        for event in scenario["events"]:
            if "edge" in event:
                assert tuple(sorted(event["edge"].split("-", 1))) in edges


def test_online_modules_do_not_subscribe_to_ground_truth():
    for module in ONLINE_MODULES:
        source = (ROOT / "rangermini_dynamic_semantic" / module).read_text()
        assert 'create_subscription(String, "/benchmark/ground_truth' not in source


def test_dynamic_entities_exist_in_world():
    cfg = load_config()
    tree = ET.parse(ROOT / "worlds" / "dynamic_indoor_benchmark.sdf")
    names = {model.attrib["name"] for model in tree.findall(".//model")}
    assert set(cfg["initial_entities"]).issubset(names)
    for scenario in cfg["scenarios"].values():
        assert {event["entity"] for event in scenario["events"]}.issubset(names)
