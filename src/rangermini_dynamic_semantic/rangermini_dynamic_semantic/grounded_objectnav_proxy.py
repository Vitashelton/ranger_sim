"""Reproducible *graph-level proxy* ObjectNav ablation for the uploaded world.

It uses the real dynamic_benchmark.yaml topology, synthetic camera detections,
synthetic semantic tracks, and hidden target-room truth in the EVALUATOR ONLY.
It does not execute Gazebo or claim official Habitat SR/SPL comparability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import random
import statistics

from .grounded_mission_core import EvidenceRouteSelector, PlaceCatalog

TARGETS = ("parcel_red", "parcel_blue", "inspection_panel", "charging_station")
METHODS = ("prior_only", "no_memory", "no_risk", "single_frame", "full", "oracle")


@dataclass(frozen=True)
class Episode:
    seed: int
    target_category: str
    target_room_oracle: str
    start_region: str
    shifted: bool
    occluded: bool
    graph_observed: dict
    detections: dict


@dataclass(frozen=True)
class Score:
    seed: int
    target_category: str
    scenario: str
    method: str
    proxy_SR: int
    proxy_SPL: float
    proxy_SoftSPL: float
    traveled_graph_m: float
    reference_shortest_m: float
    queried_regions: int
    processed_camera_frames: int
    false_stop: int
    navigation_failure_proxy: int


def _observed_graph(cfg: dict, category: str, truth_region: str,
                    shifted: bool, seed: int) -> dict:
    """Synthetic appearance history. Evaluator seeds it; policy sees only graph."""
    edges = [{"from": e["from"], "to": e["to"],
              "base_cost": float(e["cost"]), "effective_cost": float(e["cost"]),
              "state": "FREE"} for e in cfg["topology_edges"]]
    if seed % 3 == 0:
        for edge in edges:
            if {edge["from"], edge["to"]} == {"loading", "packing"}:
                edge["state"] = "TEMP_BLOCKED"
                edge["effective_cost"] = edge["base_cost"] * 9
    elif seed % 3 == 1:
        for edge in edges:
            if {edge["from"], edge["to"]} == {"inspection", "storage"}:
                edge["state"] = "TEMP_BLOCKED"
                edge["effective_cost"] = edge["base_cost"] * 9
    # Separate past observations are only a noisy sensor-memory fixture,
    # not the hidden target region exported to the online agent.
    tracks = []
    old_region = {"parcel_red": "receiving", "parcel_blue": "packing",
                  "inspection_panel": "inspection", "charging_station": "office"}[category]
    tracks.append({"category": category, "region": old_region,
                   "state": "CONFIRMED", "confidence": .93,
                   "last_seen": 9.0 if shifted else 79.0})
    if shifted:
        # A possible recent observation after migration, with dropout and noise.
        rng = random.Random(90071 + seed * 19)
        if rng.random() < .75:
            tracks.append({"category": category, "region": truth_region,
                           "state": "CONFIRMED", "confidence": .86,
                           "last_seen": 78.5})
    return {"revision": 7, "timestamp": 80.0,
            "place_nodes": [{"id": n} for n in cfg["regions"]],
            "edges": edges, "semantic_nodes": tracks}


def create_episode(catalog: PlaceCatalog, cfg: dict, seed: int,
                   *, tpr: float = .86, fpr: float = .12) -> Episode:
    if not 0 <= tpr <= 1 or not 0 <= fpr <= 1:
        raise ValueError("TPR/FPR out of bounds")
    rng = random.Random(8237463 + seed * 71)
    category = TARGETS[seed % len(TARGETS)]
    default_room = {"parcel_red": "receiving", "parcel_blue": "packing",
                    "inspection_panel": "inspection", "charging_station": "office"}[category]
    shifted = category == "parcel_red" and (seed // len(TARGETS)) % 2 == 1
    actual = "storage" if shifted else default_room
    start = rng.choice(["lobby", "office", "inspection", "packing", "east_junction"])
    graph = _observed_graph(cfg, category, actual, shifted, seed)
    reachable = EvidenceRouteSelector(catalog).route_costs(graph, start)
    if actual not in reachable:
        start = "lobby"
        assert actual in EvidenceRouteSelector(catalog).route_costs(graph, start)
    detections = {}
    for idx, region in enumerate(sorted(catalog.places)):
        local = random.Random(44791 + seed * 953 + idx * 271)
        prob = tpr if region == actual else fpr
        # Pre-generate independent camera-frame detections for *all* methods.
        detections[region] = [local.random() < prob for _ in range(3)]
    return Episode(seed, category, actual, start, shifted, False, graph, detections)


def run_episode(catalog: PlaceCatalog, ep: Episode, method: str,
                *, max_region_queries: int = 6) -> Score:
    if method not in METHODS:
        raise ValueError("invalid method")
    selector = EvidenceRouteSelector(catalog)
    actual_costs = selector.route_costs(ep.graph_observed, ep.start_region)
    shortest = actual_costs.get(ep.target_room_oracle, float("inf")) + 0.65
    current = ep.start_region
    visit_set: set[str] = set()
    traveled, queried, frames, false_stop, failure_proxy = 0.0, 0, 0, 0, 0
    stopped = None
    if method == "oracle":
        stopped = ep.target_room_oracle
        traveled = shortest
    else:
        for _ in range(max_region_queries):
            policy_mode = {
                "prior_only": "prior_only", "no_memory": "no_memory",
                "no_risk": "no_risk", "single_frame": "full",
                "full": "full",
            }[method]
            ranking = selector.ranked(ep.graph_observed, current,
                                      ep.target_category, visit_set,
                                      now=80.0, mode=policy_mode)
            if not ranking:
                break
            dest = ranking[0]["region"]
            navigation_costs = selector.route_costs(ep.graph_observed, current)
            if dest not in navigation_costs:
                break
            # The no-risk ablation may choose blocked edges; the evaluator
            # charges one recovery penalty for overoptimistic graph routes.
            if method == "no_risk" and ranking[0]["route_cost"] < navigation_costs[dest] - 1e-6:
                traveled += 2.5
                failure_proxy += 1
            traveled += navigation_costs[dest]
            current = dest
            queried += 1
            visit_set.add(dest)
            evidence = ep.detections[dest]
            if method in ("full", "no_memory", "no_risk"):
                frames += 3
                observed = sum(evidence) >= 2
            else:
                frames += 1
                observed = bool(evidence[0])
            if observed:
                traveled += 0.65   # synthetic safe approach within <=1m; not Gazebo
                stopped = dest
                if stopped != ep.target_room_oracle:
                    false_stop = 1
                break
    success = int(stopped == ep.target_room_oracle and not ep.occluded
                  and math.isfinite(shortest))
    efficiency = shortest / max(shortest, traveled, 1e-9) if math.isfinite(shortest) else 0.0
    remaining_cost = selector.route_costs(ep.graph_observed, current).get(
        ep.target_room_oracle, float("inf"))
    # A surrogate soft goal-progress measure; not Habitat SoftSPL.
    remaining = remaining_cost if math.isfinite(remaining_cost) else shortest
    progress = max(0.0, 1.0 - remaining / max(shortest, 1e-9))
    return Score(ep.seed, ep.target_category, "relocated" if ep.shifted else "static",
                 method, success, success * efficiency, progress * efficiency,
                 traveled, shortest, queried, frames, false_stop, failure_proxy)


def summary(rows: list[Score]) -> list[dict]:
    answer = []
    for method in METHODS:
        subset = [row for row in rows if row.method == method]
        if not subset:
            continue
        answer.append({"method": method, "episodes": len(subset),
                       "proxy_SR": statistics.mean(row.proxy_SR for row in subset),
                       "proxy_SPL": statistics.mean(row.proxy_SPL for row in subset),
                       "proxy_SoftSPL": statistics.mean(row.proxy_SoftSPL for row in subset),
                       "camera_frames": statistics.mean(row.processed_camera_frames for row in subset),
                       "queried_regions": statistics.mean(row.queried_regions for row in subset),
                       "false_stop_rate": statistics.mean(row.false_stop for row in subset),
                       "route_failure_proxy": statistics.mean(row.navigation_failure_proxy for row in subset)})
    return answer


def paired_bootstrap(rows: list[Score], baseline: str, *, target="full",
                     field="proxy_SPL", seed=2026, bootstrap_draws=2000):
    pairs = {}
    for row in rows:
        pairs.setdefault(row.seed, {})[row.method] = getattr(row, field)
    if not pairs or any(baseline not in p or target not in p for p in pairs.values()):
        raise ValueError("need matched pairs")
    diffs = [entry[target] - entry[baseline] for entry in pairs.values()]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                   for _ in range(bootstrap_draws))
    return {"baseline": baseline, "target": target,
            "mean_difference": statistics.mean(diffs),
            "ci95_low": draws[int(.025 * bootstrap_draws)],
            "ci95_high": draws[int(.975 * bootstrap_draws) - 1],
            "episodes": len(diffs)}
