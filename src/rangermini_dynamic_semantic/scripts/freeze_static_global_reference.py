#!/usr/bin/env python3
"""Generate a shared footprint-feasible reference using static SDF geometry only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from rangermini_dynamic_semantic.footprint_route_guide import FootprintRouteGuide


def freeze(world, start, goal, start_yaw, goal_yaw, output):
    graph = SpatialGraph(world)
    planner = FootprintRouteGuide(graph)
    guide, path = planner.guide((start[0], start[1], start_yaw), goal)
    points = [(float(x), float(y)) for x, y in path]
    if not points:
        if not planner.segment(start, goal, start_yaw):
            raise RuntimeError('Static-map planner found no footprint-feasible reference')
        points = [tuple(start), tuple(goal)]
    else:
        if math.dist(points[0], start) > 1e-6:
            points.insert(0, tuple(start))
        if math.dist(points[-1], goal) > 1e-6:
            points.append(tuple(goal))
    # Store a dense, immutable polyline so every local planner can advance the
    # same reference by arc length without replanning it from its own pose.
    dense = [points[0]]
    for a, b in zip(points, points[1:]):
        count = max(1, math.ceil(math.dist(a, b)/.12))
        dense.extend((a[0]+(b[0]-a[0])*k/count,
                      a[1]+(b[1]-a[1])*k/count) for k in range(1, count+1))
    points = dense
    for a, b in zip(points, points[1:]):
        yaw = math.atan2(b[1]-a[1], b[0]-a[0])
        if not planner.segment(a, b, yaw):
            raise RuntimeError(f'Static-map footprint check failed on segment {a} -> {b}')
    static = np.asarray(graph.static_boxes, dtype='<f8')
    payload = {
        'format': 'ranger_static_global_reference_v1',
        'planner': 'FootprintRouteGuide static-map A* with line-of-sight shortcut',
        'world_geometry_source': str(Path(world).resolve()),
        'static_geometry_sha256': hashlib.sha256(static.tobytes()).hexdigest(),
        'static_box_count': len(graph.static_boxes),
        'dynamic_actor_policy': 'moving people and trolleys are excluded; each local planner handles its own observations',
        'start_pose': [float(start[0]), float(start[1]), float(start_yaw)],
        'goal_pose': [float(goal[0]), float(goal[1]), float(goal_yaw)],
        'path_xy': [list(p) for p in points],
        'path_length_m': sum(math.dist(a, b) for a, b in zip(points, points[1:])),
        'all_segments_passed_static_footprint_check': True,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', required=True)
    parser.add_argument('--start', nargs=2, type=float, required=True)
    parser.add_argument('--start-yaw', type=float, default=0.)
    parser.add_argument('--goal', nargs=2, type=float, required=True)
    parser.add_argument('--goal-yaw', type=float, default=0.)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    freeze(args.world, args.start, args.goal, args.start_yaw, args.goal_yaw, args.output)
