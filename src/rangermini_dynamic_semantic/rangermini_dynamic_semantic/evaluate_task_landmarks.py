#!/usr/bin/env python3
import argparse
import csv

from .task_landmark_core import greedy_select, load_problem, route_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="task_landmark_results.csv")
    args = parser.parse_args()
    cfg, landmarks = load_problem(args.config)
    budget = int(cfg["budget"])
    ids = sorted(landmarks)
    uniform = {ids[round(i * (len(ids) - 1) / max(budget - 1, 1))] for i in range(budget)}
    doorway = {11, 13, 15, 17} & set(ids)
    all_routes = [pose for route in cfg["tasks"].values() for pose in route]
    global_static = greedy_select(all_routes, landmarks, cfg, budget)
    rows = []
    for room, route in cfg["tasks"].items():
        layouts = {
            "uniform_fixed": uniform,
            "doorway_manual": doorway,
            "global_d_optimal": global_static,
            "task_conditioned": greedy_select(route, landmarks, cfg, budget),
        }
        for method, selected in layouts.items():
            rows.append({
                "task_room": room,
                "method": method,
                "selected_ids": " ".join(map(str, sorted(selected))),
                **route_metrics(route, selected, landmarks, cfg),
            })
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['task_room']} {row['method']:<18} tags={row['selected_ids']:<11} "
              f"p_det={row['mean_detection_probability']:.3f} "
              f"PEB={row['mean_position_error_bound_m']:.3f}m")


if __name__ == "__main__":
    main()
