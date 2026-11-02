#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if v not in ("", "nan", "NaN")})
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--method", default="Ours")
    parser.add_argument("--goal_y", type=float, default=2.75)
    args = parser.parse_args()

    rows = read_csv(args.csv)
    min_dist = [r.get("min_distance", 0.0) for r in rows]
    inter = [r.get("intervention_score", 0.0) for r in rows]
    y = [r.get("y", 0.0) for r in rows]
    t = [r.get("time", 0.0) for r in rows]

    min_clearance = min(min_dist)
    max_intervention = max(inter)
    avg_intervention = mean(inter)

    active = [1 if v > 0.10 else 0 for v in inter]
    intervention_count = 0
    for a, b in zip(active[:-1], active[1:]):
        if a == 0 and b == 1:
            intervention_count += 1

    success = int((y[-1] >= args.goal_y) and (min_clearance > 0.0))
    collision = int(min_clearance <= 0.0)

    traversal_time = t[-1]
    for ti, yi in zip(t, y):
        if yi >= args.goal_y:
            traversal_time = ti
            break

    vs = [(r.get("vx_safe", 0.0), r.get("vy_safe", 0.0), r.get("wz_safe", 0.0)) for r in rows]
    if len(vs) > 2:
        sq = []
        for a, b in zip(vs[:-1], vs[1:]):
            sq.append((b[0]-a[0])**2 + (b[1]-a[1])**2 + (b[2]-a[2])**2)
        jitter = math.sqrt(mean(sq))
    else:
        jitter = 0.0

    result = {
        "method": args.method,
        "csv": args.csv,
        "minimum_obstacle_distance": min_clearance,
        "intervention_count": intervention_count,
        "average_intervention_intensity": avg_intervention,
        "maximum_intervention_intensity": max_intervention,
        "success": success,
        "collision": collision,
        "traversal_time": traversal_time,
        "velocity_jitter": jitter,
    }

    print("Trial metrics:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out.exists()
        with open(out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(result)
        print(f"Saved summary: {out}")


if __name__ == "__main__":
    main()
