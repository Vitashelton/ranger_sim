\
#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

DOOR_FRONTS = {
    "902": (3.00, 4.15),
    "904": (8.00, 4.15),
    "906": (13.35, 4.15),
    "908": (18.55, 4.15),
}

def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {}
            for k, v in row.items():
                try:
                    out[k] = float(v)
                except Exception:
                    pass
            rows.append(out)
    if not rows:
        raise ValueError(f"empty csv: {path}")
    return rows

def mean(xs):
    return sum(xs) / max(len(xs), 1)

def nearest_door(x, y):
    best_room = None
    best_dist = 1e9
    for room, (dx, dy) in DOOR_FRONTS.items():
        d = math.hypot(x - dx, y - dy)
        if d < best_dist:
            best_room, best_dist = room, d
    return best_room, best_dist

def latest_csv(log_dir):
    files = sorted(Path(log_dir).glob("corridor_trial_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No corridor_trial_*.csv found in {log_dir}")
    return files[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="")
    parser.add_argument("--log_dir", default="/tmp/rangermini_corridor_logs")
    parser.add_argument("--target_room", default="906")
    parser.add_argument("--success_radius", type=float, default=0.70)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else latest_csv(args.log_dir)
    rows = read_csv(csv_path)

    x = [r.get("x", 0.0) for r in rows]
    y = [r.get("y", 0.0) for r in rows]
    t = [r.get("time", 0.0) for r in rows]
    min_dist = [r.get("min_distance", 999.0) for r in rows]
    inter = [r.get("intervention_score", 0.0) for r in rows]
    vx = [r.get("vx_safe", 0.0) for r in rows]
    vy = [r.get("vy_safe", 0.0) for r in rows]
    wz = [r.get("wz_safe", 0.0) for r in rows]

    target = DOOR_FRONTS[args.target_room]
    final_d = math.hypot(x[-1] - target[0], y[-1] - target[1])
    success = int(final_d <= args.success_radius and min(min_dist) > 0.0)
    collision = int(min(min_dist) <= 0.0)

    nearest_room_final, nearest_room_dist = nearest_door(x[-1], y[-1])
    wrong_door = int(nearest_room_final != args.target_room and nearest_room_dist < 1.2)

    active = [1 if v > 0.10 else 0 for v in inter]
    intervention_count = sum(1 for a, b in zip(active[:-1], active[1:]) if a == 0 and b == 1)

    stop_count = 0
    stopped = [1 if abs(a) < 0.03 and abs(b) < 0.03 and abs(c) < 0.03 else 0 for a, b, c in zip(vx, vy, wz)]
    for a, b in zip(stopped[:-1], stopped[1:]):
        if a == 0 and b == 1:
            stop_count += 1

    jitter_terms = []
    for i in range(1, len(vx)):
        jitter_terms.append((vx[i]-vx[i-1])**2 + (vy[i]-vy[i-1])**2 + (wz[i]-wz[i-1])**2)
    velocity_jitter = math.sqrt(mean(jitter_terms)) if jitter_terms else 0.0

    arrival_time = t[-1]
    for ti, xi, yi in zip(t, x, y):
        if math.hypot(xi - target[0], yi - target[1]) <= args.success_radius:
            arrival_time = ti
            break

    row = {
        "csv": str(csv_path),
        "target_room": args.target_room,
        "success": success,
        "collision": collision,
        "wrong_door": wrong_door,
        "nearest_room_final": nearest_room_final,
        "final_target_distance": final_d,
        "minimum_clearance": min(min_dist),
        "average_intervention": mean(inter),
        "maximum_intervention": max(inter),
        "intervention_count": intervention_count,
        "stop_count": stop_count,
        "arrival_time": arrival_time,
        "velocity_jitter": velocity_jitter,
    }

    print("Corridor trial metrics:")
    for k, v in row.items():
        print(f"  {k}: {v}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not out.exists()
        with open(out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"Saved summary: {out}")

if __name__ == "__main__":
    main()
