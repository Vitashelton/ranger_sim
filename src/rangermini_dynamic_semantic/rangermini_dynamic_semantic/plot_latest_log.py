#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def latest_csv(log_dir):
    files = sorted(Path(log_dir).glob("doorway_trial_*.csv"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No doorway_trial_*.csv found in {log_dir}")
    return files[-1]


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items() if v not in ("", "nan", "NaN")})
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def col(rows, name):
    return [r.get(name, 0.0) for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="/tmp/rangermini_doorway_logs")
    parser.add_argument("--csv", default="")
    parser.add_argument("--out_dir", default="/tmp/rangermini_doorway_logs/figures")
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else latest_csv(args.log_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_csv(csv_path)
    t = col(rows, "time")
    print(f"Using CSV: {csv_path}")

    for key, label in [("vx", "vx / m s^-1"), ("vy", "vy / m s^-1"), ("wz", "wz / rad s^-1")]:
        plt.figure(figsize=(9, 4))
        plt.plot(t, col(rows, f"{key}_h"), label=f"{key}_human", linewidth=2)
        plt.plot(t, col(rows, f"{key}_safe"), label=f"{key}_safe", linewidth=2)
        plt.xlabel("Time / s")
        plt.ylabel(label)
        plt.title(f"Human input vs safe output: {key}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / f"timeseries_{key}.png", dpi=220)
        plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(t, col(rows, "intervention_score"), label="intervention_score", linewidth=2)
    plt.plot(t, col(rows, "min_distance"), label="min_distance", linewidth=2)
    plt.plot(t, col(rows, "risk_score"), label="risk_score", linewidth=2)
    plt.xlabel("Time / s")
    plt.ylabel("Value")
    plt.title("Intervention intensity, clearance, and risk")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "timeseries_intervention_clearance_risk.png", dpi=220)
    plt.close()

    x = col(rows, "x")
    y = col(rows, "y")
    plt.figure(figsize=(6, 7))
    plt.plot(x, y, linewidth=2.5, label="executed path")
    plt.scatter([x[0]], [y[0]], label="start")
    plt.scatter([x[-1]], [y[-1]], marker="*", s=120, label="end")
    plt.fill_between([-3.2, -0.55], 1.10, 1.45, alpha=0.25, color="black")
    plt.fill_between([0.55, 3.2], 1.10, 1.45, alpha=0.25, color="black")
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.title("Executed BEV trajectory")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "trajectory_bev.png", dpi=220)
    plt.close()

    print(f"Saved figures to: {out}")


if __name__ == "__main__":
    main()
