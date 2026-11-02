#!/usr/bin/env python3
"""Paired graph-level synthetic ObjectNav proxy; results are not official Habitat."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rangermini_dynamic_semantic.grounded_mission_core import PlaceCatalog
from rangermini_dynamic_semantic.grounded_objectnav_proxy import (
    METHODS, create_episode, run_episode, summary, paired_bootstrap)


def run(output: Path, episodes: int = 240, *, tpr=.86, fpr=.12):
    catalog_file = ROOT / "config/grounded_places_dynamic.json"
    cfg_file = ROOT / "config/dynamic_benchmark.yaml"
    catalog = PlaceCatalog.load(catalog_file)
    cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
    fixtures = [create_episode(catalog, cfg, seed, tpr=tpr, fpr=fpr)
                for seed in range(episodes)]
    rows = [run_episode(catalog, ep, method) for ep in fixtures for method in METHODS]
    stats = summary(rows)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    cis = [paired_bootstrap(rows, base) for base in METHODS if base not in ("full", "oracle")]
    manifest = {
        "protocol": "RangerDynamic-Graph-ObjectNav-PROXY-v1",
        "uses_Gazebo": False, "uses_ROS": False,
        "uses_real_VLM": False, "uses_real_RGBD": False,
        "official_Habitat_ObjectNav": False,
        "topology_source": "config/dynamic_benchmark.yaml",
        "truth_usage": "seeded evaluator ONLY; no oracle region in policy input",
        "shortest_path": "weighted region-graph Dijkstra + 0.65m synthetic approach",
        "simulated_approach": "no continuous collision checking",
        "synthetic_camera": "3 independent Bernoulli frames per room with TPR/FPR",
        "object_confirmed": "full=2/3 independent frames, ablation=one frame",
        "success": "oracle target region matched, abstract target approach, oracle visible",
        "metric_labels": "proxy_SR / proxy_SPL / proxy_SoftSPL (not Habitat scores)",
        "draws": episodes, "tpr": tpr, "fpr": fpr,
        "bootstrap": "paired 2000 resamples (exploratory intervals)",
        "config_sha256": hashlib.sha256(cfg_file.read_bytes()).hexdigest(),
        "catalog_sha256": hashlib.sha256(catalog_file.read_bytes()).hexdigest(),
        "ci95": cis,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for s in stats:
        print(f'{s["method"]:<16} proxy_SR={s["proxy_SR"]:.3f} '
              f'proxy_SPL={s["proxy_SPL"]:.3f} false_stop={s["false_stop_rate"]:.3f}')
    return rows, stats, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/grounded_proxy"))
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--tpr", type=float, default=.86)
    parser.add_argument("--fpr", type=float, default=.12)
    args = parser.parse_args()
    if args.episodes < 4:
        parser.error("episodes must be >=4")
    run(args.out, args.episodes, tpr=args.tpr, fpr=args.fpr)


if __name__ == "__main__":
    main()
