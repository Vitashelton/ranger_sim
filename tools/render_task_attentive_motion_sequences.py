#!/usr/bin/env python3
"""Render a source-grounded task-grounding / hybrid MPPI evidence figure.

Panel (a) is the RGB frame used by two real task prompts. Panel (b) shows
actual VLM bindings and the resulting same-bank candidate re-scoring at one
logged Ranger state. All curves are logged planner rollouts, not hand-drawn
or executed trajectories.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/object_reference_20261026/pilot_v1"
RED_RUN = PILOT / "object_reference_a_red_s41"
BLUE_RUN = PILOT / "object_reference_a_blue_s41"
EP = RED_RUN / "episode"
RESULTS = ROOT / "results/08_object_reference_pilot"
OUT = ROOT / "paper/figures/task_attentive_motion_sequences"

INK = "#222222"
MID = "#777777"
LIGHT = "#d4d4d4"
OCHRE = "#9a762e"
RED = "#9a4943"
BLUE = "#486d8b"

plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 7.4,
    "axes.labelsize": 7.0, "axes.titlesize": 8.2,
    "xtick.labelsize": 6.7, "ytick.labelsize": 6.7,
    "axes.linewidth": .65, "lines.linewidth": 1.1,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def wrap_angle(x):
    return math.atan2(math.sin(x), math.cos(x))


def clearance_cost(traj, track, cfg):
    """Task clearance cost and minimum oriented-box gap for a logged rollout."""
    tr = np.asarray(traj, dtype=float)
    bx, by, angle, hx, hy = map(float, track["registered_box"])
    dx, dy, yaw = bx-tr[:, 0], by-tr[:, 1], tr[:, 2]
    co, si = np.abs(np.cos(yaw-angle)), np.abs(np.sin(yaw-angle))
    hl, hw = cfg["body_length"]/2, cfg["body_width"]/2
    gap = np.maximum.reduce([
        np.abs(dx*np.cos(yaw)+dy*np.sin(yaw))-hl-hx*co-hy*si,
        np.abs(-dx*np.sin(yaw)+dy*np.cos(yaw))-hw-hx*si-hy*co,
        np.abs(dx*math.cos(angle)+dy*math.sin(angle))-hx-hl*co-hw*si,
        np.abs(-dx*math.sin(angle)+dy*math.cos(angle))-hy-hl*co-hw*si,
    ])
    penalty = float(np.sum((np.maximum(.85-gap, 0)/.25)**2)
                    * cfg["dt"] * 9.0)
    return penalty, float(np.min(gap))


def robot_poly(pose, length, width):
    x, y, yaw = map(float, pose)
    local = np.array([[-length/2, -width/2], [length/2, -width/2],
                      [length/2, width/2], [-length/2, width/2]])
    rot = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
    return local @ rot.T + np.array([x, y])


def family_name(record):
    if record.get("structure"):
        return "".join(record["structure"])
    return {"ACKERMANN": "A", "PARALLEL": "P", "SPIN": "R"}.get(
        record.get("behavior"), "")


def main():
    rgb_red = EP / "proposal_001_raw_rgb.jpg"
    rgb_blue = BLUE_RUN / "episode/proposal_001_raw_rgb.jpg"
    assert hashlib.sha256(rgb_red.read_bytes()).digest() == hashlib.sha256(rgb_blue.read_bytes()).digest()

    red_input = json.loads((EP / "proposal_001_input.json").read_text())
    blue_input = json.loads((BLUE_RUN / "episode/proposal_001_input.json").read_text())
    red_proposal = json.loads((EP / "proposal_001_response.json").read_text())["proposal"]
    blue_proposal = json.loads((BLUE_RUN / "episode/proposal_001_response.json").read_text())["proposal"]
    assert red_input["candidates"] == blue_input["candidates"]
    assert red_proposal["relevant_object_ids"] == ["map_object_1"]
    assert blue_proposal["relevant_object_ids"] == ["map_object_2"]

    plans = read_jsonl_gz(EP / "hybrid_rollouts.jsonl.gz")
    obs_rows = read_jsonl_gz(EP / "planner_observations.jsonl.gz")
    cfg = json.loads((EP / "manifest.json").read_text())["mppi_config"]
    record = next(r for r in plans if abs(float(r["request_sim_time"])-8.268) < 1e-3)
    obs = min(obs_rows, key=lambda o: abs(float(o["sim_time"])-record["request_sim_time"]))
    tracks = {t["evidence_id"]: t for t in obs["tracks"]}
    assert {"map_object_1", "map_object_2"}.issubset(tracks)

    bank = list(record["plan"].get("candidate_summary", []))
    bank += [c for c in record["plan"].get("hybrid_schedule_candidates", []) if c.get("feasible")]
    scored = []
    for cand in bank:
        terms = cand.get("cost_terms", {})
        base = float(cand["cost"] - terms.get("task_clearance_preference", 0.0)
                     - terms.get("task_heading_preference", 0.0))
        red_term, red_gap = clearance_cost(cand["trajectory"], tracks["map_object_1"], cfg)
        blue_term, blue_gap = clearance_cost(cand["trajectory"], tracks["map_object_2"], cfg)
        scored.append({"record": cand, "base": base,
                       "red": base+red_term, "blue": base+blue_term,
                       "red_gap": red_gap, "blue_gap": blue_gap})

    # Pick logged representatives from the same candidate bank. The transition
    # example is the red-condition minimum and carries its alignment trace.
    def best_for_structure(code, condition):
        subset = [x for x in scored if family_name(x["record"]) == code]
        return min(subset, key=lambda x: x[condition])

    cand_c = best_for_structure("A", "red")
    cand_p = best_for_structure("P", "red")
    winners = {"red": min(scored, key=lambda x: x["red"]),
               "blue": min(scored, key=lambda x: x["blue"])}
    assert family_name(winners["red"]["record"]) == "PA"
    assert family_name(winners["blue"]["record"]) == "A"
    cand_pc = winners["red"]

    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax_rgb, ax_rank) = plt.subplots(
        1, 2, figsize=(7.16, 2.55),
        gridspec_kw={"width_ratios": [1.03, 1.35], "wspace": .20})
    fig.subplots_adjust(left=.04, right=.985, bottom=.19, top=.96)

    # The left panel is the exact RGB input used by both actual proposals.
    rgb = Image.open(rgb_red).convert("RGB")
    ax_rgb.imshow(rgb)
    ax_rgb.add_patch(Rectangle((145, 160), 87, 143, fill=False, lw=1.0, ec=RED))
    ax_rgb.add_patch(Rectangle((416, 160), 91, 143, fill=False, lw=1.0, ec=BLUE))
    ax_rgb.text(149, 157, "1", color="white", fontsize=7, ha="left", va="bottom",
                bbox={"boxstyle": "square,pad=.12", "fc": RED, "ec": "none"})
    ax_rgb.text(420, 157, "2", color="white", fontsize=7, ha="left", va="bottom",
                bbox={"boxstyle": "square,pad=.12", "fc": BLUE, "ec": "none"})
    ax_rgb.set_axis_off()
    ax_rgb.set_title("(a)", loc="left", pad=2)

    # Candidate costs are recomputed from one logged MPPI bank. Only the
    # grounded object changes between the two task conditions.
    categories = ["Curved", "Parallel", "Parallel → curved"]
    items = [cand_c, cand_p, cand_pc]
    ys = np.arange(len(items))
    red_vals = [float(item["red"]) for item in items]
    blue_vals = [float(item["blue"]) for item in items]
    ax_rank.set_yticks(ys, categories)
    ax_rank.invert_yaxis()
    for i, (rval, bval) in enumerate(zip(red_vals, blue_vals)):
        ax_rank.plot([rval, bval], [i, i], color="#bdbdbd", lw=.75, zorder=1)
    ax_rank.scatter(red_vals, ys, s=25, color=INK, edgecolor="white", linewidth=.45,
                    label="Red-box trolley", zorder=3)
    ax_rank.scatter(blue_vals, ys, s=27, facecolor="white", edgecolor=OCHRE,
                    linewidth=1.05, label="Blue-box trolley", zorder=3)
    red_best = int(np.argmin(red_vals))
    blue_best = int(np.argmin(blue_vals))
    ax_rank.scatter([red_vals[red_best]], [red_best], s=68, facecolor="none",
                    edgecolor=INK, linewidth=.8, zorder=2)
    ax_rank.scatter([blue_vals[blue_best]], [blue_best], s=68, facecolor="none",
                    edgecolor=OCHRE, linewidth=.8, zorder=2)
    ax_rank.set_xlabel("MPPI objective (lower is better)")
    ax_rank.set_xlim(57.5, 67.4)
    ax_rank.set_xticks([58, 60, 62, 64, 66])
    ax_rank.tick_params(axis="both", labelsize=7, length=2, pad=2)
    ax_rank.grid(axis="x", color="#e5e5e5", lw=.45, zorder=0)
    ax_rank.set_axisbelow(True)
    ax_rank.spines["top"].set_visible(False)
    ax_rank.spines["right"].set_visible(False)
    ax_rank.legend(frameon=False, loc="lower right", fontsize=6.8,
                   ncol=1, handletextpad=.45, borderpad=.1)
    ax_rank.set_title("(b)", loc="left", pad=2)

    fig.savefig(OUT.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=500, facecolor="white")
    plt.close(fig)

    # Preserve all data-selection and provenance details beside the figure.
    meta = {
        "source_state_sim_time_s": float(record["request_sim_time"]),
        "source_pose": record["pose"],
        "wheel_angles_rad": record["wheel_steering_rad"],
        "source_candidate_count": len(scored),
        "candidate_paths_are": "logged feasible MPPI predictions",
        "counterfactual": "same logged candidate bank re-scored with two actual VLM object bindings; no second closed-loop trajectory is implied",
        "conditions": {
            "red": {"object_id": "map_object_1", "clearance_m": .85,
                    "winner": family_name(winners["red"]["record"]),
                    "cost": winners["red"]["red"]},
            "blue": {"object_id": "map_object_2", "clearance_m": .85,
                     "winner": family_name(winners["blue"]["record"]),
                     "switch_steps": winners["blue"]["record"].get("switch_steps"),
                     "alignment_s": winners["blue"]["record"].get("alignment_s"),
                     "cost": winners["blue"]["blue"]},
        },
        "figure_sha256_inputs": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [rgb_red, rgb_blue, EP/"proposal_001_input.json",
                      EP/"proposal_001_response.json", EP/"hybrid_rollouts.jsonl.gz",
                      EP/"planner_observations.jsonl.gz", EP/"manifest.json",
                      BLUE_RUN/"episode/proposal_001_input.json",
                      BLUE_RUN/"episode/proposal_001_response.json"]
        },
    }
    OUT.with_suffix(".provenance.json").write_text(json.dumps(meta, indent=2)+"\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
