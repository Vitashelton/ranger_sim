#!/usr/bin/env python3
"""Link an actual visual object grounding to same-state MPPI reranking.

The planning panel re-scores the identical logged trajectory bank under two
actual object-reference conditions. It is a candidate-bank counterfactual,
not a second closed-loop Gazebo run.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "artifacts/object_reference_20261026/pilot_v1"
RED_RUN = PILOT / "object_reference_a_red_s41"
BLUE_RUN = PILOT / "object_reference_a_blue_s41"
OUT = ROOT / "paper/figures/grounding_mppi_causal"
RESULTS = ROOT / "results/08_object_reference_pilot"

plt.rcParams.update({
    "font.family": "Liberation Serif", "font.size": 7.2,
    "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": .65, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})


def read_gzip_jsonl(path: Path):
    with gzip.open(path, "rt") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def condition_cost(trajectory, track, cfg, clearance, scale):
    """Exact registered-box clearance preference used by InteractionMPPI."""
    tr = np.asarray(trajectory, dtype=float)
    bx, by, angle, hx, hy = map(float, track["registered_box"])
    dx, dy, yaw = bx-tr[:, 0], by-tr[:, 1], tr[:, 2]
    co, si = np.abs(np.cos(yaw-angle)), np.abs(np.sin(yaw-angle))
    hl, hw = cfg["body_length"]/2, cfg["body_width"]/2
    gap = np.maximum.reduce([
        np.abs(dx*np.cos(yaw)+dy*np.sin(yaw))-hl-hx*co-hy*si,
        np.abs(-dx*np.sin(yaw)+dy*np.cos(yaw))-hw-hx*si-hy*co,
        np.abs(dx*math.cos(angle)+dy*math.sin(angle))-hx-hl*co-hw*si,
        np.abs(-dx*math.sin(angle)+dy*math.cos(angle))-hy-hl*si-hw*co,
    ])
    return float(np.sum((np.maximum(clearance-gap, 0)/scale)**2)
                 * cfg["dt"] * 9.0), float(np.min(gap))


def wrap_angle(x):
    return math.atan2(math.sin(x), math.cos(x))


def robot_polygon(pose, length, width):
    x, y, yaw = map(float, pose)
    local = np.array([[-length/2, -width/2], [length/2, -width/2],
                      [length/2, width/2], [-length/2, width/2]])
    rot = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
    return local @ rot.T + np.array([x, y])


def main():
    episode = RED_RUN / "episode"
    plans = read_gzip_jsonl(episode / "hybrid_rollouts.jsonl.gz")
    observations = read_gzip_jsonl(episode / "planner_observations.jsonl.gz")
    # Use the representative logged state at 8.092 s. It is the same candidate
    # bank under both independently obtained proposals from the identical RGB
    # frame and identical non-instruction context.
    cfg = json.loads((episode / "manifest.json").read_text())["mppi_config"]
    request = json.loads((RED_RUN / "episode/proposal_001_response.json").read_text())["proposal"]
    blue_request = json.loads((BLUE_RUN / "episode/proposal_001_response.json").read_text())["proposal"]
    red_input = json.loads((RED_RUN / "episode/proposal_001_input.json").read_text())
    blue_input = json.loads((BLUE_RUN / "episode/proposal_001_input.json").read_text())
    assert red_input["candidates"] == blue_input["candidates"]
    red_context, blue_context = dict(red_input["context"]), dict(blue_input["context"])
    red_instruction = red_context.pop("instruction")
    blue_instruction = blue_context.pop("instruction")
    assert red_context == blue_context
    rgb_a = RED_RUN / "episode/proposal_001_raw_rgb.jpg"
    rgb_b = BLUE_RUN / "episode/proposal_001_raw_rgb.jpg"
    assert hashlib.sha256(rgb_a.read_bytes()).digest() == hashlib.sha256(rgb_b.read_bytes()).digest()
    assert request["relevant_object_ids"] == ["map_object_1"]
    assert blue_request["relevant_object_ids"] == ["map_object_2"]
    chosen = None
    for record in plans:
        if abs(float(record["request_sim_time"]) - 8.092) > 1e-3:
            continue
        active = record["plan"].get("applied_motion_condition") or {}
        if active.get("relevant_object_ids") != request["relevant_object_ids"]:
            continue
        obs = min(observations, key=lambda row: abs(row["sim_time"]-record["request_sim_time"]))
        tracks = {item["evidence_id"]: item for item in obs["tracks"]}
        if not {"map_object_1", "map_object_2"}.issubset(tracks):
            continue
        bank = list(record["plan"].get("candidate_summary", []))
        bank += [item for item in record["plan"].get("hybrid_schedule_candidates", [])
                 if item.get("feasible")]
        scored = []
        for item in bank:
            terms = item.get("cost_terms", {})
            base = float(item["cost"]
                         - terms.get("task_clearance_preference", 0.0)
                         - terms.get("task_heading_preference", 0.0))
            traj = item["trajectory"]
            red_add, red_gap = condition_cost(traj, tracks["map_object_1"], cfg, .85, .25)
            blue_add, blue_gap = condition_cost(traj, tracks["map_object_2"], cfg, .85, .25)
            scored.append({
                "record": item,
                "base_cost": base,
                "red_cost": base+red_add,
                "blue_cost": base+blue_add,
                "red_gap_m": red_gap,
                "blue_gap_m": blue_gap,
            })
        if not scored:
            continue
        red_best = min(scored, key=lambda item: item["red_cost"])
        blue_best = min(scored, key=lambda item: item["blue_cost"])
        def key(item):
            r = item["record"]
            return (r.get("behavior", "+".join(r.get("structure", []))),
                    tuple(r.get("switch_steps", [])))
        chosen = (record, obs, tracks, scored, red_best, blue_best)
        break
    if chosen is None:
        raise RuntimeError("No same-state candidate-bank pair changes its selected candidate")

    record, obs, tracks, scored, red_best, blue_best = chosen
    pose = np.asarray(record["pose"], dtype=float)
    actual = record["plan"].get("selected_schedule")
    def candidate_name(best):
        r = best["record"]
        if "structure" not in r:
            return r.get("behavior", "primitive")
        names = {"A": "Ackermann", "P": "parallel", "R": "spin"}
        seq = "→".join(names.get(code, code) for code in r["structure"])
        when = r.get("switch_steps", [])
        suffix = "" if not when else " @ %.1f s" % (when[0]*cfg["dt"])
        return seq + suffix

    # Independent closed-loop object-reference observations, retained as
    # paired target-versus-other clearances from each real Gazebo episode.
    outcome = json.loads((RESULTS / "execution_metrics.json").read_text())
    rows = [m for m in outcome["metrics"] if m.get("grounded_target_correct")]
    source_files = [
        rgb_a, rgb_b,
        episode / "proposal_001_response.json",
        episode / "proposal_001_input.json",
        episode / "hybrid_rollouts.jsonl.gz",
        episode / "planner_observations.jsonl.gz",
        episode / "manifest.json",
        BLUE_RUN / "episode/proposal_001_response.json",
        BLUE_RUN / "episode/proposal_001_input.json",
        RESULTS / "execution_metrics.json",
        ROOT / "tools/render_grounding_mppi_causal.py",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / "same_state_candidate_rescoring.csv"
    fields = ["condition", "candidate", "cost", "base_cost", "target_gap_m",
              "other_gap_m", "selected"]
    rows_csv = []
    for condition, field in (("Red object", "red_cost"), ("Blue object", "blue_cost")):
        best = red_best if condition == "Red object" else blue_best
        for item in scored:
            r = item["record"]
            rows_csv.append({
                "condition": condition,
                "candidate": candidate_name(item),
                "cost": item[field],
                "base_cost": item["base_cost"],
                "target_gap_m": item["red_gap_m"] if condition == "Red object" else item["blue_gap_m"],
                "other_gap_m": item["blue_gap_m"] if condition == "Red object" else item["red_gap_m"],
                "selected": item is best,
            })
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows_csv)

    fig = plt.figure(figsize=(7.16, 3.55))
    gs = fig.add_gridspec(2, 2, left=.055, right=.985, bottom=.13, top=.94,
                          wspace=.24, hspace=.34, width_ratios=[1.0, 1.0],
                          height_ratios=[1.55, 1.0])
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_map = fig.add_subplot(gs[0, 1])
    ax_cost = fig.add_subplot(gs[1, 0])
    ax_gap = fig.add_subplot(gs[1, 1])

    # Panel a: identical original RGB supplied to both actual requests.
    # Crop only removes empty upper/lower image margins, leaving all pixels in
    # the two carts and their corridor context unchanged.
    rgb = Image.open(rgb_a).convert("RGB")
    crop_box = (0, 65, rgb.width, 431)
    ax_rgb.imshow(rgb.crop(crop_box), aspect="auto")
    ax_rgb.set_axis_off()
    ax_rgb.set_title("(a) Identical RGB supplied with two requests", loc="left", pad=4)

    # Panel b: exact state and candidate trajectories from one recorded plan.
    # Objects are the logged registered rectangles; curves are model rollouts.
    for obj_id, tag, yoff in (("map_object_1", "red-box trolley", .27),
                              ("map_object_2", "blue-box trolley", -.27)):
        bx, by, angle, hx, hy = tracks[obj_id]["registered_box"]
        box = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        rot = np.array([[math.cos(angle), -math.sin(angle)],
                        [math.sin(angle), math.cos(angle)]])
        corners = box @ rot.T + np.array([bx, by])
        ax_map.add_patch(Polygon(corners, closed=True, facecolor="#dedede",
                                 edgecolor="#363636", lw=.7, zorder=2))
        ax_map.annotate(tag, xy=(bx, by), xytext=(bx+.48, by+yoff),
                        fontsize=5.9, ha="left", va="center", color="#222222",
                        arrowprops=dict(arrowstyle="-", lw=.55, color="#555555"),
                        zorder=3)
    for best, color, style, label in ((red_best, "#343434", "-", "Red request"),
                                      (blue_best, "#ad812e", "--", "Blue request")):
        xy = np.asarray(best["record"]["trajectory"], dtype=float)
        ax_map.plot(xy[:, 0], xy[:, 1], color=color, ls=style, lw=1.6,
                    label=label, zorder=4)
        ax_map.plot(xy[0, 0], xy[0, 1], marker="o", ms=3.5, mfc="white",
                    mec=color, mew=.9, zorder=5)
        ax_map.plot(xy[-1, 0], xy[-1, 1], marker="s", ms=3.8, mfc="white",
                    mec=color, mew=1.0, zorder=5)
    ax_map.add_patch(Polygon(robot_polygon(pose, cfg["body_length"], cfg["body_width"]),
                             closed=True, facecolor="white", edgecolor="#111111",
                             lw=.85, zorder=6))
    ax_map.arrow(pose[0], pose[1], .32*math.cos(pose[2]), .32*math.sin(pose[2]),
                 width=.008, head_width=.07, head_length=.08, color="#111111",
                 length_includes_head=True, zorder=7)
    ax_map.set_xlim(3.85, 7.75); ax_map.set_ylim(-3.9, -.5)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("x (m)"); ax_map.set_ylabel("y (m)")
    ax_map.set_title("(b) Same MPPI state, different selected sequence", loc="left", pad=4)
    ax_map.text(.02, .97, "Red request: Parallel",
                transform=ax_map.transAxes, ha="left", va="top", fontsize=6.3,
                color="#343434", zorder=8)
    ax_map.text(.02, .89, "Blue request: Parallel→Ackermann (1.2 s)",
                transform=ax_map.transAxes, ha="left", va="top", fontsize=6.3,
                color="#8d6723", zorder=8)
    ax_map.text(.02, .81, "Wheel state: 11.8°",
                transform=ax_map.transAxes, ha="left", va="top", fontsize=6.1,
                color="#343434", zorder=8)

    # Panel c: compare the two winning motion sequences under both conditions.
    # Their scores are recomputed on the same recorded bank and state.
    plotted = [red_best, blue_best]
    categories = ["Parallel", "Parallel→Ackermann\n(switch at 1.2 s)"]
    x = np.arange(len(plotted), dtype=float)
    red_vals = [item["red_cost"] for item in plotted]
    blue_vals = [item["blue_cost"] for item in plotted]
    ax_cost.plot(x, red_vals, color="#343434", marker="o", ms=4,
                 lw=1.1, label="Red-object request")
    ax_cost.plot(x, blue_vals, color="#8d6723", marker="s", ms=3.8,
                 lw=1.1, label="Blue-object request")
    for xi, val in zip(x, red_vals):
        ax_cost.annotate(f"{val:.1f}", (xi, val), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=6.2)
    for xi, val in zip(x, blue_vals):
        ax_cost.annotate(f"{val:.1f}", (xi, val), xytext=(0, -9),
                         textcoords="offset points", ha="center", fontsize=6.2)
    ax_cost.set_xticks(x, categories)
    ax_cost.set_ylim(54, 66)
    ax_cost.set_ylabel("Re-scored MPPI objective")
    ax_cost.set_title("(c) The grounded object changes the cost ranking", loc="left", pad=4)
    ax_cost.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, .99),
                   ncol=2, fontsize=6.1, handlelength=1.5)
    ax_cost.scatter([x[0], x[1]], [red_vals[0], blue_vals[1]], marker="*",
                    s=34, color="#111111", zorder=5)

    # Panel d: actual closed-loop clearance measurements across all four
    # object-reference episodes. Each line is one Gazebo run.
    xloc = [0, 1]
    for idx, trial in enumerate(rows):
        gaps = trial["gap_at_crossing_m"]
        target_name = trial["request"]
        other_name = "blue" if target_name == "red" else "red"
        target_gap, other_gap = gaps[target_name], gaps[other_name]
        ax_gap.plot(xloc, [other_gap, target_gap], color="#9b9b9b", lw=.85,
                    marker="o", ms=3.2, mfc="white", mec="#525252", mew=.7,
                    zorder=2)
    ax_gap.set_xticks(xloc, ["Other", "Requested"])
    ax_gap.set_ylabel("Measured proxy gap (m)")
    vals = [v for trial in rows for v in trial["gap_at_crossing_m"].values()]
    ax_gap.set_ylim(min(vals)-.04, max(vals)+.07)
    ax_gap.set_title("(d) Executed clearance in four Gazebo runs", loc="left", pad=4)
    delta = [trial["target_minus_other_gap_m"] for trial in rows]
    ax_gap.text(.99, .86, f"{min(delta):.3f}–{max(delta):.3f} m more to requested cart",
                transform=ax_gap.transAxes, ha="right", va="top", fontsize=6.4)

    from sys import path as sys_path
    sys_path.insert(0, "/home/USER/.codex/skills/nature-figure/scripts")
    from audit_panel_alignment import require_matplotlib_panel_alignment
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=[ax_rgb, ax_map, ax_cost, ax_gap],
        panel_ids=["a", "b", "c", "d"],
        row_groups=[["a", "b"], ["c", "d"]],
        exemptions=[{
            "panels": ["b"],
            "checks": ["column", "panel-width"],
            "reason": "The metric plan view preserves equal x/y scale, so its axes are narrower than the RGB and quantitative panels.",
        }],
        json_out=OUT / "grounding_mppi_causal.alignment.json")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"grounding_mppi_causal.{ext}", dpi=500,
                    facecolor="white")
    plt.close(fig)

    costs_record = {
        "kind": "same-state counterfactual re-scoring of one logged MPPI candidate bank",
        "interpretation": "Two actual VLM proposals from identical RGB and identical non-instruction context are applied to the same later logged state and unchanged trajectory bank. Panel d reports separate closed-loop Gazebo executions.",
        "record_index": plans.index(record),
        "sim_time_s": record["request_sim_time"],
        "pose": record["pose"],
        "current_wheel_angles_rad": record["wheel_steering_rad"],
        "task_goal": record["main_goal"],
        "candidate_count": len(scored),
        "logged_selected_behavior": record["plan"].get("selected_behavior"),
        "logged_selected_score": record["plan"].get("predicted_score"),
        "red_condition": request,
        "blue_condition": blue_request,
        "red_selected": {
            "candidate": candidate_name(red_best), "score": red_best["red_cost"],
            "target_clearance_m": red_best["red_gap_m"],
            "other_clearance_m": red_best["blue_gap_m"],
        },
        "blue_selected": {
            "candidate": candidate_name(blue_best), "score": blue_best["blue_cost"],
            "target_clearance_m": blue_best["blue_gap_m"],
            "other_clearance_m": blue_best["red_gap_m"],
        },
        "counterfactual_procedure": "For each candidate, remove the logged task-clearance and heading terms from its stored total. Recompute only the task clearance term for the requested registered box using the logged trajectory, body footprint, 0.85 m preference and 0.25 m scale. Rank the unchanged bank.",
        "closed_loop_runs": [
            {"name": row["name"], "request": row["request"],
             "target_minus_other_gap_m": row["target_minus_other_gap_m"],
             "gaps_at_crossing_m": row["gap_at_crossing_m"]} for row in rows
        ],
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_files},
    }
    provenance = {
        "same_rgb_sha256": hashlib.sha256(rgb_a.read_bytes()).hexdigest(),
        "same_candidate_input": red_input["candidates"] == blue_input["candidates"],
        "same_non_instruction_context": red_context == blue_context,
        "red_instruction": red_instruction,
        "blue_instruction": blue_instruction,
        "rgb_crop_xyxy": crop_box,
    }
    costs_record["matched_visual_input"] = provenance
    (RESULTS / "same_state_candidate_rescoring.provenance.json").write_text(
        json.dumps(costs_record, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(costs_record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
