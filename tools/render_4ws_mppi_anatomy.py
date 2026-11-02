"""Render one recorded, same-state MPPI comparison in world XY coordinates."""
from pathlib import Path
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts/task_condition_benchmark_20261023/task_matrix_v8/heading_passing_heading_clearance_hybrid_s53/episode/hybrid_rollouts.jsonl.gz"
OUT = ROOT / "paper/figures/4ws_mppi_anatomy"
SOURCE_TIME = 15.48

with gzip.open(SRC, "rt") as stream:
    records = map(json.loads, stream)
    row = next(r for r in records if abs(r.get("request_sim_time", -1) - SOURCE_TIME) < 0.02)

plan = row["plan"]
candidates = plan["hybrid_schedule_candidates"]
selected = next(c for c in candidates if c.get("feasible") and c["structure"] == ["P", "A", "R"] and c["switch_steps"] == [10, 30])
alternative = next(c for c in candidates if c.get("feasible") and c["structure"] == ["A", "P", "R"] and c["switch_steps"] == [10, 30])
trace_path = SRC.parent / "trace.jsonl"
trace_rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
obs_row = min(trace_rows, key=lambda r: abs(float(r["timestamp"]) - SOURCE_TIME))
trolley = next(t for t in obs_row["tracks"] if t["evidence_id"] == "track_1")

plt.rcParams.update({
    "font.family": "Liberation Serif",
    "font.size": 7.2,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "axes.linewidth": 0.65,
})

fig, ax = plt.subplots(figsize=(3.42, 2.52))
fig.subplots_adjust(left=.15, right=.985, bottom=.19, top=.98)
ink = "#242424"
gray = "#858585"
ochre = "#8D6500"

def body(ax, pose, wheel_angles, edge, lw=0.9, alpha=1.0, z=4, draw_body=True):
    x, y, yaw = pose
    transform = Affine2D().rotate(yaw).translate(x, y) + ax.transData
    if draw_body:
        ax.add_patch(Rectangle(
            (-.751 / 2, -.500 / 2), .751, .500,
            fill=False, edgecolor=edge, linewidth=lw, alpha=alpha,
            transform=transform, zorder=z,
        ))
    wheel_xy = ((.247, .182), (.247, -.182), (-.247, .182), (-.247, -.182))
    for (wx, wy), angle in zip(wheel_xy, wheel_angles):
        half = .074
        dx, dy = half * math.cos(angle), half * math.sin(angle)
        ax.plot([wx-dx, wx+dx], [wy-dy, wy+dy], color=edge, linewidth=1.35,
                alpha=alpha, solid_capstyle="butt", transform=transform, zorder=z+1)

sel_xy = np.asarray(selected["trajectory"])
alt_xy = np.asarray(alternative["trajectory"])
ax.plot(alt_xy[:, 0], alt_xy[:, 1], linestyle=(0, (3, 2)), color=gray,
        linewidth=1.15, zorder=2)
ax.plot(sel_xy[:, 0], sel_xy[:, 1], color=ochre, linewidth=1.8, zorder=3)

# The observed trolley and the two predicted sequence rollouts share one map.
from matplotlib.patches import Circle
ax.add_patch(Circle(trolley["position"], float(trolley["radius"]),
                    facecolor="#eeeeee", edgecolor=ink, linewidth=.75, zorder=1))
ax.text(trolley["position"][0], trolley["position"][1], "T", ha="center", va="center",
        fontsize=6.2, color=ink, zorder=2)

# Mark both transition locations and their actual wheel orientations.
for cand, color in ((selected, ochre), (alternative, gray)):
    tr = np.asarray(cand["trajectory"])
    for idx in cand["switch_steps"]:
        ax.scatter(*tr[idx, :2], marker="D", s=17, facecolor="white",
                   edgecolor=color, linewidth=.9, zorder=6)

# Initial measured chassis and wheel state, plus terminal predicted configurations.
body(ax, row["pose"], row["wheel_steering_rad"], ink, lw=1.0, z=5)
# Selected rollout footprints at the first transition and the final prediction
# expose both body orientation and the four propagated wheel directions.
for idx in (selected["switch_steps"][0], len(sel_xy)-1):
    body(ax, sel_xy[idx], selected["steering_trace"][min(idx, len(selected["steering_trace"])-1)], ochre,
         lw=.65, alpha=.72, z=4)

# Compact legend carries the same-call objective and alignment result.
ax.plot([], [], color=ochre, lw=1.8,
        label=f"Selected P→C→R  ({selected['cost']:.1f})")
ax.plot([], [], color=gray, lw=1.15, linestyle=(0, (3, 2)),
        label=f"Alternative C→P→R  ({alternative['cost']:.1f})")
ax.legend(loc="upper left", bbox_to_anchor=(.01, .99), frameon=False,
          fontsize=6.0, handlelength=1.4, borderpad=0, labelspacing=.2)

ax.set_xlim(4.45, 8.35)
ax.set_ylim(-4.38, -1.35)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x (m)", labelpad=1, fontsize=7)
ax.set_ylabel("y (m)", labelpad=1, fontsize=7)
ax.tick_params(labelsize=6.2, length=2, pad=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUT.with_suffix(f".{ext}"), dpi=500)

OUT.with_suffix(".provenance.json").write_text(json.dumps({
    "kind": "Same-state recorded MPPI candidate comparison with measured initial four-wheel state and predicted terminal body headings.",
    "source": str(SRC.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
    "observation_source": str(trace_path.relative_to(ROOT)),
    "observed_trolley": trolley,
    "request_sim_time_s": row["request_sim_time"],
    "initial_pose": row["pose"],
    "initial_wheel_angles_rad": row["wheel_steering_rad"],
    "applied_task_condition": plan["applied_motion_condition"],
    "selected": {k: selected[k] for k in ("structure", "switch_steps", "switch_times_s", "cost", "alignment_s", "trajectory", "steering_trace")},
    "alternative": {k: alternative[k] for k in ("structure", "switch_steps", "switch_times_s", "cost", "alignment_s", "trajectory", "steering_trace")},
    "wheelbase_m": 0.494,
    "track_m": 0.364,
    "footprint_m": [0.751, 0.500],
    "figure_note": "All chassis poses, initial wheel orientations and paths are in the recorded world XY frame. No trajectory or footprint was repositioned for illustration.",
}, indent=2) + "\n")
