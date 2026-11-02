"""Two matched planner executions in a common corridor map."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic')]
from robotics_plot_primitives import draw_map,vehicle
from plot_nav2_matched_cases import corners
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
B=ROOT/'artifacts/task_condition_benchmark_20261023/task_matrix_v8';R=B/'heading_passing_heading_clearance_hybrid_s53/episode/route_guidance.jsonl'
read=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
world=B/'heading_passing_heading_clearance_hybrid_s53/world.sdf';ref=np.asarray(read(R)[0]['path'])
plt.rcParams.update({'font.family':'Liberation Serif','font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(3.42,3.35));fig.subplots_adjust(left=.15,right=.98,bottom=.12,top=.97,hspace=.28)
files=[world,R]
for ax,(method,label,color) in zip(axs,[('nav2','Nav2 Omni MPPI','#444444'),('hybrid','Task-Attentive MPPI','#98711d')]):
 run=B/f'heading_passing_heading_clearance_{method}_s53';p=run/'episode/trace.jsonl';g=run/'clearance_trace.json';files.extend([p,g]);rows=read(p);gap=min(json.loads(g.read_text()),key=lambda r:r['trolley_signed_SAT_gap_m']);row=min(rows,key=lambda r:abs(r['timestamp']-gap['timestamp']));xy=np.asarray([r['pose'][:2] for r in rows]);lim=(4.4,8.2,-3.85,-1.2)
 draw_map(ax,SpatialGraph(world).static_boxes,lim);ax.plot(ref[:,0],ref[:,1],color='#bbbbbb',ls='--',lw=.7);ax.plot(xy[:,0],xy[:,1],color=color,lw=1.35);ax.add_patch(Polygon(corners(gap['trolley_pose'],.35,.21),closed=True,fc='#dedede',ec='#555555',lw=.6));vehicle(ax,row,color,width=.75);ax.set_title(label,loc='left',fontsize=7.3,pad=2);ax.set(xlim=lim[:2],ylim=lim[2:],ylabel='y (m)');ax.set_aspect('equal',adjustable='box');ax.set_xticks([5,6,7,8]);ax.set_yticks([-3,-2]);ax.tick_params(length=2,pad=1)
axs[0].tick_params(labelbottom=False);axs[1].set_xlabel('x (m)')
out=ROOT/'paper/figures/planner_pair_paths'
for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
out.with_suffix('.provenance.json').write_text(json.dumps({'kind':'Matched separate runs with common reference and closest approach states','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
