"""Native-control geometry and real same-call predictive evidence, not execution."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/rangermini_dynamic_semantic'))
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, '/home/USER/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
from robotics_plot_primitives import vehicle
from plot_nav2_matched_cases import corners, clipped_polygon
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--rollout-log',type=Path)
args=parser.parse_args()
SOURCE = (args.rollout_log.resolve() if args.rollout_log else ROOT / 'artifacts/hybrid_sequence_20261022/offset_portal_domain_isolated_matrix_v6/offset_pose_hybrid_s37/episode/hybrid_rollouts.jsonl.gz')
WORLD = SOURCE.parent.parent / 'world.sdf'
OUT = ROOT / 'paper/figures/task_grounded_mppi_mechanism'
with gzip.open(SOURCE, 'rt') as stream:
    observations=[json.loads(line) for line in stream]
index=1
if args.rollout_log:
    eligible=[]
    for i,r in enumerate(observations):
        schedule=r['plan'].get('selected_schedule') or {}
        if len(schedule.get('switch_steps',[]))!=1 or schedule['switch_steps']==[20]:continue
        if any(c.get('feasible') and c['structure']==schedule['structure'] and c['switch_steps']==[20]
               for c in r['plan']['hybrid_schedule_candidates']):eligible.append(i)
    if not eligible:raise ValueError('No selected two-phase prediction with an equal-boundary counterpart')
    index=eligible[0]
record=observations[index]
plan = record['plan']
schedule = plan['selected_schedule']
candidates = plan['hybrid_schedule_candidates']
selected = min((c for c in candidates if c.get('feasible') and c['structure'] == schedule['structure']
                and c['switch_steps'] == schedule['switch_steps']), key=lambda c: c['cost'])
equal = min((c for c in candidates if c.get('feasible') and c['structure'] == schedule['structure']
             and c['switch_steps'] == [20]), key=lambda c: c['cost'])
assert abs(selected['cost'] - plan['predicted_score']) < 1e-8
plt.rcParams.update({'font.family': ['Liberation Serif', 'Arial'], 'font.size': 7,
                     'axes.labelsize': 8, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.linewidth': .65,
                     'axes.spines.top': False, 'axes.spines.right': False})
teal, orange, dark = '#16858D', '#DE8646', '#374248'
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15))
fig.subplots_adjust(left=.065, right=.985, bottom=.25, top=.84, wspace=.49)
a, b, c = axes

# Actual separately evaluated objectives for this saved motion structure.
points=[x for x in candidates if x.get('feasible') and x['structure']==schedule['structure']]
a.scatter([x['switch_steps'][0] for x in points],[x['cost'] for x in points],
          s=19,c='#8B949B',zorder=2)
a.scatter(equal['switch_steps'][0],equal['cost'],s=28,c=orange,marker='s',zorder=3)
a.scatter(selected['switch_steps'][0],selected['cost'],s=30,c=teal,marker='s',zorder=4)
a.set(xlabel='Transition index',ylabel='Rollout objective',
      xlim=(7,34),xticks=[10,20,30])
a.set_title('(a) Evaluated boundaries',loc='left',fontsize=7.5,pad=7)

# Measured initial state, saved predicted candidates; no fabricated path.
xy = np.concatenate([np.asarray(x['trajectory'])[:, :2] for x in (selected, equal)])
cx, cy = (xy.min(axis=0) + xy.max(axis=0)) / 2
span = max(2.6, np.ptp(xy[:, 0]) + 1.2, np.ptp(xy[:, 1]) + 1.2)
lim = (cx-span/2, cx+span/2, cy-span/2, cy+span/2)
for x, y, yaw, hx, hy in SpatialGraph(WORLD).static_boxes:
    polygon = clipped_polygon(corners((x, y, yaw), hx, hy), lim)
    if polygon:
        b.add_patch(Polygon(polygon, fc='#D1D5D7', ec='#8C959A', lw=.4, zorder=0))
for sample, color, ls, label in [(equal, orange, '--', 'Equal boundary'),
                                 (selected, teal, '-', 'Selected')]:
    trajectory = np.asarray(sample['trajectory'])
    b.plot(trajectory[:, 0], trajectory[:, 1], color=color, ls=ls, lw=1.35, label=label)
    boundary = sample['switch_steps'][0]-1
    vehicle(b,dict(pose=trajectory[boundary],actuation={'wheel_steering_rad':sample['steering_trace'][boundary]}),color,width=.6)
    b.scatter(*trajectory[boundary, :2], c=color, s=14, marker='s', zorder=5)
    for k in (boundary, len(trajectory)-1):
        b.arrow(trajectory[k, 0], trajectory[k, 1], .13*np.cos(trajectory[k, 2]),
                .13*np.sin(trajectory[k, 2]), color=color, lw=.6,
                head_width=.025, length_includes_head=True, zorder=5)
b.set(xlim=lim[:2], ylim=lim[2:], xlabel='x (m)', ylabel='y (m)')
b.set_aspect('equal', adjustable='datalim')
handles,labels=b.get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=6.5,frameon=False,
           bbox_to_anchor=(.5,-.005),handlelength=1.5)
b.set_title('(b) Predicted motion', loc='left', fontsize=7.5, pad=7)

# The same front-left channel and steering-hold records for both candidates.
for sample, color, ls in [(equal, orange, '--'), (selected, teal, '-')]:
    wheel = np.asarray(sample['steering_trace'])[:, 0]
    t = np.arange(1, len(wheel)+1) * .12
    c.plot(np.r_[0., t], np.rad2deg(np.r_[record['wheel_steering_rad'][0], wheel]),
           color=color, ls=ls, lw=1.2)
    switch = sample['switch_times_s'][0]
    c.axvline(switch, color=color, ls=ls, lw=.6, alpha=.75)
    holds = np.asarray(sample['alignment_trace'], dtype=bool)
    y = .09 if sample is equal else .035
    for k in np.flatnonzero(holds):
        c.plot([k*.12, (k+1)*.12], [y, y], transform=c.get_xaxis_transform(),
               color=color, lw=2, clip_on=False, solid_capstyle='butt')
c.set(xlabel='Prediction time (s)', ylabel=r'$\delta_{FL}$ (deg)', xlim=(0, 4.8), ylim=(-65, 35))
c.set_title('(c) Steering preparation', loc='left', fontsize=7.5, pad=7)

require_matplotlib_panel_alignment(fig, json_out=str(OUT)+'.alignment.json', strict=True)
fig.savefig(str(OUT)+'.pdf')
fig.savefig(str(OUT)+'.svg')
fig.savefig(str(OUT)+'.png', dpi=600)
provenance = dict(kind='All panels use one saved optimizer call: evaluated objectives, predicted body/wheel poses and steering preparation.',
    source=str(SOURCE.relative_to(ROOT)), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    world_sha256=hashlib.sha256(WORLD.read_bytes()).hexdigest(), observation_index=index,
    renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    selection_rule='First eligible saved return with a selected two-phase structure, a non-midpoint boundary and a feasible equal-phase counterpart.',
    selected_schedule=schedule, selected_cost=selected['cost'], equal_cost=equal['cost'],
    selected_alignment_s=selected['alignment_s'], equal_alignment_s=equal['alignment_s'],
    interpretation='The equal and selected candidates have separately evaluated controls. This figure explains structure/steering-aware evaluation; it does not isolate a universal timing benefit or establish a broad benchmark.',
    figure_contract=dict(question='How does the evaluated transition boundary change the motion and preparation cost?',
        panel_roles=['Actual boundary-index objective evaluations for the selected structure',
                     'Same-state predicted motion with two phase boundaries',
                     'Wheel-state propagation and preparation intervals'],
        style='IEEE width, white background, user-requested serif, teal/orange; no AI imagery.'))
Path(str(OUT)+'.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
print(json.dumps({k: provenance[k] for k in ('selected_schedule', 'selected_cost', 'equal_cost')}, indent=2))

# Compact manuscript variant keeps evaluated costs and the steering mechanism.
fig,axes=plt.subplots(1,2,figsize=(3.45,1.8))
fig.subplots_adjust(left=.14,right=.99,bottom=.31,top=.83,wspace=.75)
a,c=axes
points=[x for x in candidates if x.get('feasible') and x['structure']==schedule['structure']]
a.scatter([x['switch_steps'][0] for x in points],[x['cost'] for x in points],s=12,c='#8B949B')
for sample,color in ((equal,orange),(selected,teal)):
    a.scatter(sample['switch_steps'][0],sample['cost'],c=color,s=19,marker='s',zorder=4)
a.set(xlabel='Transition index',ylabel='Objective',xlim=(7,34),xticks=[10,20,30])
a.set_title('(a) Boundary costs',loc='left',fontsize=7,pad=5)
for sample,color,ls in ((equal,orange,'--'),(selected,teal,'-')):
    wheel=np.asarray(sample['steering_trace'])[:,0];t=np.arange(1,len(wheel)+1)*.12
    c.plot(np.r_[0.,t],np.rad2deg(np.r_[record['wheel_steering_rad'][0],wheel]),color=color,ls=ls,lw=1.)
    c.axvline(sample['switch_times_s'][0],color=color,ls=ls,lw=.5)
c.set(xlabel='Prediction time (s)',ylabel=r'$\delta_{FL}$ (deg)',xlim=(0,4.8),xticks=[0,2,4],yticks=[-60,0,30])
c.set_title('(b) Wheel preparation',loc='left',fontsize=7,pad=5)
fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=6.5,frameon=False,bbox_to_anchor=(.5,-.02))
for ext in ('pdf','svg','png'):fig.savefig(str(OUT)+'_compact.'+ext,dpi=600)
Path(str(OUT)+'_compact.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

# Manuscript mechanism: spatial consequence paired with steering state.
# Objective samples remain in the full auxiliary version.
fig,axes=plt.subplots(1,2,figsize=(3.45,2.15))
fig.subplots_adjust(left=.13,right=.99,bottom=.29,top=.86,wspace=.67)
b,c=axes
for box in SpatialGraph(WORLD).static_boxes:
    polygon=clipped_polygon(corners(box[:3],*box[3:]),lim)
    if polygon:b.add_patch(Polygon(polygon,fc='#E0E2E3',ec='#A7ADB1',lw=.35,zorder=0))
for sample,color,ls,label in ((equal,'#737A80','--','Equal phases'),(selected,teal,'-','Searched')):
    q=np.asarray(sample['trajectory']);b.plot(q[:,0],q[:,1],color=color,ls=ls,lw=1.0,label=label)
    k=sample['switch_steps'][0]-1
    vehicle(b,{'pose':q[k],'actuation':{'wheel_steering_rad':sample['steering_trace'][k]}},color,width=.55)
    t=np.arange(1,len(sample['steering_trace'])+1)*.12
    c.plot(np.r_[0,t],np.rad2deg(np.r_[record['wheel_steering_rad'][0],np.asarray(sample['steering_trace'])[:,0]]),
           color=color,ls=ls,lw=1.)
    c.axvline(sample['switch_times_s'][0],color=color,ls=ls,lw=.5)
b.set(xlim=lim[:2],ylim=lim[2:],xlabel='x (m)',ylabel='y (m)')
b.set_aspect('equal',adjustable='datalim')
b.set_title('(a) Predicted motion',loc='left',fontsize=7,pad=4)
c.set(xlabel='Prediction time (s)',ylabel=r'$\delta_{FL}$ (deg)',xlim=(0,4.8),xticks=[0,2,4],yticks=[-60,0,30])
c.set_title('(b) Wheel preparation',loc='left',fontsize=7,pad=4)
h,l=b.get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=2,frameon=False,fontsize=6.5,bbox_to_anchor=(.5,-.005))
require_matplotlib_panel_alignment(fig,json_out=str(OUT)+'_motion.alignment.json',strict=True)
for ext in ('pdf','svg','png'):fig.savefig(str(OUT)+'_motion.'+ext,dpi=500)
Path(str(OUT)+'_motion.provenance.json').write_text(json.dumps(dict(provenance,panel_roles=['Predicted path and wheel pose at transition','Steering preparation for the same candidates']),indent=2)+'\n')
