"""Task-condition effect on one newly logged MPPI candidate bank."""
from pathlib import Path
import gzip,hashlib,json,math,sys
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
run=ROOT/'artifacts/task_attention_service_20261030/pair_v4/fragile';ep=run/'episode';source=ep/'hybrid_rollouts.jsonl.gz';time=25.08
with gzip.open(source,'rt') as f:
 rec=min((json.loads(line) for line in f),key=lambda row:abs(row['request_sim_time']-time))
plan=rec['plan'];condition=plan['applied_motion_condition'];assert condition['heading_policy']=='route_heading' and condition['relevant_object_ids']==['track_1']
with gzip.open(ep/'planner_observations.jsonl.gz','rt') as stream:
 obs=min((json.loads(line) for line in stream),key=lambda x:abs(x['sim_time']-rec['request_sim_time']))
track=next(t for t in obs['tracks'] if t['evidence_id']=='track_1')
bank=plan.get('candidate_summary',[])+[x for x in plan.get('hybrid_schedule_candidates',[]) if x.get('feasible')]
bank=[x for x in bank if isinstance(x.get('cost'),(int,float)) and x.get('trajectory') and math.isfinite(x['cost'])]
for c in bank:
 tr=np.asarray(c['trajectory']);dt=.12;dur=np.arange(len(tr))*dt;age=max(0,rec['request_sim_time']-track['stamp'])
 centers=np.asarray(track['position'])+(dur+age)[:,None]*np.asarray(track['velocity'])
 dx,dy=centers[:,0]-tr[:,0],centers[:,1]-tr[:,1];co,si=np.cos(tr[:,2]),np.sin(tr[:,2]);unc=np.minimum(.16,.025+.035*dur)
 gap=np.hypot(np.maximum(np.abs(co*dx+si*dy)-.751/2,0),np.maximum(np.abs(-si*dx+co*dy)-.5/2,0))-track['radius']-unc
 def score(target): return float(np.sum((np.maximum(target-gap,0)/.15)**2))*dt*9
 actual=score(condition['preferred_clearance_m']);ordinary=score(.65)
 assert abs(actual-c['cost_terms']['dynamic'])<1e-4
 c['ordinary_cost']=c['cost']-actual+ordinary
a=min(bank,key=lambda x:x['cost']);b=min(bank,key=lambda x:x['ordinary_cost'])
assert a.get('structure')==['P','A','R'] and b.get('behavior')=='PARALLEL'
trace=[json.loads(x) for x in (ep/'trace.jsonl').read_text().splitlines()];actual=min(trace,key=lambda x:abs(x['timestamp']-rec['request_sim_time']))
gaps=json.loads((run/'clearance_trace.json').read_text());gap=min(gaps,key=lambda x:abs(x['timestamp']-rec['request_sim_time']))
ref=[json.loads(x) for x in (ep/'route_guidance.jsonl').read_text().splitlines()];ref=min(ref,key=lambda x:abs(x['sim_time']-rec['request_sim_time']))
plt.rcParams.update({'font.family':'Liberation Serif','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(3.42,3.4));fig.subplots_adjust(left=.15,right=.98,bottom=.13,top=.80)
lim=(4.75,7.65,-4.05,-1.55);draw_map(ax,SpatialGraph(run/'world.sdf').static_boxes,lim)
path=np.asarray(ref['path']);ax.plot(path[:,0],path[:,1],color='#c0c0c0',ls=':',lw=.75,zorder=1)
ax.add_patch(Polygon(corners(gap['trolley_pose'],.35,.21),closed=True,fc='#dddddd',ec='#777777',lw=.6,zorder=2))
for c,color,style,label in [(a,'#96701a','-','Fragile request: P→C→R'),(b,'#555555','--','Ordinary clearance: P')]:
 tr=np.asarray(c['trajectory']);ax.plot(tr[:,0],tr[:,1],color=color,ls=style,lw=1.35,label=label,zorder=4)
 wheels=c.get('terminal_steering_angles') or (c.get('steering_trace') or [rec['wheel_steering_rad']])[-1]
 vehicle(ax,{'pose':tr[-1].tolist(),'actuation':{'wheel_steering_rad':wheels}},color,width=.65)
vehicle(ax,actual,'#222222',width=.8)
ax.set(xlim=lim[:2],ylim=lim[2:],xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal',adjustable='box');ax.set_xticks([5,6,7]);ax.set_yticks([-4,-3,-2]);ax.tick_params(length=2,pad=1)
ax.legend(frameon=False,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=6.6,ncol=1,handlelength=1.6)
out=ROOT/'paper/figures/task_attention_live_case'
for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
meta={'kind':'Same-bank counterfactual scoring of observed online clearance conditions from a new Gazebo pair','time':rec['request_sim_time'],'condition':condition,'fragile':{'structure':a['structure'],'cost':a['cost'],'terminal_pose':a['trajectory'][-1]},'ordinary':{'behavior':b['behavior'],'cost':b['ordinary_cost'],'terminal_pose':b['trajectory'][-1]},'note':'The alternative is a re-score of logged candidates, not a second executed trajectory. Only the relevant-object preferred clearance changes from 0.95 m to 0.65 m. The independently recomputed original dynamic cost matches each logged cost.','sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,ep/'planner_observations.jsonl.gz',ep/'trace.jsonl',run/'clearance_trace.json',ep/'route_guidance.jsonl',run/'world.sdf']}}
out.with_suffix('.provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
