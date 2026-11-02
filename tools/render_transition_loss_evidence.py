from pathlib import Path
import json,math,sys,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic'),'/home/USER/.codex/skills/nature-figure/scripts']
from robotics_plot_primitives import draw_map,vehicle
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_panel_alignment import require_matplotlib_panel_alignment
base=ROOT/'artifacts/transition_target_pairs_20261027/task_pairs_v5'; configs=[('fixed','Fixed transition','#3b3b3b','--'),('hybrid','Searched transition','#9a7519','-')]
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.1,3.35));gs=fig.add_gridspec(2,2,width_ratios=[1.1,1],left=.09,right=.98,top=.90,bottom=.19,wspace=.36,hspace=.30)
a=fig.add_subplot(gs[:,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,1]);graph=SpatialGraph(base/'target1_fixed_s53/world.sdf');draw_map(a,graph.static_boxes,(10.8,13.35,-1.9,.9));metrics=[];sources=[]
for key,label,color,ls in configs:
 p=base/f'target1_{key}_s53/episode/trace.jsonl';sources.append(p);rows=[json.loads(l) for l in p.open() if l.strip()];xy=np.array([r['pose'][:2] for r in rows]);t=np.array([r['timestamp']-rows[0]['timestamp'] for r in rows]);yaw=np.unwrap(np.array([r['pose'][2] for r in rows]));w=np.array([r['actuation']['executed']['wz'] for r in rows]); modes=[r['actuation']['mode'] for r in rows];dt=np.diff(t);spinning=sum(dt[i] for i in range(len(dt)) if modes[i]=='SPINNING'); switches=sum(modes[i]!=modes[i-1] for i in range(1,len(modes)))
 a.plot(xy[:,0],xy[:,1],color=color,ls=ls,lw=1.4,zorder=3);idx=next(i for i,r in enumerate(rows) if math.dist(r['pose'][:2],(12.8,.55))<=.18);vehicle(a,rows[idx],color,width=.75)
 b.plot(t,np.rad2deg(yaw),color=color,ls=ls,lw=1.1);c.plot(t,w,color=color,ls=ls,lw=.95)
 metrics.append({'method':key,'trace_duration_s':float(t[-1]),'spinning_s':float(spinning),'mode_state_changes':switches,'path_m':float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),'first_near_goal_heading_error_deg':abs(math.degrees(math.atan2(math.sin(yaw[idx]-math.pi/2),math.cos(yaw[idx]-math.pi/2))))})
a.plot(12.8,.55,'*',color='#222222',ms=6);a.set(xlabel='x (m)',ylabel='y (m)');a.set_title('(a) Executed trajectories',loc='left',fontsize=8)
b.axhline(90,c='#bcbcbc',lw=.6,ls=':');b.set(ylabel='Heading (deg)',xlim=(0,34));b.set_title('(b) Body heading',loc='left',fontsize=8);b.tick_params(labelbottom=False)
c.axhline(0,c='#bcbcbc',lw=.6);c.set(xlabel='Elapsed time (s)',ylabel='Yaw rate (rad/s)',xlim=(0,34));c.set_title('(c) Executed yaw rate',loc='left',fontsize=8)
for ax in (b,c):ax.spines[['top','right']].set_visible(False)
fig.legend([Line2D([0],[0],c=x[2],ls=x[3],lw=1.2) for x in configs],[x[1] for x in configs],loc='lower center',bbox_to_anchor=(.5,.025),ncol=2,frameon=False,fontsize=8)
out=ROOT/'results/local_motion_preview/transition_loss_evidence';require_matplotlib_panel_alignment(fig,json_out=str(out)+'.alignment.json',axes=[b,c],panel_ids=['heading','yaw_rate'],column_groups=[['heading','yaw_rate']],row_groups=[],strict=True)
for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=400)
Path(str(out)+'.provenance.json').write_text(json.dumps({'task':'matched offset entry pose; no VLM manipulation','note':'Model-based simulated wheel states. Configurations also differ in candidate allocation; cannot attribute results to transition timing in isolation.','metrics':metrics,'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
