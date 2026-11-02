from pathlib import Path
import math,json,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic'),'/home/USER/.codex/skills/nature-figure/scripts']
from robotics_plot_primitives import draw_map,vehicle
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_panel_alignment import require_matplotlib_panel_alignment
base=ROOT/'artifacts/transition_target_pairs_20261027/task_pairs_v5'; graph=SpatialGraph(base/'target1_fixed_s53/world.sdf')
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(7.1,3.5));fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.87,wspace=.22)
colors={'DUAL_ACKERMANN':'#343434','PARALLEL':'#987525','SPINNING':'#a0a0a0','STOPPED':'#d4d4d4'};sources=[];items=[]
for ax,key,title in zip(axes,['fixed','hybrid'],['(a) Fixed transition','(b) Searched transition']):
 p=base/f'target1_{key}_s53/episode/trace.jsonl';sources.append(p);rows=[json.loads(l) for l in p.open() if l.strip()];xy=np.array([r['pose'][:2] for r in rows]);lim=(10.8,13.3,-1.9,.9)
 draw_map(ax,graph.static_boxes,lim)
 for i in range(len(rows)-1):ax.plot(xy[i:i+2,0],xy[i:i+2,1],color=colors.get(rows[i]['actuation']['mode'],'#888888'),lw=1.8,zorder=3)
 idx=next(i for i,r in enumerate(rows) if math.dist(r['pose'][:2],(12.8,.55))<=.18)
 dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy[:idx+1],axis=0),axis=1))]
 indices=sorted(set([int(np.argmin(abs(dist-f*dist[-1]))) for f in (.15,.58,1.0)]))
 for i in indices:vehicle(ax,rows[i],'#1b1b1b',width=.8)
 ax.plot(12.8,.55,marker='*',ms=7,color='#111111',zorder=8);ax.set_title(title,loc='left',fontsize=9);ax.set(xlabel='x (m)',ylabel='y (m)')
 err=abs(math.degrees(math.atan2(math.sin(rows[idx]['pose'][2]-math.pi/2),math.cos(rows[idx]['pose'][2]-math.pi/2))))
 items.append({'method':key,'entry_heading_error_deg':err,'sample_timestamps':[rows[i]['timestamp'] for i in indices]})
fig.legend([Line2D([0],[0],c=colors[k],lw=1.8) for k in ('DUAL_ACKERMANN','PARALLEL','SPINNING')],['Curved','Parallel','Reorientation'],loc='lower center',ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(.5,.025))
out=ROOT/'results/local_motion_preview/mppi_maneuver_pair'
require_matplotlib_panel_alignment(fig,json_out=str(out)+'.alignment.json',strict=True)
for ext in ('png','svg','pdf'):fig.savefig(str(out)+'.'+ext,dpi=400)
Path(str(out)+'.provenance.json').write_text(json.dumps({'source':'Matched independent Gazebo episodes, not an online VLM test','comparison':'Fixed and searched transition variants differ in candidate allocation as well as transition search','goal':[12.8,.55],'goal_heading_deg':90,'entry_definition':'First recorded pose within 0.18 m of goal position','results':items,'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
