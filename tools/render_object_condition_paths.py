"""Matched executed paths under two grounded object references."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic')]
from robotics_plot_primitives import vehicle
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
B=ROOT/'artifacts/object_reference_20261026/pilot_v1'
plt.rcParams.update({'font.family':'Liberation Serif','font.size':7.5,'axes.labelsize':7.5,'xtick.labelsize':6.8,'ytick.labelsize':6.8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(3.42,2.25));fig.subplots_adjust(left=.16,right=.98,bottom=.2,top=.85)
for x,y,label in [(6.5,-.95,'1'),(6.5,-3.25,'2')]:
 ax.add_patch(Rectangle((x-.35,y-.21),.7,.42,fc='#dddddd',ec='#555555',lw=.65));ax.text(x,y,label,ha='center',va='center',fontsize=7)
files=[]
for name,label,color,ls in [('red','Object 1','#333333','--'),('blue','Object 2','#98711d','-')]:
 p=B/f'object_reference_a_{name}_s41/episode/trace.jsonl';files.append(p)
 rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()];xy=np.asarray([r['pose'][:2] for r in rows]);ax.plot(xy[:,0],xy[:,1],color=color,ls=ls,lw=1.3,label=label)
 row=min(rows,key=lambda r:abs(r['pose'][0]-6.5));vehicle(ax,row,color,width=.65)
ax.set(xlim=(4.5,8.4),ylim=(-3.8,-.4),xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal',adjustable='box');ax.set_xticks([5,6,7,8]);ax.set_yticks([-3,-2,-1]);ax.tick_params(length=2,pad=1)
ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.02),fontsize=7,handlelength=1.5)
out=ROOT/'paper/figures/object_condition_paths'
for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
out.with_suffix('.provenance.json').write_text(json.dumps({'kind':'Separate executed object-reference trials in same scene','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'object_boxes':'registered proxy half dimensions 0.35 x 0.21 m'},indent=2))
