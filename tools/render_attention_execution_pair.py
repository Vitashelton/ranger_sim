"""Same-scene online service-request pair, using synchronized actor and robot logs."""
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic')]
from robotics_plot_primitives import draw_map,vehicle
from plot_nav2_matched_cases import corners
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
ap=argparse.ArgumentParser();ap.add_argument('--pair',default='pair_v2');args=ap.parse_args()
base=ROOT/'artifacts/task_attention_service_20261030'/args.pair
world=base/'ordinary/world.sdf';ref=json.load(open(base/'static_reference.json'))['path_xy'];ref=np.asarray(ref,float)
plt.rcParams.update({'font.family':'Liberation Serif','font.size':7.3,'axes.labelsize':7.3,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,1,figsize=(3.42,4.6));fig.subplots_adjust(left=.16,right=.97,bottom=.09,top=.96,hspace=.24)
meta=[];sources=[world,base/'static_reference.json'];lim=(4.3,8.35,-3.95,-1.20)
for ax,name,title,color in zip(axs,('ordinary','fragile'),('Ordinary delivery','Fragile trolley'),('#484848','#98711d')):
 run=base/name;p=run/'episode/trace.jsonl';c=run/'clearance_trace.json';sources += [p,c];rows=[json.loads(l) for l in p.read_text().splitlines()];clear=json.load(open(c));g=min(clear,key=lambda x:x['trolley_signed_SAT_gap_m']);row=min(rows,key=lambda x:abs(x['timestamp']-g['timestamp']));xy=np.asarray([r['pose'][:2] for r in rows]);draw_map(ax,SpatialGraph(world).static_boxes,lim);ax.plot(ref[:,0],ref[:,1],color='#bbbbbb',ls=':',lw=.8);ax.plot(xy[:,0],xy[:,1],color=color,lw=1.25);ax.add_patch(Polygon(corners(g['trolley_pose'],.35,.21),closed=True,fc='#e2e2e2',ec='#666666',lw=.65));vehicle(ax,row,color,width=.8);ax.text(.98,.93,f'{math.degrees(row["pose"][2]):.1f}°',transform=ax.transAxes,ha='right',va='top',fontsize=7);ax.set_title(title,fontsize=7.5,loc='left',pad=2);ax.set(xlim=lim[:2],ylim=lim[2:],ylabel='y (m)');ax.set_aspect('equal',adjustable='box');ax.set_xticks([5,6,7,8]);ax.set_yticks([-3,-2]);ax.tick_params(length=2,pad=1);meta.append({'run':name,'time':g['timestamp'],'signed_gap_m':g['trolley_signed_SAT_gap_m'],'robot_pose':row['pose'],'trolley_pose':g['trolley_pose'],'executed_mode':row['actuation']['mode']})
axs[0].tick_params(labelbottom=False);axs[1].set_xlabel('x (m)');out=ROOT/'paper/figures/task_attention_execution_pair'
for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
out.with_suffix('.provenance.json').write_text(json.dumps({'kind':'Two independent closed-loop Gazebo trials under matched initial conditions and different natural-language requests','cases':meta,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
