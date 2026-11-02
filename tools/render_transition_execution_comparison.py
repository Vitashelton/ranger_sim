"""Matched executed entry trajectories and wheel-state views, from Gazebo logs."""
from pathlib import Path
import hashlib,json,sys,math
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

def main():
 base=ROOT/'artifacts/transition_target_pairs_20261027/task_pairs_v5';methods=[('fixed','Fixed boundaries','#c87d43','--'),('hybrid','Searched boundaries','#087d81','-')]
 paths={};source=[]
 for key,label,color,style in methods:
  run=base/f'target1_{key}_s53';p=run/'episode/trace.jsonl';paths[key]=[json.loads(l) for l in p.read_text().splitlines()];source.extend([p,run/'world.sdf'])
 graph=SpatialGraph(base/'target1_fixed_s53/world.sdf');plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig=plt.figure(figsize=(3.45,4.8));mainax=fig.add_axes([.16,.48,.78,.46]);insets=[fig.add_axes([.12,.08,.35,.27]),fig.add_axes([.55,.08,.35,.27])]
 wide=(10.8,13.35,-1.9,.85);zoom=(12.05,13.30,-.55,.85);draw_map(mainax,graph.static_boxes,wide);mainax.set_aspect('equal',adjustable='box')
 frames=[]
 for ax,(key,label,color,style) in zip(insets,methods):
  trace=paths[key];xy=np.asarray([r['pose'][:2] for r in trace]);mainax.plot(xy[:,0],xy[:,1],color=color,ls=style,lw=1.2)
  draw_map(ax,graph.static_boxes,zoom);ax.set_aspect('equal',adjustable='box');inside=(xy[:,0]>=zoom[0])&(xy[:,0]<=zoom[1])&(xy[:,1]>=zoom[2])&(xy[:,1]<=zoom[3]);x=np.where(inside,xy[:,0],np.nan);y=np.where(inside,xy[:,1],np.nan);ax.plot(x,y,color=color,ls=style,lw=1.1)
  row=next(r for r in trace if math.dist(r['pose'][:2],(12.8,.55)) <= .18);vehicle(ax,row,color,width=.8);ax.set_title(label,loc='left',fontsize=8,pad=1,color=color);ax.set_xticks([]);ax.set_yticks([]);ax.plot(12.8,.55,'*',color='#26343a',ms=5)
  frames.append(dict(method=key,pose=row['pose'],actuation=row['actuation'],timestamp=row['timestamp']))
 mainax.scatter([11],[-1.65],s=15,facecolors='white',edgecolors='#26343a',zorder=5);mainax.plot(12.8,.55,'*',color='#26343a',ms=7);mainax.set(xlabel='x (m)',ylabel='y (m)');mainax.set_title('(a) Matched executed trajectories',loc='left',fontsize=8,pad=2);mainax.tick_params(labelsize=7)
 fig.legend([Line2D([0],[0],color='#c87d43',ls='--'),Line2D([0],[0],color='#087d81')],['Fixed boundaries','Searched boundaries'],loc='lower center',bbox_to_anchor=(.52,.37),ncol=2,fontsize=7,frameon=False)
 require_matplotlib_panel_alignment(fig,json_out=ROOT/'paper/figures/transition_execution_comparison.alignment.json',axes=insets,panel_ids=['fixed','searched'],column_groups=[],row_groups=[['fixed','searched']],strict=True)
 out=ROOT/'paper/figures/transition_execution_comparison'
 for ext in ['pdf','svg','png']:fig.savefig(str(out)+'.'+ext,dpi=450)
 out.with_suffix('.provenance.json').write_text(json.dumps(dict(kind='Recorded Gazebo pose trajectories, same map and target; insets show actual model wheel state',matched_target=1,selected_initialization=53,views=frames,selection_rule='First logged pose within 0.18 m of the same goal in each independent run; not a simultaneous-time snapshot',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}),indent=2)+'\n')
if __name__=='__main__':main()
