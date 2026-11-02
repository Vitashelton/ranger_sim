"""Measured trajectory comparison and matched local-planner metrics."""
import argparse,hashlib,json,math,sys,csv,statistics
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'));sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,'/home/USER/.codex/skills/nature-figure/scripts')
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from plot_nav2_matched_cases import corners,clipped_polygon
from audit_panel_alignment import require_matplotlib_panel_alignment
p=argparse.ArgumentParser();p.add_argument('batch',type=Path);a=p.parse_args();batch=a.batch.resolve();out=batch/'analysis';out.mkdir(exist_ok=True)
protocol=json.loads((batch/'protocol.json').read_text());methods=protocol['methods'];plot_methods=[m for m in ('primitive','fixed','hybrid','nav2') if m in methods];seeds=protocol['seeds']
COL={'nav2':'#6D7479','primitive':'#D67A35','fixed':'#716484','hybrid':'#167F83'};NAME={'nav2':'Nav2 Omni/Ranger','primitive':'Primitive MPPI','fixed':'Fixed timing','hybrid':'Optimized timing'}
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
metrics=[];sources={};data={}
for seed in seeds:
 worlds=[]
 for method in methods:
  run=batch/f'offset_pose_{method}_s{seed}';ep=run/'episode'
  if not (ep/'result.json').exists():raise RuntimeError('Incomplete trial '+str(run))
  result=json.loads((ep/'result.json').read_text());trace=rows(ep/'trace.jsonl');graph=SpatialGraph(run/'world.sdf')
  for source in [ep/'result.json',ep/'trace.jsonl',run/'world.sdf']:
   sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
  worlds.append(sources[str((run/'world.sdf').relative_to(ROOT))]);active=[];hold_s=0.;minimum=math.inf;overlaps=0;hybrid_calls=0
  boxes=np.asarray(graph.static_boxes);ox,oy,oa,hx,hy=boxes.T
  for i,r in enumerate(trace):
   ac=r.get('actuation',{});v=ac.get('executed',{})
   if math.hypot(v.get('vx',0),v.get('vy',0))>.001 or abs(v.get('wz',0))>.001:active.append(ac.get('mode'))
   if i+1<len(trace) and ac.get('steering_transition_stop'):hold_s+=max(0,trace[i+1]['timestamp']-r['timestamp'])
   hybrid_calls+=r['plan'].get('selected_schedule') is not None
   x,y,yaw=r['pose'];dx,dy=ox-x,oy-y;c,s=math.cos(yaw),math.sin(yaw);co,si=np.abs(np.cos(yaw-oa)),np.abs(np.sin(yaw-oa));ca,sa=np.cos(oa),np.sin(oa)
   sep=np.maximum.reduce([np.abs(dx*c+dy*s)-.3755-hx*co-hy*si,np.abs(-dx*s+dy*c)-.25-hx*si-hy*co,np.abs(dx*ca+dy*sa)-hx-.3755*co-.25*si,np.abs(-dx*sa+dy*ca)-hy-.3755*si-.25*co])
   gap=float(sep.min());minimum=min(minimum,gap);overlaps+=gap<0
  record=dict(seed=seed,method=method,arrived=bool(result['task_success']),path_m=result['path_length_m'],elapsed_sim_s=result['elapsed_sim_s'],min_static_signed_SAT_m=minimum,static_overlap_pose_count=overlaps,nonzero_mode_changes=sum(x!=y for x,y in zip(active,active[1:])),sampled_alignment_s=hold_s,hybrid_selected_cycles=hybrid_calls,planning_mean_ms=result['mean_bank_planning_ms'],max_rollout_evaluations=max((r['plan'].get('rollout_evaluations',0) for r in trace),default=0),final_heading_error_deg=result['heading_error_deg'])
  metrics.append(record);data[seed,method]=(trace,graph,result)
 assert len(set(worlds))==1,'Matched methods have different world geometry'
summary=[]
for method in methods:
 mm=[r for r in metrics if r['method']==method];arrived=[r for r in mm if r['arrived']]
 summary.append(dict(method=method,n=len(mm),arrived=len(arrived),mean_arrived_path_m=statistics.mean(r['path_m'] for r in arrived) if arrived else None,mean_arrived_time_s=statistics.mean(r['elapsed_sim_s'] for r in arrived) if arrived else None,min_static_signed_SAT_m=min(r['min_static_signed_SAT_m'] for r in mm),mean_nonzero_mode_changes=statistics.mean(r['nonzero_mode_changes'] for r in mm),mean_sampled_alignment_s=statistics.mean(r['sampled_alignment_s'] for r in mm),mean_planning_ms=statistics.mean(r['planning_mean_ms'] for r in mm),hybrid_selected_cycles=sum(r['hybrid_selected_cycles'] for r in mm)))
(out/'metrics.json').write_text(json.dumps(dict(episodes=len(metrics),protocol=protocol,metrics=metrics,summary=summary,source_sha256=sources,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),definitions={'alignment':'left-held integration of logged steering_transition_stop over simulation timestamps','mode_changes':'adjacent changes among nonzero physically executed modes; excludes STOP/alignment-only samples','static_gap':'signed rectangle SAT over logged body poses and registered collision boxes; not mesh or continuous-time safety','path_time':'means over arrived trials; all attempts retained in per-trial metrics'}),indent=2)+'\n')
with (out/'metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=metrics[0]);w.writeheader();w.writerows(metrics)
tex=[r'\begin{table*}[t]\centering\footnotesize',r'\caption{Matched local-pose mechanism ablation, five seeds. $L$ is mean path length among arrivals; $d$ is minimum sampled static separation; $N$ is mean nonzero executed-mode changes. All attempts are retained.}\label{tab:hybridlocal}',r'\begin{tabular}{lrrrr}\toprule',r'Planner & Arrived & L (m) & d (m) & N \\ \midrule']
for r in summary:
 if r['method'] not in plot_methods:continue
 fmt=lambda v:'--' if v is None else f'{v:.2f}'
 vals=[fmt(r['mean_arrived_path_m']),f"{r['min_static_signed_SAT_m']:.3f}",fmt(r['mean_nonzero_mode_changes'])]
 tex.append(NAME[r['method']]+f" & {r['arrived']}/{r['n']} & "+' & '.join(vals)+r' \\')
tex+=[r'\bottomrule\end{tabular}\end{table*}'];(out/'results_table.tex').write_text('\n'.join(tex)+'\n')
plt.rcParams.update({'font.family':'Liberation Sans','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none'})
fig=plt.figure(figsize=(7.16,2.75));gs=fig.add_gridspec(1,len(plot_methods),left=.065,right=.985,top=.82,bottom=.18,wspace=.23);axs=[fig.add_subplot(gs[0,i]) for i in range(len(plot_methods))]
all_poses=np.vstack([np.asarray([r['pose'] for r in trace]) for trace,_,_ in data.values()]);lim=(min(10.45,float(all_poses[:,0].min()-.25)),max(13.65,float(all_poses[:,0].max()+.25)),min(-2.55,float(all_poses[:,1].min()-.25)),max(1.05,float(all_poses[:,1].max()+.25)))
for ax,method in zip(np.atleast_1d(axs),plot_methods):
 graph=data[seeds[0],method][1]
 for x,y,yaw,hx,hy in graph.static_boxes:
  pts=clipped_polygon(corners((x,y,yaw),hx,hy),lim)
  if pts:ax.add_patch(Polygon(pts,fc='#C8CDD0',ec='#939BA0',lw=.3))
 for seed in seeds:
  trace,_,result=data[seed,method];poses=np.asarray([r['pose'] for r in trace]);visible=(poses[:,0]>=lim[0])&(poses[:,0]<=lim[1])&(poses[:,1]>=lim[2])&(poses[:,1]<=lim[3]);xy=np.where(visible[:,None],poses[:,:2],np.nan)
  ax.plot(xy[:,0],xy[:,1],c=COL[method],ls='-' if result['task_success'] else ':',lw=1.0,alpha=.62)
  if visible[-1]:ax.scatter(*poses[-1,:2],c=COL[method],marker='o' if result['task_success'] else 'x',s=9,zorder=4,alpha=.8)
 ax.scatter(11.,-1.65,s=12,c='#333333');ax.scatter(*protocol['goal'][:2],s=22,marker='*',c='#333333',zorder=4)
 ax.set(xlim=lim[:2],ylim=lim[2:],xlabel='x (m)');ax.set_aspect('equal',adjustable='box');ax.set_title(NAME[method],loc='left',fontsize=8)
 if method==plot_methods[0]:ax.set_ylabel('y (m)')
 ax.spines[['top','right']].set_visible(False)
fig.legend(handles=[Line2D([0],[0],c=COL[m],ls='-',lw=1.2,label=f'{NAME[m]} (5 seeds)') for m in plot_methods],loc='upper center',ncol=len(plot_methods),frameon=False,fontsize=7)
require_matplotlib_panel_alignment(fig,json_out=out/'trajectories.alignment.json',strict=True)
fig.savefig(out/'trajectories.pdf')
fig.savefig(out/'trajectories.svg')
fig.savefig(out/'trajectories.png',dpi=600)
print(json.dumps(summary,indent=2))
