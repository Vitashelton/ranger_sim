"""Plot measured planner trajectories for complete matched three-scene trials."""
import argparse,csv,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle
from matplotlib.lines import Line2D
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
sys.path.insert(0,'/home/USER/.codex/skills/nature-figure/scripts')
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_panel_alignment import require_matplotlib_panel_alignment
CASES=['static_detour','heading_passing','offset_portal']
METHODS=['nav2-omni','multimode','no-steering-prediction']
COLORS={'nav2-omni':'#D67A35','multimode':'#167F83','no-steering-prediction':'#70787E'}
NAMES={'nav2-omni':'Nav2 Omni MPPI','multimode':'Multimode MPPI','no-steering-prediction':'No steering prediction'}
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def corners(pose,hx=.3755,hy=.25):
 x,y,yaw=pose;c,s=math.cos(yaw),math.sin(yaw)
 return [(x+c*dx-s*dy,y+s*dx+c*dy) for dx,dy in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]]
def clipped_polygon(poly,lim):
 for dim,edge,greater in [(0,lim[0],True),(0,lim[1],False),(1,lim[2],True),(1,lim[3],False)]:
  out=[]
  for q,r in zip(poly,poly[1:]+poly[:1]):
   iq=q[dim]>=edge if greater else q[dim]<=edge;ir=r[dim]>=edge if greater else r[dim]<=edge
   if iq:out.append(q)
   if iq!=ir:
    f=(edge-q[dim])/(r[dim]-q[dim]);out.append(tuple(q[j]+f*(r[j]-q[j]) for j in (0,1)))
  poly=out
  if not poly:break
 return poly
def main():
 ap=argparse.ArgumentParser();ap.add_argument('batch',type=Path);ap.add_argument('--seed',type=int,default=17)
 ap.add_argument('--output',type=Path,default=ROOT/'paper/figures')
 ap.add_argument('--nav2-batch',type=Path,help='Corrected public-adapter batch; native method records remain frozen')
 a=ap.parse_args();a.batch=a.batch.resolve();a.output.mkdir(parents=True,exist_ok=True)
 protocol=json.loads((a.batch/'protocol.json').read_text());cases=protocol['scenarios'];data={};sources={};metrics=[]
 public_protocol=None
 if a.nav2_batch:
  a.nav2_batch=a.nav2_batch.resolve();public_protocol=json.loads((a.nav2_batch/'protocol.json').read_text())
  changed={k for k,v in protocol['source_sha256'].items() if public_protocol['source_sha256'].get(k)!=v}
  assert changed=={'src/rangermini_dynamic_semantic/rangermini_dynamic_semantic/nav2_omni_mppi.py'}, changed
  assert protocol['common']==public_protocol['common'] and protocol['budgets']==public_protocol['budgets']
 for scenario in cases:
  worlds=[]
  for method in METHODS:
   base=a.nav2_batch if a.nav2_batch and method=='nav2-omni' else a.batch
   run=base/f'{scenario}_{method}_s{a.seed}'
   if not (run/'episode/result.json').exists():raise RuntimeError('Matched trial incomplete: '+str(run))
   trace=rows(run/'episode/trace.jsonl');result=json.loads((run/'episode/result.json').read_text())
   gap=json.loads((run/'clearance_trace.json').read_text());graph=SpatialGraph(run/'world.sdf')
   for p in [run/'world.sdf',run/'episode/trace.jsonl',run/'episode/result.json',run/'clearance_trace.json']:
    sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
   worlds.append(sources[str((run/'world.sdf').relative_to(ROOT))])
   active=[];align_s=0.;static_min=math.inf;static_overlap=0
   boxes=np.asarray(graph.static_boxes);ox,oy,oa,hx,hy=boxes.T
   for i,row in enumerate(trace):
    act=row.get('actuation',{});v=act.get('executed',{})
    if math.hypot(v.get('vx',0),v.get('vy',0))>.001 or abs(v.get('wz',0))>.001:
     active.append(act.get('mode','UNKNOWN'))
    if i<len(trace)-1 and act.get('steering_transition_stop',False):
     align_s+=max(0,trace[i+1]['timestamp']-row['timestamp'])
    x,y,yaw=row['pose'];dx,dy=ox-x,oy-y;c,s=math.cos(yaw),math.sin(yaw)
    co,si=np.abs(np.cos(yaw-oa)),np.abs(np.sin(yaw-oa));ca,sa=np.cos(oa),np.sin(oa)
    sep=np.maximum.reduce([np.abs(dx*c+dy*s)-.3755-hx*co-hy*si,
      np.abs(-dx*s+dy*c)-.25-hx*si-hy*co,
      np.abs(dx*ca+dy*sa)-hx-.3755*co-.25*si,
      np.abs(-dx*sa+dy*ca)-hy-.3755*si-.25*co])
    minimum=float(sep.min());static_min=min(static_min,minimum);static_overlap+=minimum<0
   heading=[abs(math.atan2(math.sin(r['pose'][2]),math.cos(r['pose'][2])))*180/math.pi for r in trace if r['task_stage']=='APPROACH' and (scenario!='heading_constrained_passage' or r['pose'][0]<=10.)]
   moving_gap=min(min(g['trolley_signed_SAT_gap_m'],g['person_signed_gap_m']) for g in gap)
   metrics.append(dict(scene=scenario,method=method,seed=a.seed,arrived=bool(result['task_success']),
     path_m=result['path_length_m'],elapsed_sim_s=result['elapsed_sim_s'],
     moving_mode_switches=sum(x!=y for x,y in zip(active,active[1:])),
     steering_alignment_s=align_s,approach_peak_heading_error_deg=max(heading,default=0),
     minimum_actor_proxy_gap_m=moving_gap,static_signed_SAT_min_gap_m=static_min,
     minimum_environment_gap_m=min(moving_gap,static_min),
     static_overlap_pose_count=static_overlap,proxy_clear=moving_gap>0,
     heading_compliant=bool(scenario!='heading_constrained_passage' or max(heading,default=0)<=18.),
     qualified=bool(result['task_success'] and moving_gap>0 and static_min>=0 and
       (scenario!='heading_constrained_passage' or max(heading,default=0)<=18.))))
   data[scenario,method]=(trace,result,gap,graph)
  assert len(set(worlds))==1, 'World geometry differs within '+scenario
 plt.rcParams.update({'font.family':'Liberation Sans','font.size':7,'axes.labelsize':7,
   'xtick.labelsize':7,'ytick.labelsize':7,'axes.titlesize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
 fig=plt.figure(figsize=(7.16,3.9));gs=fig.add_gridspec(2,3,left=.065,right=.985,bottom=.13,top=.87,
   height_ratios=[1,1.45],wspace=.33,hspace=.43)
 axes=[]
 for col,scenario in enumerate(cases):
  top=fig.add_subplot(gs[0,col]);zoom=fig.add_subplot(gs[1,col]);axes.append((top,zoom))
  graph=data[scenario,'multimode'][3]
  # Full-route maps share metric limits; zoom limits are fixed per physical case.
  limits=[(2.,14.,-4.1,1.9)]
  if scenario in ('static_detour','heading_constrained_passage'):center=(9.,-2.1)
  elif scenario=='offset_portal':center=(12.4,-.9)
  else:
   points=[]
   for method in METHODS:
    tr,_,gg,_=data[scenario,method];i=min(range(len(gg)),key=lambda i:gg[i]['trolley_signed_SAT_gap_m']);points.append(tr[min(i,len(tr)-1)]['pose'][:2])
   center=np.mean(points,axis=0)
  limits.append((center[0]-1.8,center[0]+1.8,center[1]-1.45,center[1]+1.45))
  for ax,lim in zip((top,zoom),limits):
   for x,y,yaw,hx,hy in graph.static_boxes:
    # Explicit polygon clipping avoids PDF paths outside the plotting region.
    poly=corners((x,y,yaw),hx,hy)
    for dim,edge,greater in [(0,lim[0],True),(0,lim[1],False),(1,lim[2],True),(1,lim[3],False)]:
     out=[]
     for q,r in zip(poly,poly[1:]+poly[:1]):
      iq=q[dim]>=edge if greater else q[dim]<=edge;ir=r[dim]>=edge if greater else r[dim]<=edge
      if iq:out.append(q)
      if iq!=ir:
       f=(edge-q[dim])/(r[dim]-q[dim]);out.append(tuple(q[j]+f*(r[j]-q[j]) for j in (0,1)))
     poly=out
     if not poly:break
    if poly:ax.add_patch(Polygon(poly,fc='#C8CDD0',ec='#90989D',lw=.25,zorder=0))
   for method in ['no-steering-prediction','nav2-omni','multimode']:
    tr,result,gaps,_=data[scenario,method];poses=np.asarray([r['pose'] for r in tr]);color=COLORS[method]
    visible=(poses[:,0]>=lim[0])&(poses[:,0]<=lim[1])&(poses[:,1]>=lim[2])&(poses[:,1]<=lim[3])
    xy=np.where(visible[:,None],poses[:,:2],np.nan)
    ax.plot(xy[:,0],xy[:,1],c=color,lw=1.5 if method=='multimode' else 1.0,
            ls='--' if method=='no-steering-prediction' else '-',zorder=3)
    if visible[-1]:ax.scatter(*poses[-1,:2],c=color,marker='s' if result['task_success'] else 'x',s=15,zorder=5)
   ax.set(xlim=lim[:2],ylim=lim[2:],xlabel='x (m)')
   ax.set_aspect('equal',adjustable='box')
   if col==0:ax.set_ylabel('y (m)')
   for spine in ax.spines.values():spine.set_color('#A5ACB0');spine.set_linewidth(.5)
  top.set_title(['(a) Static detour','(b) Heading-constrained','(c) Narrow doorway'][col],loc='left',pad=4)
  zoom.set_title(['(d) Detour detail','(e) Encounter detail','(f) Entry detail'][col],loc='left',pad=4)
  # Recorded footprints at the closest approach to the case's interaction region.
  for method in METHODS:
   tr,result,gaps,_=data[scenario,method]
   k=min(range(len(tr)),key=lambda k:math.dist(tr[k]['pose'][:2],center))
   footprint=clipped_polygon(corners(tr[k]['pose']),limits[1])
   if footprint:zoom.add_patch(Polygon(footprint,fc='none',ec=COLORS[method],lw=.7,zorder=4))
  top.add_patch(Rectangle((limits[1][0],limits[1][2]),limits[1][1]-limits[1][0],limits[1][3]-limits[1][2],
                 fc='none',ec='#454C51',ls=':',lw=.5,zorder=4))
 fig.legend(handles=[Line2D([0],[0],c=COLORS[m],ls='--' if m=='no-steering-prediction' else '-',lw=1.5,label=NAMES[m]) for m in METHODS],
            loc='upper center',bbox_to_anchor=(.53,1.),ncol=3,frameon=False,fontsize=7)
 fig.canvas.draw()
 for top, zoom in axes:
  p,q=top.get_position(),zoom.get_position()
  height=p.height*q.width/p.width
  top.set_position([q.x0,p.y0+(p.height-height)/2,q.width,height])
 # Zooms preserve equal metric scales; top maps use identical plot rectangles.
 require_matplotlib_panel_alignment(fig,json_out=a.output/'nav2_matched_trajectories.alignment.json',strict=True,
   row_groups=[{'id':'global-maps','panels':['a','c','e']},{'id':'detail-maps','panels':['b','d','f']}],column_groups=[])
 fig.savefig(a.output/'nav2_matched_trajectories.pdf');fig.savefig(a.output/'nav2_matched_trajectories.svg');fig.savefig(a.output/'nav2_matched_trajectories.png',dpi=600)
 with (a.output/'nav2_matched_metrics.csv').open('w') as stream:
  w=csv.DictWriter(stream,fieldnames=metrics[0].keys());w.writeheader();w.writerows(metrics)
 summary={'batch':str(a.batch),'seed':a.seed,'episodes':len(metrics),'metrics':metrics,'source_sha256':sources,
  'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'mode_switch_definition':'Changes in mode labels between samples with nonzero executed motion; STOP/steering holds excluded',
  'steering_alignment_definition':'Sum of logged simulation intervals marked steering_transition_stop; other stationary time excluded',
  'heading_definition':'Peak wrapped yaw deviation from zero during APPROACH, the registered heading-constrained interval',
  'static_gap_definition':'Minimum signed rectangle SAT separating-axis gap over logged poses and registered static collision boxes',
  'continuous_mesh_collision_claimed':False,'worlds_identical_within_case':True,'protocol':protocol,
  'public_adapter_protocol':public_protocol,'public_adapter_batch':str(a.nav2_batch) if a.nav2_batch else None}
 (a.output/'nav2_matched_metrics.json').write_text(json.dumps(summary,indent=2))
 labels={'nav2-omni':'Omni','multimode':'Full','no-steering-prediction':'$-$Steer'}
 tex=['\\begin{table*}[t]\\centering','\\caption{Matched Gazebo trials, seed '+str(a.seed)+'. L: path length; t: episode duration; N: nonzero executed-mode changes; H: steering-alignment time. Heading error is measured during the heading-constrained approach. All attempts, including timeouts, are retained.}\\label{tab:nav2matched}', '\\footnotesize\\setlength{\\tabcolsep}{3pt}']
 for scenario,title,fields,heads in [
  ('static_detour','Static detour',['path_m','elapsed_sim_s','moving_mode_switches'],['L (m)','t (s)','N']),
  (cases[1],'Heading-constrained passage',['approach_peak_heading_error_deg','minimum_actor_proxy_gap_m'],['yaw (deg)','gap (m)']),
  ('offset_portal','Narrow doorway',['steering_alignment_s','moving_mode_switches'],['H (s)','N'])]:
  tex+=['\\begin{minipage}[t]{.32\\textwidth}\\centering\\textbf{'+title+'}\\par\\smallskip','\\begin{tabular}{l'+'c'*(1+len(fields))+'}\\toprule','Method & Arr. & '+' & '.join(heads)+r'\\'+'\n'+r'\midrule']
  for method in METHODS:
   r=next(r for r in metrics if r['scene']==scenario and r['method']==method)
   values=[str(r[k]) if k=='moving_mode_switches' else f'{r[k]:.3f}' if 'gap' in k else f'{r[k]:.2f}' for k in fields]
   tex.append(labels[method]+' & '+('yes' if r['arrived'] else 'no')+' & '+' & '.join(values)+r'\\')
  tex+=['\\bottomrule\\end{tabular}\\end{minipage}\\hfill']
 tex+=['\\end{table*}'];(a.output/'nav2_matched_table.tex').write_text('\n'.join(tex)+'\n')
 print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
