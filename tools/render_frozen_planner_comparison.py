"""Visualize replay-controlled Nav2 and 4WS MPPI runs on their common map."""
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
base=ROOT/'artifacts/task_attention_service_20261030/frozen_planner_comparison_v1';nav=base/'nav2_s67';native=base/'hybrid_s67';world=nav/'world.sdf';ref=base/'static_reference.json';graph=SpatialGraph(world);route=np.asarray(json.load(open(ref))['path_xy'])
read=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def readgz(p):
 with gzip.open(p,'rt') as f:return [json.loads(l) for l in f if l.strip()]
navrows=read(nav/'episode/trace.jsonl');nav_gap=min(json.load(open(nav/'clearance_trace.json')),key=lambda x:x['trolley_signed_SAT_gap_m']);nr=min(navrows,key=lambda r:abs(r['timestamp']-nav_gap['timestamp']))
obsfile=nav/'episode/planner_observations.jsonl.gz';obs=min(readgz(obsfile),key=lambda x:abs(x['sim_time']-nr['timestamp']));assert abs(obs['sim_time']-nr['timestamp'])<.05
pose=np.asarray(obs['pose']);cfg=json.load(open(nav/'episode/manifest.json'))['mppi_config'];res=.05;n=161;x0=pose[0]-n*res/2;y0=pose[1]-n*res/2;xx,yy=np.meshgrid(x0+(np.arange(n)+.5)*res,y0+(np.arange(n)+.5)*res);grid=np.zeros((n,n),dtype=np.uint8)
for x,y,yaw,hx,hy in graph.static_boxes:
 dx,dy=xx-x,yy-y;c,s=math.cos(yaw),math.sin(yaw);grid[(np.abs(c*dx+s*dy)<=hx)&(np.abs(-s*dx+c*dy)<=hy)]=254
scan=obs.get('scan');points=[]
if scan:
 ranges=scan['ranges'];a0=scan['angle_min'];da=scan['angle_increment'];stride=max(1,len(ranges)//100);sx=pose[0]+cfg['lidar_x']*math.cos(pose[2]);sy=pose[1]+cfg['lidar_x']*math.sin(pose[2]);cp=math.cos(cfg['lidar_pitch'])
 for i in range(0,len(ranges),stride):
  v=ranges[i]
  if v is None or not math.isfinite(v) or v<.12 or v>5:continue
  ang=a0+i*da;bx,by=v*cp*math.cos(ang),v*math.sin(ang);points.append((sx+math.cos(pose[2])*bx-math.sin(pose[2])*by,sy+math.sin(pose[2])*bx+math.cos(pose[2])*by))
if points:
 ij=np.floor((np.asarray(points)-[x0,y0])/res).astype(int);ij=ij[(ij[:,0]>=0)&(ij[:,0]<n)&(ij[:,1]>=0)&(ij[:,1]<n)];grid[ij[:,1],ij[:,0]]=254
events=read(nav/'episode/selection_events.jsonl');condition=[e['proposal'] for e in events if e['event']=='TASK_CONDITION_BOUND' and e['sim_time']<=nr['timestamp']][-1];assert condition['relevant_object_ids']==['track_1']
for t in obs['tracks']:
 if t.get('registered_box') is not None:continue
 grid[(xx-t['position'][0])**2+(yy-t['position'][1])**2<=t['radius']**2]=254
for t in obs['tracks']:
 if t['evidence_id'] not in condition['relevant_object_ids']:continue
 gap=np.hypot(xx-t['position'][0],yy-t['position'][1])-t['radius']-.25;preferred=condition['preferred_clearance_m'];soft=np.asarray(200*np.clip((preferred-gap)/max(preferred,.01),0.,1.),dtype=np.uint8);grid=np.maximum(grid,soft)
out=ROOT/'paper/figures/frozen_planner_comparison';np.savez_compressed(str(out)+'.grid.npz',grid=grid,resolution=res,origin=np.array([x0,y0]),sim_time=obs['sim_time'])
with gzip.open(native/'episode/hybrid_rollouts.jsonl.gz','rt') as f:rolls=[json.loads(l) for l in f]
native_gap=min(json.load(open(native/'clearance_trace.json')),key=lambda x:x['trolley_signed_SAT_gap_m']);hybrid=min(rolls,key=lambda r:abs(r['request_sim_time']-native_gap['timestamp']));assert (hybrid['plan'].get('applied_motion_condition') or {}).get('relevant_object_ids')==['track_1'] and len(hybrid['plan'].get('selected_trajectory',[]))>1
rows=read(native/'episode/trace.jsonl');actual=min(rows,key=lambda r:abs(r['timestamp']-hybrid['request_sim_time']));
plt.rcParams.update({'font.family':'Liberation Serif','font.size':7.3,'axes.labelsize':7.3,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(3.42,4.6));fig.subplots_adjust(left=.16,right=.97,bottom=.09,top=.96,hspace=.24);lim=(4.25,8.5,-3.95,-1.15);cases=[]
for ax,run,title,color in zip(axes,(nav,native),('Nav2 Omni MPPI','Task-Attentive MPPI'),('#333333','#98711d')):
 draw_map(ax,graph.static_boxes,lim);data=read(run/'episode/trace.jsonl');xy=np.asarray([r['pose'][:2] for r in data]);ax.plot(route[:,0],route[:,1],color='#aaaaaa',ls=':',lw=.85,zorder=3)
 if run==nav:
  ax.imshow(grid,extent=(x0,x0+n*res,y0,y0+n*res),origin='lower',cmap='Greys',vmin=0,vmax=254,alpha=.65,interpolation='nearest',zorder=1)
  row=nr;stamp=obs['sim_time']
 else:
  local=np.asarray(hybrid['plan']['selected_trajectory']);ax.plot(local[:,0],local[:,1],color=color,ls='--',lw=1.0,zorder=5);row={'pose':hybrid['pose'],'actuation':{'wheel_steering_rad':hybrid['wheel_steering_rad']}};stamp=hybrid['request_sim_time']
 ax.plot(xy[:,0],xy[:,1],color=color,lw=1.25,zorder=4)
 gaps=json.load(open(run/'clearance_trace.json'));g=min(gaps,key=lambda x:abs(x['timestamp']-stamp));ax.add_patch(Polygon(corners(g['trolley_pose'],.35,.21),closed=True,fc='#dddddd',ec='#555555',lw=.6,zorder=5));vehicle(ax,row,color,width=.8)
 ax.set_title(title,loc='left',fontsize=7.5,pad=2);ax.set(xlim=lim[:2],ylim=lim[2:],ylabel='y (m)');ax.set_aspect('equal',adjustable='box');ax.set_xticks([5,6,7,8]);ax.set_yticks([-3,-2]);ax.tick_params(length=2,pad=1);cases.append({'run':str(run.relative_to(ROOT)),'snapshot_sim_s':stamp,'pose':row['pose'],'trolley_pose':g['trolley_pose']})
axes[0].tick_params(labelbottom=False);axes[1].set_xlabel('x (m)')
for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
sources=[world,ref,nav/'episode/trace.jsonl',nav/'episode/selection_events.jsonl',obsfile,nav/'episode/manifest.json',native/'episode/trace.jsonl',native/'episode/hybrid_rollouts.jsonl.gz',nav/'clearance_trace.json',native/'clearance_trace.json',nav/'source_snapshot/nav2_omni_mppi.py']
out.with_suffix('.provenance.json').write_text(json.dumps({'kind':'Matched planner execution with replayed real VLM condition','nav2_grid':'Offline reconstruction from logged same-time state, scan and tracks using frozen Nav2 adapter logic, before its C++ inflation layer. Not a recorded RViz image.','grid_dimensions':[n,n],'resolution':res,'origin':[x0,y0],'cases':cases,'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
