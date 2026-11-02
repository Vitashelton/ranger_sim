from pathlib import Path
import json,sys,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic'),'/home/USER/.codex/skills/nature-figure/scripts']
from robotics_plot_primitives import draw_map,vehicle
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
run=ROOT/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run';ep=run/'episode'
rows=[json.loads(x) for x in (ep/'trace.jsonl').read_text().splitlines() if x.strip()]
start=11.04; end=35.0
rows=[r for r in rows if start<=r['timestamp']<=end]
img=np.asarray(Image.open(ep/'proposal_002_raw_rgb.jpg').convert('RGB'))
fig,(a,b)=plt.subplots(1,2,figsize=(7.15,2.8),gridspec_kw={'width_ratios':[1,1.15]})
plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'pdf.fonttype':42})
fig.subplots_adjust(left=.03,right=.98,top=.86,bottom=.16,wspace=.24)
a.imshow(img);a.axis('off');a.set_title('(a) VLM observation',loc='left',fontsize=9)
lims=(5.0,8.8,-3.8,-.4); graph=SpatialGraph(run/'world.sdf');draw_map(b,graph.static_boxes,lims)
xy=np.array([r['pose'][:2] for r in rows]);b.plot(xy[:,0],xy[:,1],color='#987525',lw=1.8,label='Robot execution',zorder=3)
track=[]
for r in rows:
 t=next((v for v in r.get('tracks',[]) if v['evidence_id']=='track_2'),None)
 if t and 5.5<t['position'][0]<8.3:track.append((r,t))
if track:
 pos=np.array([t['position'] for _,t in track]);b.plot(pos[:,0],pos[:,1],color='#333333',ls='--',lw=1,label='Tracked trolley',zorder=2)
 for k in (0,len(track)-1):
  r,t=track[k];b.add_patch(Circle(t['position'],t['radius'],fill=False,ec='#666666',lw=.6,zorder=2))
moving=[r for r in rows if r['pose'][0]>5.9]
for k in (0,len(moving)//2,len(moving)-1):vehicle(b,moving[k],'#151515',width=.8)
b.set(xlabel='x (m)',ylabel='y (m)');b.set_title('(b) Closed-loop motion',loc='left',fontsize=9);b.legend(loc='lower right',frameon=True,fontsize=7)
out=ROOT/'results/local_motion_preview/vlm_mppi_encounter'
for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=400)
sources=[ep/'proposal_002_raw_rgb.jpg',ep/'proposal_002_response.json',ep/'selection_events.jsonl',ep/'trace.jsonl',run/'world.sdf']
Path(str(out)+'.provenance.json').write_text(json.dumps({'run':str(run.relative_to(ROOT)),'proposal':2,'condition_bound_sim_s':start,'shown_interval_sim_s':[start,end],'input':'Actual raw VLM input; associated response names track_2 as trolley and requests heading preservation and 0.85 m preferred clearance.','track':'Lidar tracked object proxy; circles show tracker radius, not physical trolley mesh.','execution':'Measured kinematic Gazebo closed-loop trace, not planner predictions or comparison.','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
