"""A recorded multimode planning decision; lines are predictions."""
from pathlib import Path
import gzip,json,hashlib,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];ep=R/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run/episode';src=ep/'hybrid_rollouts.jsonl.gz'
with gzip.open(src,'rt') as f:
 for i,line in enumerate(f):
  if i==72:r=json.loads(line);break
p=r['plan'];scene=np.asarray(Image.open(ep/'overview_frames/00485.jpg'));origin=np.asarray(r['pose'][:2]);goal=np.asarray(r['main_goal'])-origin
plt.rcParams.update({'font.family':'Liberation Serif','font.size':7,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(3.45,3.0));a=fig.add_axes([.04,.55,.35,.38]);b=fig.add_axes([.55,.51,.40,.43]);a.imshow(scene[265:455,214:434]);a.axis('off');a.set_title('(a) Gazebo',loc='left',fontsize=7)
for v in p['hybrid_schedule_candidates']:
 if v.get('feasible'):
  xy=np.asarray(v['trajectory'])[:,:2]-origin;b.plot(xy[:,0],xy[:,1],color='#aeb5b7',lw=.45,alpha=.5)
for typ,style in [('ACKERMANN',':'),('PARALLEL','--')]:
 c=next(x for x in p['candidate_summary'] if x['behavior']==typ);xy=np.asarray(c['trajectory'])[:,:2]-origin;b.plot(xy[:,0],xy[:,1],color='#b96932',ls=style,lw=.9)
xy=np.asarray(p['selected_trajectory'])[:,:2]-origin;b.plot(xy[:,0],xy[:,1],color='#0b7c80',lw=1.4);b.scatter([0],[0],s=10,facecolor='white',edgecolor='#222',zorder=5);b.plot(*goal,'*',ms=6,color='#222')
b.set(xlim=(-.04,.39),ylim=(-.025,.55),xlabel=r'$x-x_t$ (m)',ylabel=r'$y-y_t$ (m)');b.set_aspect('equal');b.tick_params(labelsize=6);b.set_title('(b) Predicted candidates',loc='left',fontsize=7)
c=fig.add_axes([.16,.15,.79,.22])
chosen=min((v for v in p['hybrid_schedule_candidates'] if v.get('feasible') and v['structure']==p['selected_schedule']['structure'] and v['switch_steps']==p['selected_schedule']['switch_steps']),key=lambda v:v['cost'])
wheel=np.asarray(chosen['steering_trace']);t=np.arange(len(wheel)+1)*.12
for j,(color,ls,label) in enumerate([('#087d81','-','FL'),('#087d81','--','FR'),('#6c7478','-','RL'),('#6c7478','--','RR')]):
 c.plot(t,np.rad2deg(np.r_[r['wheel_steering_rad'][j],wheel[:,j]]),color=color,ls=ls,lw=.75,label=label)
c.set(xlim=(0,4.8),ylim=(-100,105),xlabel='Prediction time (s)',ylabel='Steering (deg)',yticks=[-90,0,90],xticks=[0,2,4]);c.tick_params(labelsize=6);c.set_title('(c) Predicted wheel state',loc='left',fontsize=7,pad=2)
c.legend(loc='lower center',bbox_to_anchor=(.5,-.72),ncol=4,frameon=False,fontsize=6,handlelength=1.4,columnspacing=.8)
fig.text(.04,.435,'Parallel → rotate: 2.13  |  best primitive reference: 5.10',fontsize=6.6,color='#087d81')
out=R/'paper/figures/mppi_decision_case'
for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=400)
(out.with_suffix('.provenance.json')).write_text(json.dumps({'record_index':72,'prediction_not_execution':True,'source':str(src.relative_to(R)),'frame':'overview_frames/00485.jpg','record_time':r['request_sim_time'],'condition':p['applied_motion_condition'],'selected_schedule':p['selected_schedule'],'cost':p['predicted_score'],'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
