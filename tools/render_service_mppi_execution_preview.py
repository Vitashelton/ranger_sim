"""Gazebo scene, feasible multimode rollouts, and four predicted wheel angles."""
from pathlib import Path
import gzip,hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,'/home/USER/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

def main():
    run=ROOT/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run';ep=run/'episode';source=ep/'hybrid_rollouts.jsonl.gz'
    with gzip.open(source,'rt') as f:records=[json.loads(line) for line in f]
    index,r=next((i,r) for i,r in enumerate(records) if (r['plan'].get('applied_motion_condition') or {}).get('anchor_id')=='D619' and r['plan'].get('selected_schedule') and len(r['plan']['candidate_summary'])==3)
    p=r['plan'];schedule=p['selected_schedule'];condition=p['applied_motion_condition']
    chosen=min((c for c in p['hybrid_schedule_candidates'] if c.get('feasible') and c['structure']==schedule['structure'] and c['switch_steps']==schedule['switch_steps']),key=lambda c:c['cost'])
    assert abs(chosen['cost']-p['predicted_score'])<1e-8
    frame_index=ep/'overview_frames.jsonl';frames=[json.loads(l) for l in frame_index.read_text().splitlines()]
    frame=min(frames,key=lambda x:abs(x['sim_time']-r['request_sim_time']))
    assert abs(frame['sim_time']-r['request_sim_time']) < .2
    frame_path=ep/'overview_frames'/f"{frame['frame']:05d}.jpg"
    img=np.asarray(Image.open(frame_path).convert('RGB'))
    # The camera is stationary and the crop is centered on the logged vehicle
    # projection (324,317) in the selected, same-call frame.
    crop=(130,155,420,350);x,y,w,h=crop
    assert img.shape[:2]==(720,1280) and frame['frame']==485
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':7.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.16,3.0));gs=fig.add_gridspec(1,3,left=.025,right=.99,bottom=.29,top=.84,wspace=.36,width_ratios=[1.20,1.,1.18])
    a,b,c=[fig.add_subplot(gs[0,j]) for j in range(3)]
    a.imshow(img[y:y+h,x:x+w]);a.set_axis_off();a.set_title('(a) Gazebo scene',loc='left',fontsize=8,pad=5)
    initial=np.asarray(r['pose'][:2]);valid=[v for v in p['hybrid_schedule_candidates'] if v.get('feasible')]
    for v in valid:
        q=np.asarray(v['trajectory'])[:,:2]-initial
        b.plot(q[:,0],q[:,1],c='#969DA1',alpha=.45,lw=.55,zorder=1)
    for behavior,ls in [('ACKERMANN',':'),('PARALLEL','--')]:
        primitive=next(v for v in p['candidate_summary'] if v['behavior']==behavior)
        q_primitive=np.asarray(primitive['trajectory'])[:,:2]-initial
        b.plot(q_primitive[:,0],q_primitive[:,1],c='#4F585F',ls=ls,lw=.9,zorder=2)
    q=np.asarray(p['selected_trajectory'])[:,:2]-initial
    b.plot(q[:,0],q[:,1],c='#987525',lw=1.5,zorder=3)
    trace=[json.loads(line) for line in (ep/'trace.jsonl').read_text().splitlines() if line.strip()]
    execution=[v for v in trace if r['request_sim_time'] <= v['timestamp'] <= r['request_sim_time']+4.8]
    actual=np.asarray([v['pose'][:2] for v in execution])-initial
    b.plot(actual[:,0],actual[:,1],c='#111111',ls='--',lw=1.1,zorder=5)

    b.plot(0,0,'o',ms=3,mfc='white',mec='#26343A',mew=.7,zorder=4)
    goal=np.asarray(r['main_goal'])-initial;b.plot(*goal,marker='*',ms=6,c='#26343A',zorder=4)
    b.set(xlim=(-.04,.39),ylim=(-.025,.55),xlabel=r'$x-x_t$ (m)',ylabel=r'$y-y_t$ (m)')
    b.set_aspect('equal',adjustable='box');b.set_title('(b) MPPI sequence evaluation',loc='left',fontsize=8,pad=5)
    t=np.arange(1,len(chosen['steering_trace'])+1)*.12;wheel=np.asarray(chosen['steering_trace']);initial_wheel=np.asarray(r['wheel_steering_rad'])
    styles=[('#137E83','-','FL'),('#137E83','--','FR'),('#555E65','-','RL'),('#555E65','--','RR')]
    handles=[]
    for j,(color,ls,label) in enumerate(styles):
        line,=c.plot(np.r_[0,t],np.rad2deg(np.r_[initial_wheel[j],wheel[:,j]]),c=color,ls=ls,lw=.9,label=label);handles.append(line)
    c.axvline(chosen['switch_times_s'][0],color='#777E84',ls=':',lw=.6)
    holds=np.asarray(chosen['alignment_trace'],bool)
    for k in np.flatnonzero(holds):c.plot([k*.12,(k+1)*.12],[-102,-102],c='#242424',lw=1.8,solid_capstyle='butt')
    c.set(xlim=(0,4.8),ylim=(-110,108),xlabel='Prediction time (s)',ylabel=r'Steering angle (deg)',xticks=[0,2,4],yticks=[-90,0,90])
    c.set_title('(c) Predicted wheel angles',loc='left',fontsize=8,pad=5)
    fig.legend([Line2D([0],[0],c='#969DA1',lw=.8),Line2D([0],[0],c='#4F585F',ls=':',lw=.9),Line2D([0],[0],c='#4F585F',ls='--',lw=.9),Line2D([0],[0],c='#987525',lw=1.4),Line2D([0],[0],c='#111111',ls='--',lw=1.0)],['Feasible','Curved','Parallel','Selected prediction','Executed'],loc='lower center',bbox_to_anchor=(.50,.005),ncol=2,fontsize=6.5,frameon=False,handlelength=1.3,columnspacing=.7)
    fig.legend(handles,[h.get_label() for h in handles],loc='lower center',bbox_to_anchor=(.85,.01),ncol=4,fontsize=6.5,frameon=False,handlelength=1.4,columnspacing=.6)
    require_matplotlib_panel_alignment(fig,json_out=ROOT/'results/local_motion_preview/mppi_execution.alignment.json',strict=True,
        panel_ids=['a','b','c'],row_groups=[['a','b','c']],
        exemptions=[{'panels':['a','b','c'],'checks':['panel-width','horizontal-gutter'],'reason':'Recorded image and metric trajectory use different aspect ratios; neither is distorted.'}])
    out=ROOT/'results/local_motion_preview/mppi_execution'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
    sources=[source,frame_index,frame_path,run/'world.sdf',ep/'trace.jsonl']
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'Recorded Gazebo camera frame paired with same-call 4WS model predictions',
      'run':str(run.relative_to(ROOT)),'record_index':index,'selection_rule':'First D619 approach with valid grounded condition, selected hybrid and all primitive representatives.',
      'frame':frame,'frame_time_difference_s':frame['sim_time']-r['request_sim_time'],'crop_xywh':crop,
      'execution':'Actual closed-loop trace from planning time to 4.8 s later, with replanning; not an open-loop execution of the selected prediction.',
      'candidates':'All feasible saved hybrid sequence representatives plus curved and parallel primitive representatives; these are predictions, not independent methods or measured trajectories.',
      'selected_structure':schedule,'selected_cost':chosen['cost'],'selected_alignment_s':chosen['alignment_s'],
      'primitive_costs':{v['behavior']:v['cost'] for v in p['candidate_summary']},
      'primitive_cost_terms':{v['behavior']:v['cost_terms'] for v in p['candidate_summary']},
      'condition':condition,'goal':r['main_goal'],'state':r['pose'],'wheel_state':r['wheel_steering_rad'],
      'steering_order':['front_left','front_right','rear_left','rear_right'],
      'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},
      'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Same episode, record',index,'frame',frame['frame'],'feasible sequences',len(valid))
if __name__=='__main__':main()
