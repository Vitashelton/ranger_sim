"""Actual RGB input paired with a vector replot of the same saved candidates."""
from pathlib import Path
import hashlib
import gzip
import json
import math
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Polygon
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic'),str(ROOT/'tools')]
from rangermini_dynamic_semantic.candidate_view import body_coordinates
from plot_nav2_matched_cases import corners
from robotics_plot_primitives import vehicle


def main():
    run=ROOT/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run'
    packet_path=run/'episode/proposal_002_input.json'
    image_path=run/'episode/proposal_002_input.jpg'
    response_path=run/'episode/proposal_002_response.json'
    packet=json.loads(packet_path.read_text());pose=packet['context']['robot_pose']
    candidates=packet['candidates'];proposal=json.loads(response_path.read_text())['proposal']
    out=ROOT/'paper/figures/visual_candidate_input';out.parent.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':['Liberation Serif','Arial'],'font.size':8,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.65,
        'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(7.16,2.8))
    rgb=fig.add_axes([.015,.12,.46,.78])
    rgb.imshow(np.asarray(Image.open(image_path))[38:518,:640]);rgb.set_axis_off()
    rgb.set_title('(a) RGB with projected candidates',loc='left',fontsize=8,pad=5)
    ax=fig.add_axes([.555,.26,.265,.62])
    handles=[];labels=[]
    aliases={'ACKERMANN':'Curved','PARALLEL':'Parallel',
             'SPIN_THEN_DRIVE':'Rotate–drive','STOP':'Hold'}
    shorthand={'CURVE':'A','PARALLEL':'P','SPIN':'R'}
    for i,candidate in enumerate(candidates):
        if not candidate['feasible']:continue
        points=body_coordinates(candidate['trajectory'],pose)
        color=np.asarray(candidate['color_rgb'])/255.
        line,=ax.plot(-points[:,1],points[:,0],color=color,lw=1.25,
                      ls='-' if i<3 else '--',zorder=4)
        end=points[-1];world=corners(end)
        xy=np.asarray([(-p[1],p[0]) for p in world])
        ax.add_patch(Polygon(xy,fill=False,ec=color,lw=.65,zorder=3))
        ax.annotate('',xy=(-end[1]-.23*math.sin(end[2]),end[0]+.23*math.cos(end[2])),
                    xytext=(-end[1],end[0]),arrowprops=dict(arrowstyle='->',lw=.65,color=color,mutation_scale=5))
        description='–'.join(shorthand[x] for x in candidate.get('motion_sequence',[]))
        description=description or aliases.get(candidate['motion_behavior'],candidate['motion_behavior'])
        handles.append(line);labels.append(f"{candidate['candidate_id']}  {description}")
    for obj in packet['context']['objects']:
        point=body_coordinates([[*obj['position'],pose[2]]],pose)[0]
        ax.add_patch(Circle((-point[1],point[0]),obj['radius'],fill=False,ec='#A7AFB4',lw=.7))
        ax.text(-point[1],point[0]+obj['radius']+.07,obj['evidence_id'],
                ha='center',fontsize=7,color='#52626C')
    ax.add_patch(Polygon(np.asarray([(-p[1],p[0]) for p in corners((0,0,0))]),
                         fill=False,ec='#263D4A',lw=1.0))
    ax.annotate('',xy=(0,.28),xytext=(0,-.2),
                arrowprops=dict(arrowstyle='->',color='#263D4A',lw=.9,mutation_scale=7))
    ax.axhline(0,color='#D5DBDE',lw=.5,zorder=0)
    ax.axvline(0,color='#D5DBDE',lw=.5,zorder=0)
    ax.set(xlim=(-.85,2.35),ylim=(-.55,3.45),xlabel='Right (m)',ylabel='Forward (m)')
    ax.set_aspect('equal',adjustable='box')
    fig.text(.51,.925,'(b) Same candidates in body frame',fontsize=8)
    fig.legend(handles,labels,loc='center right',bbox_to_anchor=(.995,.56),
               frameon=False,fontsize=7,handlelength=1.5,labelspacing=.65)
    fig.text(.52,.10,'A: curved   P: parallel   R: spin',fontsize=7)
    fig.text(.52,.035,'VLM condition: trolley · retain heading',fontsize=7.5)
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=600)
    files=[packet_path,image_path,response_path]
    provenance=dict(kind='Left: actual transmitted RGB crop; right: vector replot of its exact saved candidate set',
        run=str(run.relative_to(ROOT)),query=2,actual_rgb_crop_xywh=[0,38,640,480],
        ground_condition=proposal,displayed_candidate_ids=[c['candidate_id'] for c in candidates if c['feasible']],
        body_transform='Horizontal=-y_body (right), vertical=x_body (forward); metric coordinates and true footprint dimensions.',
        original_full_input=str(image_path.relative_to(ROOT)),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(str(out)+'.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


def render_legacy_object_overview():
    run=ROOT/'artifacts/object_reference_20261026/pilot_v1/object_reference_a_red_s41'
    ep=run/'episode';packet_path=ep/'proposal_001_input.json'
    image_path=ep/'proposal_001_raw_rgb.jpg';response_path=ep/'proposal_001_response.json'
    packet=json.loads(packet_path.read_text());proposal=json.loads(response_path.read_text())['proposal']
    source=ep/'hybrid_rollouts.jsonl.gz'
    with gzip.open(source,'rt') as stream:records=[json.loads(line) for line in stream]
    eligible=[(i,r) for i,r in enumerate(records) if 4.8<r['pose'][0]<6.5
              and (r['plan'].get('applied_motion_condition') or {}).get('relevant_object_ids')==proposal['relevant_object_ids']
              and r['plan'].get('selected_schedule')
              and len(r['plan'].get('candidate_summary',[]))==3
              and max(v[0] for v in r['plan']['selected_trajectory'])>=6.5]
    index,record=eligible[0];plan=record['plan'];condition=plan['applied_motion_condition']
    objects=packet['context']['objects'];target=next(x for x in objects if x['evidence_id'] in condition['relevant_object_ids'])
    fig=plt.figure(figsize=(7.16,3.35))
    fig.text(.025,.975,'“Deliver to Office 619. Give the trolley carrying the red box more room.”',
             fontsize=8.5,va='top')
    rgb=fig.add_axes([.025,.32,.29,.46]);rgb.imshow(Image.open(image_path));rgb.set_axis_off()
    rgb.set_title('(a) Task and VLM grounding',fontsize=8,loc='left',pad=5)
    fig.text(.025,.25,'VLM associates the red-box trolley.',fontsize=7)
    limits=(4.45,7.55,-3.85,-.35)
    def spatial(ax):
        for obj in objects:
            x,y=obj['position'];ax.add_patch(Polygon(corners((x,y,0),.35,.21),
                fc='#E1E4E6',ec='#78838B',lw=.65,zorder=3))
            color='#B15B52' if obj['evidence_id']=='map_object_1' else '#4E729D'
            ax.plot([x-.22,x+.22],[y,y],c=color,lw=2.4,zorder=4)
        ax.set(xlim=limits[:2],ylim=limits[2:],xlabel='x (m)',ylabel='y (m)')
        ax.set_aspect('equal',adjustable='box');ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax=fig.add_axes([.395,.32,.235,.46]);spatial(ax)
    x=np.linspace(limits[0],limits[1],120);y=np.linspace(limits[2],limits[3],120)
    xx,yy=np.meshgrid(x,y);tx,ty=target['position']
    gap=np.maximum(np.abs(xx-tx)-(.3755+.35),np.abs(yy-ty)-(.25+.21))
    field=np.clip(condition['preferred_clearance_m']-gap,0,None)
    ax.contourf(xx,yy,field,levels=[1e-8,.3,.7,1.5],colors=['#ECF4F3','#DCECE9','#C8E1DD'],zorder=0)
    ax.contour(xx,yy,field,levels=[1e-7],colors=['#6C9D98'],linewidths=.6,linestyles='--',zorder=1)
    vehicle(ax,dict(pose=record['pose'],actuation={'wheel_steering_rad':record['wheel_steering_rad']}),'#334C5B')
    goal=record['main_goal'];edge_x=limits[1]-.05
    edge_y=record['pose'][1]+(goal[1]-record['pose'][1])*(edge_x-record['pose'][0])/(goal[0]-record['pose'][0])
    ax.annotate('To 619',xy=(edge_x,edge_y),xytext=(6.55,-2.7),fontsize=7,
                arrowprops=dict(arrowstyle='->',lw=.7,color='#354955'),va='center')
    ax.set_title('(b) Grounded navigation condition',fontsize=8,loc='left',pad=5)
    fig.text(.37,.19,'Goal: 619 · Object: red-box trolley',fontsize=7)
    fig.text(.37,.145,'Heading: route · Space: extra clearance',fontsize=7)
    ax=fig.add_axes([.745,.32,.235,.46]);spatial(ax)
    selected=np.asarray(plan['selected_trajectory'])
    for c,color,ls,label in [(x,'#8A929A' if x['behavior']=='ACKERMANN' else '#D78A48' if x['behavior']=='PARALLEL' else '#81639D',':' if x['behavior']=='ACKERMANN' else '--','Curved' if x['behavior']=='ACKERMANN' else 'Parallel' if x['behavior']=='PARALLEL' else 'Rotate–drive')
                            for x in plan['candidate_summary']]:
        xy=np.asarray(c['trajectory']);ax.plot(xy[:,0],xy[:,1],c=color,lw=.8,ls=ls,alpha=.85,label=label)
    step=plan['selected_schedule']['switch_steps'][0]
    ax.plot(selected[:step+1,0],selected[:step+1,1],c='#16858D',lw=1.65,label='Selected parallel → curved',zorder=5)
    ax.plot(selected[step:,0],selected[step:,1],c='#426786',lw=1.65,zorder=5)
    vehicle(ax,dict(pose=selected[-1],actuation={'wheel_steering_rad':plan['terminal_steering_angles']}),'#426786',width=.65)
    ax.plot(*selected[step-1,:2],marker='s',ms=3,c='#16858D',zorder=7)
    vehicle(ax,dict(pose=record['pose'],actuation={'wheel_steering_rad':record['wheel_steering_rad']}),'#334C5B')
    ax.set_title('(c) Steering-state-aware MPPI',fontsize=8,loc='left',pad=5)
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower right',bbox_to_anchor=(.995,.095),frameon=False,fontsize=6.5,labelspacing=.2,ncol=2,columnspacing=.8,handlelength=1.3)
    for start,end,label in ((.326,.381,'Grounding'),(.65,.73,'Reference + cost')):
        fig.add_artist(matplotlib.patches.FancyArrowPatch((start,.83),(end,.83),
            transform=fig.transFigure,arrowstyle='->',mutation_scale=8,lw=.7,color='#43575F'))
        fig.text((start+end)/2,.855,label,ha='center',fontsize=6.5,color='#43575F')
    fig.text(.025,.085,'Goal and object association → spatial condition → steering-feasible motion sequence',fontsize=7.5)
    fig.text(.025,.035,'Observed pose, wheel state and task events update the next planning cycle.',fontsize=7,color='#44565F')
    out=ROOT/'paper/figures/task_grounded_overview'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=600)
    inputs=[packet_path,image_path,response_path,source,run/'world.sdf']
    provenance=dict(kind='Method overview grounded in one recorded object-reference episode',
        query=1,rollout_record_index=index,
        selection_rule='First conditioned hybrid return before the object station whose horizon reaches that station and retains all three feasible primitive representatives.',
        instruction_translation='Faithful shortened English translation of the recorded conversational instruction.',
        geometric_panel='Heading-zero registered-rectangle SAT clearance slice derived from the logged preferred clearance. Shading is a derived preference region, not a measured occupancy probability.',
        temporal_relation='The query observation precedes the conditioned replanning example in the same episode. Registered objects are stationary.',
        selected_schedule=plan['selected_schedule'],selected_cost=plan['predicted_score'],
        source_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    Path(str(out)+'.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':
    main()
    from render_task_planning_overview import main as render_overview
    render_overview()
