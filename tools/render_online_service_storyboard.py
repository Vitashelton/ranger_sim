"""One-run robot experiment storyboard and synchronized Gazebo demo video."""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import sys
import math
import xml.etree.ElementTree as ET

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/home/USER/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from robotics_plot_primitives import draw_map,vehicle,mode_strip,MODE_COLORS
from audit_panel_alignment import require_matplotlib_panel_alignment


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main(a):
    run = a.run.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((run/'integrated_audit.json').read_text())
    if not audit.get('unrestricted_integrated_qualified',False):
        raise ValueError('Integrated episode has not passed the independent audit')
    ep = run/'episode'
    events = rows(ep/'selection_events.jsonl')
    trace = rows(ep/'trace.jsonl')
    views = rows(ep/'frames.jsonl')
    overview = rows(ep/'overview_frames.jsonl')
    clearance = json.loads((run/'clearance_trace.json').read_text())
    confirms = [e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED']
    bound = next(e for e in events if e['event']=='TASK_CONDITION_BOUND'
                 and 'trolley' in e['proposal']['object_labels'].values())
    response_paths = sorted(ep.glob('proposal_*_response.json'))
    response = next(p for p in response_paths
                    if json.loads(p.read_text())['proposal']==bound['proposal'])
    query = int(response.name.split('_')[1])
    request = next(e for e in events if e['event']=='SEMANTIC_QUERY_STARTED' and e['query']==query)
    transition = next(e for e in events if e['event']=='SERVICE_ROUTE_ADVANCED'
                      and e.get('old_stage')=='EXIT_ALIGN' and e.get('stop_index')==1)
    encounter = min(clearance, key=lambda r:r['trolley_signed_SAT_gap_m'])['timestamp']

    def frame_at(table, stamp, folder):
        r = min(table, key=lambda r:abs(r['sim_time']-stamp))
        return ep/folder/f"{r['frame']:05d}.jpg", r['sim_time']

    frames = [(ep/f'proposal_{query:03d}_raw_rgb.jpg', request['sim_time'], '(a) Observe trolley', 'RGB')]
    for stamp, title, view in [(encounter, '(b) Maintain clearance', 'overview'),
            (confirms[0]['sim_time'], '(c) Collect at 617', 'overview'),
            (confirms[2]['sim_time'], '(d) Exchange at 614', 'overview'),
            (transition['sim_time']-.6, '(e) Reorient for departure', 'overview'),
            (confirms[3]['sim_time'], '(f) Deliver to 619', 'overview')]:
        path, actual = frame_at(overview if view=='overview' else views, stamp,
                                'overview_frames' if view=='overview' else 'frames')
        frames.append((path, actual, title, view))
    assert all(x[0].is_relative_to(run) for x in frames)
    assert all(frames[i][1] <= frames[i+1][1] for i in range(len(frames)-1))
    camera_model=ET.parse(run/'world.sdf').getroot().find("world/model[@name='navigation_overhead_camera']")
    camera_pose=list(map(float,camera_model.findtext('pose').split()))
    fov=float(camera_model.findtext('link/sensor/camera/horizontal_fov'))
    roll,pitch,yaw=camera_pose[3:]
    cr,sr,cp,sp,cy,sy=math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
    rotation=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                       [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
    trace_times=[r['timestamp'] for r in trace]
    def pixel(pose,shape):
        q=rotation.T@(np.array([pose[0],pose[1],.20])-np.array(camera_pose[:3]))
        focal=shape[1]/(2*math.tan(fov/2))
        return shape[1]/2-focal*q[1]/q[0],shape[0]/2-focal*q[2]/q[0]
    crops={}
    def crop_recorded(image,stamp,width,height,draw_tail=True):
        end=max(0,bisect_right(trace_times,stamp)-1)
        center=pixel(trace[end]['pose'],image.shape)
        if draw_tail:
            begin=max(0,bisect_right(trace_times,stamp-3.)-1)
            points=np.array([pixel(r['pose'],image.shape) for r in trace[begin:end+1]],dtype=np.int32)
            if len(points)>1:cv2.polylines(image,[points],False,(20,133,143),2,cv2.LINE_AA)
        width=min(width,image.shape[1]);height=min(height,image.shape[0])
        x=int(np.clip(round(center[0]-width/2),0,image.shape[1]-width))
        y=int(np.clip(round(center[1]-height/2),0,image.shape[0]-height))
        return image[y:y+height,x:x+width],(x,y,width,height)
    plt.rcParams.update({'font.family':['Liberation Serif','Arial'], 'font.size':8,
                         'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(2,3,figsize=(7.16,3.18))
    fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.91,wspace=.035,hspace=.32)
    for ax, (path, stamp, title, view) in zip(axes.flat,frames):
        image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        if view=='overview':
            image,box=crop_recorded(image.copy(),stamp,380,240)
            crops[str(path.relative_to(ROOT))]=list(box)
        box=ax.get_position()
        panel_ratio=fig.get_figwidth()*box.width/(fig.get_figheight()*box.height)
        image_ratio=image.shape[1]/image.shape[0]
        height=min(1/panel_ratio,1/image_ratio);width=height*image_ratio
        x=(1-width)/2;y=(1/panel_ratio-height)/2
        ax.imshow(image,extent=(x,x+width,y+height,y),aspect='auto')
        ax.set_xlim(0,1);ax.set_ylim(1/panel_ratio,0)
        ax.set_axis_off()
        ax.set_title(title,loc='left',fontsize=8,pad=4)
    require_matplotlib_panel_alignment(fig,json_out=out/'online_service_storyboard.alignment.json',strict=True)
    fig.savefig(out/'online_service_storyboard.pdf')
    fig.savefig(out/'online_service_storyboard.svg')
    fig.savefig(out/'online_service_storyboard.png',dpi=600)
    evidence = [run/'integrated_audit.json',run/'task_parser_response.json',
                ep/'trace.jsonl',ep/'selection_events.jsonl',ep/'frames.jsonl',
                ep/'overview_frames.jsonl',response] + [x[0] for x in frames]
    provenance = dict(kind='Six chronological scenes from one audited Gazebo episode',
        run=str(run.relative_to(ROOT)),fresh_language_parse=True,online_visual_requests=audit['real_visual_responses'],
        frame_selection=[dict(path=str(p.relative_to(ROOT)),sim_time=t,label=label,view=view)
                         for p,t,label,view in frames],
        scene_processing='Recorded RGB and spatially cropped overhead frames. Overlays show the preceding three seconds of logged execution, projected with the recorded camera calibration. No generated frames or invented trajectories.',
        crop_xywh_pixels=crops,
        camera_pose=camera_pose,camera_horizontal_fov=fov,
        confirmation='Four simulated service acknowledgements; no physical material handling.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'online_service_storyboard.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    # Publication overview connects the complete executed route to its actual
    # observation and one maneuver. The original six-frame storyboard remains.
    fig=plt.figure(figsize=(7.16,2.9))
    ax=fig.add_axes([.065,.20,.55,.68])
    ax.set_anchor('W')
    limits=(2.25,13.6,-5.95,1.9)
    draw_map(ax,SpatialGraph(run/'world.sdf').static_boxes,limits)
    xy=np.asarray([r['pose'][:2] for r in trace])
    segments=np.stack([xy[:-1],xy[1:]],axis=1)
    colors=[MODE_COLORS.get(r.get('actuation',{}).get('mode'),'#D8DCDF') for r in trace[:-1]]
    ax.add_collection(LineCollection(segments,colors=colors,linewidths=1.5,zorder=3))
    cart=np.asarray([r['trolley_pose'][:2] for r in clearance])
    cart[(cart[:,0]<limits[0])|(cart[:,0]>limits[1])|(cart[:,1]<limits[2])|(cart[:,1]>limits[3])]=np.nan
    ax.plot(cart[:,0],cart[:,1],ls=':',c='#7C858C',lw=.7,zorder=1)
    ax.plot(*xy[0],marker='o',ms=4,mfc='white',mec='#35434A',mew=.7,zorder=8)
    ax.annotate('Start',xy=xy[0],xytext=(-3,7),textcoords='offset points',fontsize=7)
    for event,label,offset in [(confirms[0],'1 · 617',(-18,10)),
                               (confirms[2],'2–3 · 614',(16,10)),
                               (confirms[3],'4 · 619',(-24,10))]:
        row=min(trace,key=lambda r:abs(r['timestamp']-event['sim_time']))
        ax.plot(*row['pose'][:2],marker='o',ms=3,mfc='white',mec='#35434A',mew=.7,zorder=8)
        if label.startswith('2'):
            ax.annotate('614\n2–3',xy=row['pose'][:2],xytext=(4.1,-4.5),textcoords='data',fontsize=7,va='top')
        else:
            ax.annotate(label,xy=row['pose'][:2],xytext=offset,textcoords='offset points',fontsize=7)
    encounter_row=min(trace,key=lambda r:abs(r['timestamp']-encounter))
    departure_stamp=transition['sim_time']-.6
    departure_row=min(trace,key=lambda r:abs(r['timestamp']-departure_stamp))
    vehicle(ax,encounter_row,'#35434A')
    vehicle(ax,departure_row,MODE_COLORS.get(departure_row['actuation']['mode'],'#35434A'))
    ax.set(xlabel='x (m)',ylabel='y (m)')
    ax.tick_params(labelsize=7)
    ax.spines['right'].set_visible(False);ax.spines['top'].set_visible(False)
    ax.set_title('(a) Complete online mission',loc='left',fontsize=8,pad=4)
    views_used=[(ep/f'proposal_{query:03d}_input.jpg',frames[0][1],'(b) Actual VLM input','vlm_rgb_crop')]
    for k,(path,stamp,title,view) in enumerate(views_used):
        panel=fig.add_axes([.63,.34,.35,.53])
        pixels=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB)
        if view=='overview':pixels,box=crop_recorded(pixels.copy(),stamp,380,240)
        elif view=='vlm_rgb_crop':
            assert pixels.shape[:2]==(518,1120)
            pixels=pixels[38:518,:640]
        panel.imshow(pixels);panel.set_axis_off()
        panel.set_title(title,loc='left',fontsize=8,pad=4)
        if k==0:
            panel.text(0,-.08,'Trolley · retain heading',transform=panel.transAxes,fontsize=7)
    strip=fig.add_axes([.65,.15,.32,.065])
    mode_strip(strip,trace,0,'Execution')
    strip.set(xlim=(0,trace[-1]['elapsed']),ylim=(-.4,.4),yticks=[],xlabel='Task time (s)')
    strip.tick_params(axis='x',labelsize=7,pad=2)
    for side in ('left','right','top'):strip.spines[side].set_visible(False)
    strip.xaxis.label.set_size(7)
    fig.legend([Patch(fc=MODE_COLORS[k]) for k in ('DUAL_ACKERMANN','PARALLEL','SPINNING','STOPPED')],
               ['Curved','Parallel','Spin','Stopped'],loc='lower left',bbox_to_anchor=(.08,.0),
               ncol=4,frameon=False,fontsize=7,handlelength=.8,columnspacing=1.)
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'online_service_overview.{ext}',dpi=600)
    overview_provenance=dict(provenance,
        kind='One-run measured mode-colored route, real model observation and recorded departure image',
        displayed_frames=[dict(path=str(p.relative_to(ROOT)),sim_time=t,role=title)
                          for p,t,title,_ in views_used],
        footprint_states=[dict(sim_time=r['timestamp'],pose=r['pose'],actuation=r['actuation'])
                          for r in (encounter_row,departure_row)],
        mode_encoding=MODE_COLORS,
        actual_vlm_input_crop_xywh=[0,38,640,480],
        map_source_sha256=hashlib.sha256((run/'world.sdf').read_bytes()).hexdigest(),
        glyph_source_sha256=hashlib.sha256((ROOT/'tools/robotics_plot_primitives.py').read_bytes()).hexdigest())
    for image_path,_,_,_ in views_used:
        overview_provenance['source_sha256'][str(image_path.relative_to(ROOT))]=hashlib.sha256(image_path.read_bytes()).hexdigest()
    (out/'online_service_overview.provenance.json').write_text(json.dumps(overview_provenance,indent=2)+'\n')
    plt.close(fig)
    fig,route_ax=plt.subplots(figsize=(3.45,2.35))
    fig.subplots_adjust(left=.14,right=.98,bottom=.23,top=.88)
    draw_map(route_ax,SpatialGraph(run/'world.sdf').static_boxes,limits)
    route_ax.add_collection(LineCollection(segments,colors=colors,linewidths=1.15,zorder=3))
    route_ax.plot(*xy[0],marker='o',ms=3,mfc='white',mec='#35434A',mew=.7,zorder=8)
    for event,label,offset in [(confirms[0],'617',(-9,6)),(confirms[2],'614',(8,-2)),(confirms[3],'619',(-12,6))]:
        row=min(trace,key=lambda r:abs(r['timestamp']-event['sim_time']))
        route_ax.plot(*row['pose'][:2],marker='o',ms=2.5,mfc='white',mec='#35434A',mew=.6,zorder=8)
        if label=='614':route_ax.text(4.1,-4.8,label,fontsize=7)
        else:route_ax.annotate(label,xy=row['pose'][:2],xytext=offset,textcoords='offset points',fontsize=7)
    route_ax.set(xlabel='x (m)',ylabel='y (m)')
    route_ax.tick_params(labelsize=7)
    route_ax.set_title('617 collection → 614 exchange → 619 delivery',fontsize=7.5,pad=4)
    route_ax.spines['top'].set_visible(False);route_ax.spines['right'].set_visible(False)
    fig.legend([Patch(fc=MODE_COLORS[k]) for k in ('DUAL_ACKERMANN','PARALLEL','SPINNING')],
               ['Curved','Parallel','Spin'],loc='lower center',ncol=3,fontsize=7,
               frameon=False,bbox_to_anchor=(.5,0),handlelength=.8,columnspacing=1.)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'online_service_route.{ext}',dpi=600)
    (out/'online_service_route.provenance.json').write_text(json.dumps(overview_provenance,indent=2)+'\n')
    plt.close(fig)
    if not a.video:
        return
    overview_times=[r['sim_time'] for r in overview]
    view_times=[r['sim_time'] for r in views]
    condition_events=[e for e in events if e['event']=='TASK_CONDITION_BOUND']
    condition_times=[e['sim_time'] for e in condition_events]
    origin=trace[0]['timestamp']-trace[0]['elapsed']
    writer=cv2.VideoWriter(str(out/'online_service_demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,720))
    assert writer.isOpened()
    last=trace[-1]['timestamp']
    def fit_image(image,width,height):
        scale=min(width/image.shape[1],height/image.shape[0])
        w,h=round(image.shape[1]*scale),round(image.shape[0]*scale)
        canvas=np.zeros((height,width,3),dtype=np.uint8)
        x,y=(width-w)//2,(height-h)//2
        canvas[y:y+h,x:x+w]=cv2.resize(image,(w,h))
        return canvas
    for t in np.arange(origin,last,.3):
        i=max(0,bisect_right(overview_times,t)-1);j=max(0,bisect_right(view_times,t)-1)
        row=trace[max(0,bisect_right(trace_times,t)-1)]
        image=cv2.imread(str(ep/'overview_frames'/f"{overview[i]['frame']:05d}.jpg"))
        image,_=crop_recorded(image,t,800,450)
        image=fit_image(image,1280,720)
        camera=cv2.imread(str(ep/'frames'/f"{views[j]['frame']:05d}.jpg"))
        camera=fit_image(camera,426,240)
        image[464:704,838:1264]=camera
        overlay=image.copy();cv2.rectangle(overlay,(12,12),(818,148),(12,20,25),-1)
        cv2.addWeighted(overlay,.72,image,.28,0,image)
        confirmed=sum(e['sim_time']<=t for e in confirms)
        mode=row.get('actuation',{}).get('mode','UNKNOWN')
        alignment=row.get('actuation',{}).get('steering_transition_stop')
        k=bisect_right(condition_times,t)-1
        condition=condition_events[k]['task_condition'] if k>=0 else {}
        heading='keep heading' if condition.get('heading_policy')=='preserve_current_heading' else 'route heading'
        objects=', '.join(sorted(set(condition.get('object_labels',{}).values()))) or 'no active object'
        lines=['Task: 617 collection > 614 exchange > 619 delivery',
               f"Progress: {confirmed}/4 confirmations | {row['anchor_id']} | {row['task_stage']}",
               f"VLM spatial condition: {objects} | {heading}",
               f"MPPI execution: "+{'DUAL_ACKERMANN':'Curved driving','PARALLEL':'Parallel translation',
                    'SPINNING':'In-place rotation','STOPPED':'Stopped'}.get(mode,mode)
                    +(' | steering alignment' if alignment else '')]
        for n,line in enumerate(lines):
            cv2.putText(image,line,(25,38+30*n),cv2.FONT_HERSHEY_SIMPLEX,.61,(245,248,250),1,cv2.LINE_AA)
        writer.write(image)
    writer.release()
    print('Rendered storyboard and synchronized 3x-simulation-speed video:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--video',action='store_true')
    main(p.parse_args())
