"""Single-episode Gazebo observations linked to the measured service route."""
from pathlib import Path
import hashlib,json,math,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph

def box_corners(x,y,yaw,hx,hy):
    c,s=math.cos(yaw),math.sin(yaw)
    return [(x+c*dx-s*dy,y+s*dx+c*dy)
            for dx,dy in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]]

def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def main():
    run=ROOT/'artifacts/conversational_service_20261027/static_online_v1/run';ep=run/'episode'
    trace=read(ep/'trace.jsonl');events=read(ep/'selection_events.jsonl');over=read(ep/'overview_frames.jsonl')
    event=next(e for e in events if e['event']=='SEMANTIC_QUERY_STARTED' and e.get('query')==11)
    ack=[e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED']
    bg=over[-1]
    bgfile=ep/'overview_frames'/f"{bg['frame']:05d}.jpg"
    rgbfile=ep/'proposal_011_raw_rgb.jpg';response=ep/'proposal_011_response.json';packet=ep/'proposal_011_input.json'
    assert json.loads(response.read_text())['proposal']['object_labels'].get('track_2')=='person'
    raw=np.asarray(Image.open(bgfile).convert('RGB'));assert raw.shape[:2]==(720,1280)
    camera=ET.parse(run/'world.sdf').getroot().find("world/model[@name='navigation_overhead_camera']")
    cam=list(map(float,camera.findtext('pose').split()));fov=float(camera.findtext('link/sensor/camera/horizontal_fov'))
    roll,pitch,yaw=cam[3:];cr,sr,cp,sp,cy,sy=math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
    rot=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
    focal=raw.shape[1]/(2*math.tan(fov/2))
    def pixel(pose):
        q=rot.T@(np.array([pose[0],pose[1],.20])-np.asarray(cam[:3]))
        return np.array([raw.shape[1]/2-focal*q[1]/q[0],raw.shape[0]/2-focal*q[2]/q[0]])
    def tr(t):return min(trace,key=lambda q:abs(q['timestamp']-t))
    rgbtime=event['sim_time'];qr=tr(rgbtime)
    assert abs(qr['timestamp']-rgbtime)<.25
    # The background is a single recorded overhead frame; path lines are past
    # robot-center history through that frame, not a claim of simultaneous poses.
    active=[r for r in trace if r['timestamp']<=bg['sim_time']]
    uv=np.asarray([pixel(r['pose']) for r in active])
    rgbimg=np.asarray(Image.open(rgbfile).convert('RGB'))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.16,3.75));a=fig.add_axes([.025,.11,.61,.80]);b=fig.add_axes([.68,.60,.29,.30]);c=fig.add_axes([.68,.13,.29,.41])
    a.imshow(raw);a.set_axis_off();a.set_title('Service route and MPPI execution',loc='left',fontsize=8,pad=3)
    a.plot(uv[:,0],uv[:,1],color='#9A9A9A',lw=.75,zorder=3)
    for e,label,delta in [(ack[0],'617 · Collect',(6,-11)),(ack[2],'614 · Exchange',(-72,12))]:
        q=pixel(tr(e['sim_time'])['pose']);a.plot(*q,'o',ms=3,mfc='white',mec='#26343A',mew=.6,zorder=6)
        a.annotate(label,xy=q,xytext=delta,textcoords='offset points',fontsize=7,color='#222222',
                   bbox={'fc':'white','ec':'none','alpha':.8,'pad':1.0},zorder=7)
    q619=pixel(tr(bg['sim_time'])['pose']);a.plot(*q619,'*',ms=6,c='#222222',zorder=6)
    a.annotate('619 · Deliver',xy=q619,xytext=(6,7),textcoords='offset points',fontsize=7,
               bbox={'fc':'white','ec':'none','alpha':.8,'pad':1.0},zorder=7)
    # Highlight the 614 doorway whose recorded exit maneuver is enlarged in (c).
    qroi=np.asarray([pixel([x,y,.20]) for x,y in
                     [(1.95,-6.45),(4.75,-6.45),(4.75,-3.65),(1.95,-3.65)]])
    a.add_patch(Polygon(qroi,closed=True,fill=False,edgecolor='#9A7410',
                        linewidth=.8,linestyle=(0,(3,2)),zorder=8))
    # Map the active VLM person binding and its requested clearance onto the
    # same Gazebo view as the logged service route.
    grounded=next(r for r in trace if (r.get('plan',{}).get('applied_motion_condition') or {}).get('object_labels',{}).get('track_2')=='person')
    cond=grounded['plan']['applied_motion_condition']
    person=next(o for o in grounded['tracks'] if o['evidence_id']=='track_2')
    conditioned_gaps=[]
    for row in trace:
        active_condition=(row.get('plan',{}).get('applied_motion_condition') or {})
        if (active_condition.get('object_labels',{}).get('track_2')!='person' or
                abs(float(active_condition.get('preferred_clearance_m',0))-.75)>.001):
            continue
        tracked=next((o for o in row.get('tracks',[]) if o.get('evidence_id')=='track_2'),None)
        if tracked is None: continue
        dx=tracked['position'][0]-row['pose'][0];dy=tracked['position'][1]-row['pose'][1]
        co,si=math.cos(row['pose'][2]),math.sin(row['pose'][2])
        longitudinal=max(abs(co*dx+si*dy)-.751/2,0)
        lateral=max(abs(-si*dx+co*dy)-.500/2,0)
        gap=math.hypot(longitudinal,lateral)-tracked['radius']-.025
        conditioned_gaps.append((gap,row,tracked))
    min_conditioned_gap,best_gap_row,best_gap_track=min(conditioned_gaps,key=lambda q:q[0])
    px,py=best_gap_track['position']
    robot_circumradius=math.hypot(.751/2,.500/2)
    pr=person['radius']+cond['preferred_clearance_m']+robot_circumradius
    ring=np.asarray([pixel([px+pr*math.cos(t),py+pr*math.sin(t),.20]) for t in np.linspace(0,2*math.pi,65)])
    a.plot(ring[:,0],ring[:,1],color='#9A7410',lw=.8,ls=(0,(2,2)),zorder=5)
    pp=pixel([px,py,.20]);a.plot(pp[0],pp[1],marker='o',ms=3.2,mfc='white',mec='#252525',mew=.8,zorder=7)
    a.plot(pp[0],pp[1],marker='o',ms=5.5,mfc='none',mec='#9A7410',mew=.9,zorder=8)
    # Highlight only the logged path samples for which the D614 person condition
    # is active. This links the VLM-grounded object constraint to MPPI execution.
    cond_rows=[r for r in trace if
        (r.get('plan',{}).get('applied_motion_condition') or {}).get('object_labels',{}).get('track_2')=='person' and
        abs(float((r.get('plan',{}).get('applied_motion_condition') or {}).get('preferred_clearance_m',0))-.75)<.001 and
        r['timestamp']<=bg['sim_time']]
    chunks=[];chunk=[]
    for row in cond_rows:
        if chunk and row['timestamp']-chunk[-1]['timestamp']>.5:
            if len(chunk)>1: chunks.append(chunk)
            chunk=[]
        chunk.append(row)
    if len(chunk)>1: chunks.append(chunk)
    for chunk in chunks:
        xy=np.asarray([pixel(r['pose']) for r in chunk])
        a.plot(xy[:,0],xy[:,1],color='#147D83',lw=1.7,zorder=5)
    # Mark the logged closest-approach pose and its measured gap to the track proxy.
    bx,by,byaw=best_gap_row['pose'];co,si=math.cos(byaw),math.sin(byaw)
    dx,dy=px-bx,py-by
    lx=np.clip(co*dx+si*dy,-.751/2,.751/2)
    ly=np.clip(-si*dx+co*dy,-.500/2,.500/2)
    near=(bx+co*lx-si*ly,by+si*lx+co*ly)
    vec=np.asarray([px-near[0],py-near[1]],dtype=float);vec/=np.linalg.norm(vec)
    proxy_edge=np.asarray([px,py])-vec*(best_gap_track['radius']+.025)
    npx=pixel([near[0],near[1],.20]);opx=pixel([proxy_edge[0],proxy_edge[1],.20])
    a.plot([npx[0],opx[0]],[npx[1],opx[1]],color='#9A7410',lw=.9,zorder=7)
    a.annotate(f'{min_conditioned_gap:.2f} m',xy=((npx[0]+opx[0])/2,(npx[1]+opx[1])/2),
               xytext=(3,2),textcoords='offset points',fontsize=6,color='#252525',
               bbox={'fc':'white','ec':'none','alpha':.9,'pad':.6},zorder=8)
    footprint=np.asarray([pixel([x,y,.20]) for x,y in box_corners(bx,by,byaw,.751/2,.500/2)])
    a.add_patch(Polygon(footprint,closed=True,facecolor='#147D83',edgecolor='#124E52',
                        linewidth=.7,alpha=.45,zorder=6))
    a.annotate('',xy=pixel([bx+.34*co,by+.34*si,.20]),xytext=pixel([bx,by,.20]),
               arrowprops={'arrowstyle':'-|>','lw':.7,'color':'#124E52'},zorder=7)
    from matplotlib.lines import Line2D
    legend=[Line2D([0],[0],color='#9A9A9A',lw=.9,label='Service route'),
            Line2D([0],[0],color='#147D83',lw=1.7,label='MPPI under D614 condition'),
            Line2D([0],[0],color='#9A7410',lw=.8,ls=(0,(2,2)),label='VLM · 0.75 m clearance')]
    a.legend(handles=legend,loc='upper left',bbox_to_anchor=(.015,.955),ncol=3,
             frameon=False,fontsize=6,handlelength=1.1,columnspacing=.9,handletextpad=.35,borderaxespad=0)
    b.imshow(rgbimg);b.set_axis_off();b.set_title('RGB observation at 614',loc='left',fontsize=8,pad=3)
    b.annotate('D614 contact',xy=(375,150),xytext=(390,45),fontsize=7,color='#222222',
               bbox={'fc':'white','ec':'none','alpha':.9,'pad':1.},arrowprops={'arrowstyle':'->','lw':.7,'color':'#222222'})
    b.plot(375,150,marker='o',ms=7,mfc='none',mec='#9A7410',mew=.9,zorder=8)

    # Log-derived local trajectory and body poses around receipt pickup and exit.
    graph=SpatialGraph(run/'world.sdf')
    c.set_title('Reorientation at the 614 exit (X–Y, m)',loc='left',fontsize=8,pad=3)
    xmin,xmax,ymin,ymax=1.95,4.75,-6.45,-3.65
    for bx,by,byaw,bhx,bhy in graph.static_boxes:
        if bx+bhx<xmin or bx-bhx>xmax or by+bhy<ymin or by-bhy>ymax: continue
        c.add_patch(Polygon(box_corners(bx,by,byaw,bhx,bhy),closed=True,
                            facecolor='#D9D9D9',edgecolor='#8A8A8A',linewidth=.45,zorder=1))
    door=graph.doors['D614']
    c.plot(door['aperture'],[door['plane_y']]*2,color='#444444',lw=1.0,zorder=2)
    t0=ack[2]['sim_time'];t1=next(e['sim_time'] for e in events
        if e['event']=='SEMANTIC_QUERY_STARTED' and e['sim_time']>t0 and e.get('stage')=='EXIT_CLEAR')
    local=[r for r in trace if t0<=r['timestamp']<=t1 and
           xmin<=r['pose'][0]<=xmax and ymin<=r['pose'][1]<=ymax]
    c.plot([r['pose'][0] for r in local],[r['pose'][1] for r in local],
           color='#333333',lw=1.15,zorder=3)
    before=tr(t0);after=tr(89.4)
    for row,fc,ec,ls,label in [(before,'white','#666666','--','After receipt'),
                                (after,'#D8B34A','#252525','-','Exit heading')]:
        corners=box_corners(*row['pose'],.751/2,.500/2)
        c.add_patch(Polygon(corners,closed=True,facecolor=fc,edgecolor=ec,
                            linewidth=.8,linestyle=ls,alpha=.85,zorder=5))
        c.plot(row['pose'][0],row['pose'][1],marker='o',ms=2.2,color=ec,zorder=6)
    c.annotate('',xy=(before['pose'][0],before['pose'][1]-.38),
               xytext=(before['pose'][0],before['pose'][1]-.06),
               arrowprops={'arrowstyle':'-|>','lw':.8,'color':'#666666','linestyle':'--'},zorder=7)
    c.annotate('',xy=(after['pose'][0],after['pose'][1]+.38),
               xytext=(after['pose'][0],after['pose'][1]+.06),
               arrowprops={'arrowstyle':'-|>','lw':.9,'color':'#252525'},zorder=7)
    c.set_xlim(xmin,xmax);c.set_ylim(ymin,ymax);c.set_aspect('equal',adjustable='box')
    c.set_xlabel('$x$',fontsize=7,labelpad=5);c.set_ylabel('$y$ (m)',fontsize=7,labelpad=1)
    c.set_yticks([-4.5,-5.0,-5.5,-6.0])
    c.tick_params(labelsize=6,length=2,pad=1);c.grid(color='#E6E6E6',lw=.4,zorder=0)
    out=ROOT/'paper/figures/conversational_service_overview'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=400)
    sources=[bgfile,rgbfile,response,packet,ep/'trace.jsonl',ep/'selection_events.jsonl',ep/'overview_frames.jsonl',run/'world.sdf']
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'One integrated Gazebo episode, overhead camera and onboard RGB',
      'background':bg,'rgb_query':11,'rgb_query_sim_time':rgbtime,
      'local_maneuver_window_sim_time':[t0,t1],
      'local_maneuver_poses':{'receipt_pose':before['pose'],'exit_pose':after['pose']},
      'grounded_object_track':{'id':person['evidence_id'],'label':'person','position':person['position'],
        'proxy_radius_m':person['radius'],'preferred_clearance_m':cond['preferred_clearance_m'],
        'drawn_center_envelope_radius_m':pr,'robot_circumradius_m':robot_circumradius,
        'logged_min_footprint_to_track_proxy_gap_m':min_conditioned_gap,
        'gap_samples':len(conditioned_gaps)},
      'rgb_crop_xywh':None,'rgb_annotation':'Human-added D614 contact pointer marks the person labelled track_2 in the recorded response; task-directory identity is not visually verified.',
      'route':'All recorded robot-center poses up to background frame time are shown in grey. Teal segments are logged poses while the VLM-grounded D614 track_2 person condition with 0.75 m preferred clearance was active. The highlighted footprint and gap marker use the minimum logged oriented-footprint-to-track-proxy clearance in those samples. Panel (c) plots logged poses and kinematic body outlines from the same run.',
      'simulated_confirmations':[ack[0],ack[2],ack[3]],'camera_pose':cam,'horizontal_fov':fov,'projection_height_m':.2,
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
