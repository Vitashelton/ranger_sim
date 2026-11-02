"""Single-episode Gazebo observations linked to the measured service route."""
from pathlib import Path
import hashlib,json,math,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def main():
    run=ROOT/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run';ep=run/'episode'
    trace=read(ep/'trace.jsonl');events=read(ep/'selection_events.jsonl');over=read(ep/'overview_frames.jsonl')
    event=next(e for e in events if e['event']=='SEMANTIC_QUERY_STARTED' and e.get('query')==2)
    ack=[e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED']
    bg=over[-1];dep=min(over,key=lambda x:abs(x['sim_time']-98.0))
    bgfile=ep/'overview_frames'/f"{bg['frame']:05d}.jpg";depfile=ep/'overview_frames'/f"{dep['frame']:05d}.jpg"
    rgbfile=ep/'proposal_002_raw_rgb.jpg';response=ep/'proposal_002_response.json';packet=ep/'proposal_002_input.json'
    assert json.loads(response.read_text())['proposal']['object_labels'].get('track_2')=='trolley'
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
    rgbtime=event['sim_time'];dept=dep['sim_time'];qr=tr(rgbtime);dr=tr(dept)
    assert abs(qr['timestamp']-rgbtime)<.25 and abs(dr['timestamp']-dept)<.25
    # The background is a single recorded overhead frame; path lines are past
    # robot-center history through that frame, not a claim of simultaneous poses.
    active=[r for r in trace if r['timestamp']<=bg['sim_time']]
    uv=np.asarray([pixel(r['pose']) for r in active])
    center=pixel(dr['pose']);dw,dh=300,210
    x=int(np.clip(round(center[0]-dw/2),0,raw.shape[1]-dw));y=int(np.clip(round(center[1]-dh/2),0,raw.shape[0]-dh))
    depimg=np.asarray(Image.open(depfile).convert('RGB'))[y:y+dh,x:x+dw]
    rgbimg=np.asarray(Image.open(rgbfile).convert('RGB'))
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.16,3.15));a=fig.add_axes([.025,.14,.61,.76]);b=fig.add_axes([.69,.55,.28,.34]);c=fig.add_axes([.69,.12,.28,.34])
    a.imshow(raw);a.set_axis_off();a.set_title('(a) Collection, exchange and delivery',loc='left',fontsize=8,pad=3)
    a.plot(uv[:,0],uv[:,1],c='#147D83',lw=1.0,zorder=4)
    for e,label,delta in [(ack[0],'617 · Collect',(6,-11)),(ack[2],'614 · Exchange',(-64,-2))]:
        q=pixel(tr(e['sim_time'])['pose']);a.plot(*q,'o',ms=3,mfc='white',mec='#26343A',mew=.6,zorder=6)
        a.annotate(label,xy=q,xytext=delta,textcoords='offset points',fontsize=7,color='#222222',
                   bbox={'fc':'white','ec':'none','alpha':.8,'pad':1.0},zorder=7)
    q619=pixel(tr(bg['sim_time'])['pose']);a.plot(*q619,'*',ms=6,c='#222222',zorder=6)
    a.annotate('619 · Deliver',xy=q619,xytext=(6,7),textcoords='offset points',fontsize=7,
               bbox={'fc':'white','ec':'none','alpha':.8,'pad':1.0},zorder=7)
    # Object observation and departure positions remain in provenance, not linked by crossing lines.
    b.imshow(rgbimg);b.set_axis_off();b.set_title('(b) Task-relevant trolley',loc='left',fontsize=8,pad=3)
    b.annotate('Trolley',xy=(401,235),xytext=(475,80),fontsize=7,color='#222222',
               bbox={'fc':'white','ec':'none','alpha':.9,'pad':1.},arrowprops={'arrowstyle':'->','lw':.7,'color':'#222222'})
    c.imshow(depimg);c.set_axis_off();c.set_title('(c) Departure after receipt collection',loc='left',fontsize=8,pad=3)
    out=ROOT/'paper/figures/gazebo_service_evidence'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=400)
    sources=[bgfile,depfile,rgbfile,response,packet,ep/'trace.jsonl',ep/'selection_events.jsonl',ep/'overview_frames.jsonl',run/'world.sdf']
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'One integrated Gazebo episode, overhead camera and onboard RGB',
      'background':bg,'rgb_query':2,'rgb_query_sim_time':rgbtime,'departure_frame':dep,'departure_crop_xywh':[x,y,dw,dh],
      'rgb_crop_xywh':None,'rgb_annotation':'Human-added Trolley pointer identifies the task-associated track_2 in the raw RGB. No timestamp labels or cross-panel connectors.',
      'route':'All recorded robot-center poses up to background frame time projected into the stationary overhead camera; background is one instant, route is accumulated motion history.',
      'simulated_confirmations':[ack[0],ack[2],ack[3]],'camera_pose':cam,'horizontal_fov':fov,'projection_height_m':.2,
      'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
