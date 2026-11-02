"""Same-episode service feedback and MPPI execution, with actual Gazebo imagery."""
from pathlib import Path
import argparse,json,hashlib,math,subprocess,sys
from bisect import bisect_right
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rangermini_dynamic_semantic'),str(ROOT/'tools')]
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from robotics_plot_primitives import draw_map,vehicle,MODE_COLORS

def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def main(a):
 campaign=a.campaign.resolve();run=campaign/'run';ep=run/'episode';out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
 audit=json.loads((campaign/'audit.json').read_text());assert audit['qualified'],'Episode failed independent audit'
 trace=rows(ep/'trace.jsonl');events=rows(ep/'selection_events.jsonl');over=rows(ep/'overview_frames.jsonl');views=rows(ep/'frames.jsonl');ack=[e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED'];assert len(ack)==4
 cam=ET.parse(run/'world.sdf').getroot().find("world/model[@name='navigation_overhead_camera']");pose=list(map(float,cam.findtext('pose').split()));fov=float(cam.findtext('link/sensor/camera/horizontal_fov'));roll,pitch,yaw=pose[3:];cr,sr,cp,sp,cy,sy=math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
 R=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]);focal=1280/(2*math.tan(fov/2))
 def pixel(p):
  q=R.T@(np.array([p[0],p[1],.2])-np.asarray(pose[:3]));return np.array([640-focal*q[1]/q[0],360-focal*q[2]/q[0]])
 def near(t):return min(trace,key=lambda r:abs(r['timestamp']-t))
 plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 fig=plt.figure(figsize=(7.16,3.65));gs=fig.add_gridspec(2,3,left=.065,right=.99,bottom=.16,top=.91,wspace=.32,hspace=.25,height_ratios=[.8,1.6]);graph=SpatialGraph(run/'world.sdf');sources=[run/'world.sdf',ep/'trace.jsonl',ep/'selection_events.jsonl'];frames=[]
 specs=[(ack[0],'(a) 617 · Report collection',8.11,(-2.0,1.5)),(ack[2],'(b) 614 · Departure with receipt',3.25,(-5.7,-2.2)),(ack[3],'(c) 619 · Final delivery',12.8,(-2.0,1.5))]
 for j,(event,label,xcenter,ylim) in enumerate(specs):
  t=event['sim_time'];frame_time=t+10 if j==1 else t;frame=min(over,key=lambda r:abs(r['sim_time']-frame_time));fp=ep/'overview_frames'/f"{frame['frame']:05d}.jpg";img=cv2.cvtColor(cv2.imread(str(fp)),cv2.COLOR_BGR2RGB);center=pixel(near(t)['pose']);w,h=370,245;x=int(np.clip(center[0]-w/2,0,1280-w));y=int(np.clip(center[1]-h/2,0,720-h));image_ax=fig.add_subplot(gs[0,j]);image_ax.imshow(img[y:y+h,x:x+w]);image_ax.axis('off');image_ax.set_title(label,loc='left',fontsize=7.5)
  ax=fig.add_subplot(gs[1,j]);limits=(xcenter-1.1,xcenter+1.1,*ylim);draw_map(ax,graph.static_boxes,limits)
  window=[r for r in trace if t-18<=r['timestamp']<=t+(18 if j<2 else 0)]
  xy=np.array([r['pose'][:2] for r in window]);segments=np.stack([xy[:-1],xy[1:]],axis=1)
  colors=np.array([MODE_COLORS.get(r['actuation']['mode'],'#c0c4c5') for r in window[:-1]])
  visible=np.all((segments[:,:,0]>=limits[0])&(segments[:,:,0]<=limits[1])&(segments[:,:,1]>=limits[2])&(segments[:,:,1]<=limits[3]),axis=1)
  ax.add_collection(LineCollection(segments[visible],colors=colors[visible],linewidths=1.25))
  stop=near(t);ax.plot(*stop['pose'][:2],'o',ms=4,mfc='white',mec='#222',mew=.8,zorder=8)
  spins=[r for r in window if r['actuation']['mode']=='SPINNING' and abs(r['actuation'].get('executed',{}).get('wz',0))>.02]
  if spins:
   q=spins[len(spins)//2]['pose'];ax.scatter([q[0]],[q[1]],s=18,facecolors='white',edgecolors=MODE_COLORS['SPINNING'],linewidths=.8,zorder=10)
  for tt in ([t,t+10] if j<2 else [t-8]):
   row=near(tt);xx,yy=row['pose'][:2]
   if limits[0]+.42<xx<limits[1]-.42 and limits[2]+.42<yy<limits[3]-.42:vehicle(ax,row,'#39464c',width=.55)
  ax.set_xlabel('x (m)',fontsize=7);ax.set_ylabel('y (m)',fontsize=7);ax.tick_params(labelsize=6.5)
  sources.append(fp);frames.append(dict(event=event,frame=frame,crop_xywh=[x,y,w,h],trace_window=[t-18,t+(18 if j<2 else 0)]))
 fig.legend([Patch(fc=MODE_COLORS[k]) for k in ['DUAL_ACKERMANN','PARALLEL','SPINNING']],['Curved motion','Parallel motion','Rotation'],loc='lower center',ncol=3,frameon=False,fontsize=7,bbox_to_anchor=(.5,.01))
 sys.path.insert(0,'/home/USER/.codex/skills/nature-figure/scripts')
 from audit_panel_alignment import require_matplotlib_panel_alignment
 require_matplotlib_panel_alignment(fig,json_out=out/'service_feedback.alignment.json',panel_ids=['image1','path1','image2','path2','image3','path3'],row_groups=[['image1','image2','image3'],['path1','path2','path3']],exemptions=[{'panels':['image1','image2','image3','path1','path2','path3'],'checks':['column'],'reason':'Camera image crops and metric XY axes retain distinct aspect ratios; equal sizes enforced within each comparable row.'}])
 for ext in ['pdf','svg','png']:fig.savefig(out/f'service_feedback.{ext}',dpi=400)
 provenance=dict(kind='Same conversational service episode, recorded camera and executed model states',frames=frames,raw_times_retained_in_provenance_only=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
 if a.video:
  trace_times=[r['timestamp'] for r in trace];over_times=[r['sim_time'] for r in over];view_times=[r['sim_time'] for r in views];origin=trace_times[0]-trace[0]['elapsed'];end=max(trace_times[-1],ack[-1]['sim_time'])+2
  tmp=out/'service_demo_temporary.mp4';writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),10,(1280,720));assert writer.isOpened()
  for t in np.arange(origin,end,.2):
   row=trace[max(0,bisect_right(trace_times,t)-1)];ov=over[max(0,bisect_right(over_times,t)-1)];rgb=views[max(0,bisect_right(view_times,t)-1)]
   image=cv2.imread(str(ep/'overview_frames'/f"{ov['frame']:05d}.jpg"));center=pixel(row['pose']);w,h=860,484;x=int(np.clip(center[0]-w/2,0,1280-w));y=int(np.clip(center[1]-h/2,0,720-h));image=cv2.resize(image[y:y+h,x:x+w],(1280,720))
   def screen(p):return tuple(np.round((pixel(p)-[x,y])*[1280/w,720/h]).astype(int))
   q=row['pose'];cv2.arrowedLine(image,screen(q),screen([q[0]+.45*math.cos(q[2]),q[1]+.45*math.sin(q[2])]),(100,180,180),2,tipLength=.25)
   camera=cv2.imread(str(ep/'frames'/f"{rgb['frame']:05d}.jpg"));image[438:708,908:1268]=cv2.resize(camera,(360,270))
   overlay=image.copy();cv2.rectangle(overlay,(12,12),(870,124),(20,28,30),-1);image=cv2.addWeighted(overlay,.72,image,.28,0)
   confirmed=sum(e['sim_time']<=t for e in ack);mode=row['actuation']['mode'];labels={'DUAL_ACKERMANN':'Curved motion','PARALLEL':'Parallel motion','SPINNING':'Rotation','STOPPED':'Stopped'}
   room=row['anchor_id'].lstrip('DH');active='Task complete' if confirmed==4 else 'Active goal: Office '+room
   lines=['617 report collection > 614 sign-off > 619 receipt return',active+' | Simulated handovers: '+str(confirmed)+'/4','MPPI execution: '+labels.get(mode,mode)+(' | steering preparation' if row['actuation'].get('steering_transition_stop') else '')]
   for n,line in enumerate(lines):cv2.putText(image,line,(24,40+n*31),cv2.FONT_HERSHEY_SIMPLEX,.59,(245,247,248),1,cv2.LINE_AA)
   writer.write(image)
  writer.release();dest=out/'service_demo_h264.mp4';subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(tmp),'-an','-c:v','libx264','-preset','medium','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(dest),'-y'],check=True);tmp.unlink()
  provenance.update(video_speed='2x simulation time',video_includes_last_ack=True,final_camera_frame_held_until_logged_confirmation=True)
 (out/'service_feedback.provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--video',action='store_true');main(p.parse_args())
