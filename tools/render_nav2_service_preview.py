"""Matched 4WS planner executions: Gazebo frames, paths and model states."""
from pathlib import Path
import hashlib,json,math,sys,xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic'),'/home/USER/.codex/skills/nature-figure/scripts']
from robotics_plot_primitives import draw_map,vehicle
from plot_nav2_matched_cases import corners
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_panel_alignment import require_matplotlib_panel_alignment

def read(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
    batch=ROOT/'artifacts/task_condition_benchmark_20261023/task_matrix_v8';methods=[('nav2','Nav2 Omni MPPI','#333333','--'),('hybrid','Task-guided MPPI','#9a762e','-')]
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':7.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.16,3.7));gs=fig.add_gridspec(2,2,left=.07,right=.985,bottom=.17,top=.90,hspace=.35,wspace=.27,height_ratios=[.72,1.2])
    images=[fig.add_subplot(gs[0,i]) for i in range(2)];paths=[fig.add_subplot(gs[1,i]) for i in range(2)]
    world=batch/'heading_passing_heading_clearance_hybrid_s41/world.sdf';lim=(2.7,13.4,-4.0,-.3)
    for a in paths:
        draw_map(a,SpatialGraph(world).static_boxes,lim);a.set_aspect('equal',adjustable='box')
    info=[];sources=[world]
    for image_ax,a,(method,label,color,ls) in zip(images,paths,methods):
        run=batch/f'heading_passing_heading_clearance_{method}_s41';ep=run/'episode'
        path=ep/'trace.jsonl';gapfile=run/'clearance_trace.json';framesfile=ep/'overview_frames.jsonl';worldfile=run/'world.sdf';sources += [path,gapfile,framesfile,worldfile]
        trace=read(path);gap=min(json.loads(gapfile.read_text()),key=lambda x:x['trolley_signed_SAT_gap_m'])
        frame=min(read(framesfile),key=lambda x:abs(x['sim_time']-gap['timestamp']))
        assert abs(frame['sim_time']-gap['timestamp'])<.2 and frame['pose'] is not None
        imgfile=ep/'overview_frames'/f"{frame['frame']:05d}.jpg";sources.append(imgfile);pixels=np.asarray(Image.open(imgfile).convert('RGB'))
        camera=ET.parse(worldfile).getroot().find("world/model[@name='navigation_overhead_camera']")
        pose=list(map(float,camera.findtext('pose').split()));fov=float(camera.findtext('link/sensor/camera/horizontal_fov'))
        roll,pitch,yaw=pose[3:];cr,sr,cp,sp,cy,sy=math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
        R=np.array([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],[sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]])
        focal=pixels.shape[1]/(2*math.tan(fov/2))
        def pix(p):
            q=R.T@(np.array([p[0],p[1],.2])-np.asarray(pose[:3]))
            return np.array([pixels.shape[1]/2-focal*q[1]/q[0],pixels.shape[0]/2-focal*q[2]/q[0]])
        mid=(pix(frame['pose'])+pix(gap['trolley_pose']))/2;w,h=340,180
        x=int(np.clip(round(mid[0]-w/2),0,pixels.shape[1]-w));y=int(np.clip(round(mid[1]-h/2),0,pixels.shape[0]-h))
        image_ax.imshow(pixels[y:y+h,x:x+w]);image_ax.axis('off');image_ax.set_title(label,loc='left',fontsize=8,color=color,pad=3)
        xy=np.asarray([r['pose'] for r in trace]);t=np.asarray([r['elapsed'] for r in trace]);a.plot(xy[:,0],xy[:,1],c=color,ls=ls,lw=1.05,label=label)
        j=min(range(len(trace)),key=lambda i:abs(trace[i]['timestamp']-gap['timestamp']))
        vehicle(a,trace[j],color,width=.6);a.add_patch(Polygon(corners(gap['trolley_pose'],.35,.21),fill=False,ec=color,lw=.65,ls=ls))
        a.set_xlabel('x (m)');a.set_ylabel('y (m)')
        a.plot(12.8,-2.1,marker='*',color='#202020',ms=7,zorder=10); a.set_xticks([3,6,9,12]);a.set_yticks([-3.5,-2.5,-1.5])
        info.append({'method':method,'gap_timestamp':gap['timestamp'],'gap_m':gap['trolley_signed_SAT_gap_m'],
                     'frame':frame,'crop_xywh':[x,y,w,h],'robot_state':trace[j]['pose'],'trolley_state':gap['trolley_pose']})
    fig.canvas.draw()
    for ax in paths:
        lo=(*ax.get_xlim(),*ax.get_ylim())
        for line in ax.lines:
            xs=np.asarray(line.get_xdata(),float);ys=np.asarray(line.get_ydata(),float)
            ok=(xs>=lo[0])&(xs<=lo[1])&(ys>=lo[2])&(ys<=lo[3]);line.set_data(np.where(ok,xs,np.nan),np.where(ok,ys,np.nan))
    # Frame panels are directly comparable; path and state panels have distinct roles.
    require_matplotlib_panel_alignment(fig,json_out=ROOT/'results/local_motion_preview/nav2_service.alignment.json',strict=True,
        panel_ids=['img1','img2','path1','path2'],row_groups=[['img1','img2'],['path1','path2']],
        column_groups=[],exemptions=[{'panels':['img1','img2','path1','path2'],'checks':['column'],'reason':'Native image crops and metric XY plots retain their own aspect ratios. Equal widths, heights and gutters are enforced within each row.'}])
    out=ROOT/'results/local_motion_preview/nav2_service'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=400)
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'Two Gazebo overhead camera frames and executed trajectories from matched passage trials',
       'trial_selection':'First registered repeat; per-method frame nearest its own independently computed minimum trolley gap, not a common timestamp.',
       'crop':'Metric camera projection of robot/trolley midpoint, fixed 340x180 pixel crop for all methods. No synthesized frames.',
       'plot':'Separate equal-scale XY panels show each method with its own closest-approach robot/trolley configuration. Footprint and wheel outlines are logged model states; no simultaneous-pose claim across methods.',
       'cases':info,'source_sha256':{str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}},indent=2)+'\n')
if __name__=='__main__':main()
