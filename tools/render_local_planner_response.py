"""Matched local trajectories with executed heading and wheel response, no global route inset."""
from pathlib import Path
import hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic'),'/home/USER/.codex/skills/nature-figure/scripts']
from robotics_plot_primitives import draw_map,vehicle
from plot_nav2_matched_cases import corners
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    batch=ROOT/'artifacts/task_condition_benchmark_20261023/task_matrix_v8'
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':7.5,'xtick.labelsize':6.5,
      'ytick.labelsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
      'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.16,2.65))
    gs=fig.add_gridspec(2,2,left=.075,right=.985,bottom=.18,top=.83,width_ratios=[1.25,1],hspace=.47,wspace=.34)
    a=fig.add_subplot(gs[:,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,1],sharex=b)
    lim=(4.9,8.2,-3.85,-1.65);src=[]
    world=batch/'heading_passing_heading_clearance_hybrid_s41/world.sdf';src.append(world)
    draw_map(a,SpatialGraph(world).static_boxes,lim);a.set_aspect('equal',adjustable='datalim')
    for method,color,ls,label in [('nav2','#CE8449','--','Nav2 Omni/Ranger'),('fixed','#777E84',':','Fixed boundaries'),('hybrid','#137E83','-','Searched boundaries')]:
        run=batch/f'heading_passing_heading_clearance_{method}_s41'
        path=run/'episode/trace.jsonl';gapfile=run/'clearance_trace.json';src.extend([path,gapfile])
        trace=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        q=np.asarray([r['pose'] for r in trace]);t=np.array([r['elapsed'] for r in trace])
        a.plot(q[:,0],q[:,1],c=color,ls=ls,lw=1.1,label=label)
        gaps=json.loads(gapfile.read_text());g=min(gaps,key=lambda x:x['trolley_signed_SAT_gap_m'])
        j=min(range(len(trace)),key=lambda i:abs(trace[i]['timestamp']-g['timestamp']))
        vehicle(a,trace[j],color,width=.6)
        a.add_patch(Polygon(corners(g['trolley_pose'],.35,.21),fill=False,ec=color,lw=.65,ls=ls))
        yaw=np.unwrap(q[:,2]);b.plot(t,np.rad2deg(yaw-yaw[0]),c=color,ls=ls,lw=.9)
        c.plot(t,np.rad2deg([r['actuation']['wheel_steering_rad'][0] for r in trace]),c=color,ls=ls,lw=.9)
    a.set(xlabel='x (m)',ylabel='y (m)');a.set_title('(a) Local passage',loc='left',fontsize=8,pad=4)
    b.set_ylabel(r'$\Delta\psi$ (deg)');b.set_title('(b) Executed heading',loc='left',fontsize=8,pad=4)
    c.set(xlabel='Execution time (s)',ylabel=r'$\delta_{FL}$ (deg)');c.set_title('(c) Steering response',loc='left',fontsize=8,pad=4)
    b.tick_params(labelbottom=False)
    h,l=a.get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,fontsize=7,frameon=False,bbox_to_anchor=(.5,1.015),columnspacing=1.2,handlelength=1.7)
    fig.canvas.draw()
    for ax in fig.axes:
        lo=(*ax.get_xlim(),*ax.get_ylim())
        for line in ax.lines:
            x=np.asarray(line.get_xdata(),float);y=np.asarray(line.get_ydata(),float)
            valid=(x>=lo[0])&(x<=lo[1])&(y>=lo[2])&(y<=lo[3]);line.set_data(np.where(valid,x,np.nan),np.where(valid,y,np.nan))
    require_matplotlib_panel_alignment(fig,json_out=ROOT/'paper/figures/local_planner_response.alignment.json',strict=True)
    out=ROOT/'paper/figures/local_planner_response'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'Matched executed local trajectories, body heading and FL model wheel angle',
       'selection':'First registered repeat of heading-conditioned passage, all three configurations retained.',
       'obstacle_snapshots':'Each run independently evaluated closest-approach sample, robot and trolley synchronized within run.',
       'state_transform':'Original trace samples. Heading unwrapped relative to initial yaw, radians converted to degrees, no smoothing.',
       'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src}},indent=2)+'\n')

if __name__=='__main__':main()
