"""Matched service-pose approach: trajectory and heading evolution."""
from pathlib import Path
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic')]
from robotics_plot_primitives import draw_map, vehicle
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
BASE=ROOT/'artifacts/offset_pose_public_pair_20261029/matched_v1'
METHODS=[('nav2','Nav2 Omni MPPI','#333333','--'),
         ('hybrid','Task-Attentive MPPI','#9a762e','-')]
GOAL=(12.8,.55,math.pi/2)
def read(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def wrap(a): return (a+math.pi)%(2*math.pi)-math.pi

def main():
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':7.5,
        'axes.labelsize':7.5,'xtick.labelsize':6.7,'ytick.labelsize':6.7,
        'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    fig,(ax,ah)=plt.subplots(2,1,figsize=(3.42,3.55),
        gridspec_kw={'height_ratios':[1.05,.82],'hspace':.32})
    fig.subplots_adjust(left=.17,right=.98,bottom=.12,top=.97)
    world=BASE/'nav2_s53/world.sdf'; graph=SpatialGraph(world)
    draw_map(ax,graph.static_boxes,(10.8,13.35,-1.9,.95))
    records=[]; sources=[world]
    for method,label,color,style in METHODS:
        run=BASE/f'{method}_s53/episode/trace.jsonl'; rows=read(run); sources.append(run)
        xy=np.asarray([r['pose'][:2] for r in rows],float)
        yaw=np.unwrap(np.asarray([r['pose'][2] for r in rows],float))
        ds=np.r_[0,np.linalg.norm(np.diff(xy,axis=0),axis=1)]
        dist=np.cumsum(ds)
        err=np.abs(np.degrees([wrap(float(q)-GOAL[2]) for q in yaw]))
        ax.plot(xy[:,0],xy[:,1],color=color,ls=style,lw=1.2,zorder=3)
        idx=next(i for i,r in enumerate(rows) if math.dist(r['pose'][:2],GOAL[:2])<=.18)
        vehicle(ax,rows[idx],color,width=.75)
        ah.plot(dist,err,color=color,ls=style,lw=1.15,label=label)
        records.append({'method':method,'arrival_pose':rows[idx]['pose'],
                        'arrival_distance_m':float(dist[idx]),
                        'terminal_heading_error_deg':float(err[idx])})
    ax.plot(GOAL[0],GOAL[1],marker='*',ms=6,color='#222222',zorder=6)
    ax.annotate('',xy=(GOAL[0],GOAL[1]+.28),xytext=(GOAL[0],GOAL[1]),
                arrowprops={'arrowstyle':'-|>','lw':.8,'color':'#222222'})
    ax.set_xlim(10.8,13.35);ax.set_ylim(-1.9,.95);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    ax.tick_params(length=2,pad=1.5);ax.set_xticks([11,12,13]);ax.set_yticks([-1,0,.5])
    ah.set_xlabel('Distance along executed path (m)')
    ah.set_ylabel('Heading error (deg)')
    ah.set_ylim(0,100);ah.set_yticks([0,30,60,90])
    ah.tick_params(length=2,pad=1.5)
    ah.legend(frameon=False,loc='upper right',fontsize=6.4,handlelength=1.5)
    out=ROOT/'paper/figures/offset_public_pair_single'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
    out.with_suffix('.provenance.json').write_text(json.dumps({
        'kind':'Matched Gazebo service-pose approach with executed XY paths and heading-error profiles',
        'goal_pose':GOAL,'trials':['nav2_s53','hybrid_s53'],
        'heading_error':'absolute wrapped error from 90 degree goal heading, sampled from logged pose',
        'cases':records,'source_sha256':{str(p.relative_to(ROOT)):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
    plt.close(fig)
if __name__=='__main__':main()
