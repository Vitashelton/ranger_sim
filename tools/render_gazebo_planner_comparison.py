"""Single-map view of matched Gazebo trolley-passage executions."""
from pathlib import Path
import hashlib, json, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from plot_nav2_matched_cases import corners
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic')]
from robotics_plot_primitives import draw_map, vehicle
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph

BATCH=ROOT/'artifacts/task_condition_benchmark_20261023/task_matrix_v8'
SEED=53
METHODS=[('nav2','Nav2 Omni/Ranger','#333333','--'),
         ('fixed','Fixed boundary','#9b9b9b',':'),
         ('hybrid','Searched boundary','#9a762e','-')]

def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def main():
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False})
    world=BATCH/f'heading_passing_heading_clearance_hybrid_s{SEED}/world.sdf'
    graph=SpatialGraph(world)
    fig,ax=plt.subplots(figsize=(7.16,3.05))
    fig.subplots_adjust(left=.08,right=.99,bottom=.19,top=.80)
    bounds=(3.0,9.0,-4.2,-1.0)
    draw_map(ax,graph.static_boxes,bounds)
    route_path=BATCH/f'heading_passing_heading_clearance_hybrid_s{SEED}/episode/route_guidance.jsonl'
    route=read(route_path)[0]['path']
    ref=np.asarray(route,float)
    ax.plot(ref[:,0],ref[:,1],color='#aaaaaa',ls='-.',lw=.8,label='Global reference',zorder=2)
    sources=[world,route_path]
    summary=[]
    for method,label,color,style in METHODS:
        run=BATCH/f'heading_passing_heading_clearance_{method}_s{SEED}'
        trace_path=run/'episode/trace.jsonl'
        gap_path=run/'clearance_trace.json'
        sources += [trace_path,gap_path]
        trace=read(trace_path)
        gaps=json.loads(gap_path.read_text())
        closest=min(gaps,key=lambda r:r['trolley_signed_SAT_gap_m'])
        row=min(trace,key=lambda r:abs(float(r['timestamp'])-float(closest['timestamp'])))
        xy=np.asarray([r['pose'][:2] for r in trace],dtype=float)
        ax.plot(xy[:,0],xy[:,1],color=color,ls=style,lw=1.45,label=label,zorder=3)
        # Keep the object and vehicle at their logged closest-approach states.
        tx,ty=closest['trolley_pose'][:2]
        ax.add_patch(Polygon(corners(closest['trolley_pose'],.35,.21),closed=True,
                             facecolor='#eeeeee',edgecolor=color,linewidth=.8,
                             linestyle=style,zorder=4))
        vehicle(ax,row,color,width=.75)
        summary.append({'method':method,'minimum_signed_gap_m':closest['trolley_signed_SAT_gap_m'],
                        'time_s':closest['timestamp'],'robot_pose':row['pose'],
                        'trolley_pose':closest['trolley_pose']})
    import gzip
    rollouts=BATCH/f'heading_passing_heading_clearance_hybrid_s{SEED}/episode/hybrid_rollouts.jsonl.gz'
    with gzip.open(rollouts,'rt') as stream:
        plan=min((json.loads(line) for line in stream),key=lambda x:abs(x['request_sim_time']-15.48))
    selected=np.asarray(plan['plan']['selected_trajectory'],float)
    ax.plot(selected[:,0],selected[:,1],color='#9a762e',ls=(0,(4,2)),lw=1.1,
            label='Local MPPI prediction',zorder=5)
    sources.append(rollouts)
    ax.set_xlim(bounds[0],bounds[1]);ax.set_ylim(bounds[2],bounds[3])
    ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    ax.set_xticks([3,4,5,6,7,8,9]);ax.set_yticks([-4,-3,-2,-1])
    ax.tick_params(length=2,pad=2)
    ax.legend(frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),
              fontsize=6.5,handlelength=1.5,columnspacing=1.0)
    out=ROOT/'paper/figures/gazebo_planner_comparison'
    for ext in ('pdf','svg','png'): fig.savefig(str(out)+'.'+ext,dpi=450)
    source_hash={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    Path(str(out)+'.provenance.json').write_text(json.dumps({
        'kind':'Executed trajectories and closest-approach vehicle/object proxies from three matched Gazebo runs',
        'matched_trial':'heading-conditioned trolley passage, same map, task, initial condition and seed across planners',
        'display':'One metric XY map. Each line is an executed trace from a separate run. Outlined Ranger footprints and trolley disks are the logged states at each run’s minimum signed SAT gap.',
        'cases':summary,'source_sha256':source_hash},indent=2)+'\n')
    plt.close(fig)
if __name__=='__main__':main()
