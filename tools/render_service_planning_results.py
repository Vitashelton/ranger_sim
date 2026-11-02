"""Service-task spatial evidence and paired executed states from frozen logs."""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic'),'/home/USER/.codex/skills/nature-figure/scripts']
from robotics_plot_primitives import vehicle,draw_map
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_panel_alignment import require_matplotlib_panel_alignment


def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]

def main():
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    out=ROOT/'paper/figures';teal='#137E83';gray='#737A80'
    run=ROOT/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run'
    trace=rows(run/'episode/trace.jsonl');events=rows(run/'episode/selection_events.jsonl')
    confirms=[e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED']
    graph=SpatialGraph(run/'world.sdf');poses=np.asarray([r['pose'] for r in trace])
    fig=plt.figure(figsize=(7.16,2.9))
    ax=fig.add_axes([.06,.17,.46,.75]);limits=(2.3,13.6,-5.95,1.85)
    draw_map(ax,graph.static_boxes,limits)
    ax.plot(poses[:,0],poses[:,1],color=teal,lw=1.25)
    ax.plot(*poses[0,:2],marker='o',ms=3.5,mfc='white',mec='black',mew=.7)
    # Room labels refer to actual confirmed arrival positions, not inferred waypoints.
    cases=[('617',confirms[0],'Collect documents'),('614',confirms[2],'Deliver / collect receipt'),('619',confirms[3],'Deliver receipt')]
    for room,event,label in cases:
        j=min(range(len(trace)),key=lambda i:abs(trace[i]['timestamp']-event['sim_time']))
        x,y=poses[j,:2];ax.plot(x,y,'o',ms=3,color='#222222')
        ax.annotate(room,xy=(x,y),xytext=((8,-12) if room=='614' else (0,9)),textcoords='offset points',fontsize=8,ha='center')
    ax.set(xlabel='x (m)',ylabel='y (m)')
    ax.set_title('(a) Continuous service route',fontsize=8,loc='left',pad=3)
    # Three service poses use the same metric zoom scale and logged steering channels.
    axes=[]
    for col,(room,event,title) in enumerate(cases):
        a=fig.add_axes([.57+col*.15,.26,.10,.58]);axes.append(a)
        door=graph.doors['D'+room];x=door['center_x'];py=door['plane_y'];direction=door['direction']
        limits=(x-1.15,x+1.15,py-1.4 if direction>0 else py-2.15,py+2.15 if direction>0 else py+1.4)
        # All local panels have equal x/y spans, irrespective of room side.
        draw_map(a,graph.static_boxes,limits)
        indices=[i for i,r in enumerate(trace) if r['anchor_id'] in ('D'+room,'H'+room)]
        xy=poses[indices];a.plot(xy[:,0],xy[:,1],c=teal,lw=1.0)
        j=min(indices,key=lambda i:abs(trace[i]['timestamp']-event['sim_time']))
        vehicle(a,trace[j],'#25343C',width=.55)
        # Add heading arrows at actual approach and departure samples.
        sampled=[]
        for target in (event['sim_time']-5,event['sim_time']+4):
            k=min(indices,key=lambda i:abs(trace[i]['timestamp']-target))
            q=poses[k]
            if limits[0]<q[0]<limits[1] and limits[2]<q[1]<limits[3]:
                a.annotate('',xy=q[:2]+.34*np.array([np.cos(q[2]),np.sin(q[2])]),xytext=q[:2],
                    arrowprops={'arrowstyle':'->','lw':.85,'color':gray})
                sampled.append(k)
        a.set(xlabel='x (m)',ylabel='y (m)' if col==0 else '')
        a.tick_params(labelsize=6)
        a.set_title(f'({chr(98+col)}) {room}',fontsize=7.5,loc='left',pad=3)
    # Clip line geometry to the displayed metric domain before PDF export.
    for a in fig.axes:
        lim=(*a.get_xlim(),*a.get_ylim())
        for line in a.lines:
            x=np.asarray(line.get_xdata(),dtype=float);y=np.asarray(line.get_ydata(),dtype=float)
            good=(x>=lim[0])&(x<=lim[1])&(y>=lim[2])&(y<=lim[3])
            line.set_data(np.where(good,x,np.nan),np.where(good,y,np.nan))
    # Intentional unequal-span overview, comparable local zooms only.
    require_matplotlib_panel_alignment(fig,json_out=out/'service_navigation.alignment.json',strict=True,
                                      panel_ids=['a','b','c','d'],row_groups=[['b','c','d']])
    for ext in ('pdf','svg','png'):fig.savefig(out/f'service_navigation.{ext}',dpi=400)
    source=[run/'world.sdf',run/'episode/trace.jsonl',run/'episode/selection_events.jsonl',run/'long_task_plan.json']
    (out/'service_navigation.provenance.json').write_text(json.dumps({'kind':'One-run actual service navigation, no new experiments',
        'run':str(run.relative_to(ROOT)),'confirmation':'Explicitly simulated acknowledgements',
        'local_panel_selection':'All trace samples whose active anchor belongs to the registered room. Vehicle at closest confirmation sample. Heading arrows at closest samples five seconds before and four seconds after confirmation, when inside zoom.',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}},indent=2)+'\n')
    plt.close(fig)
    batch=ROOT/'artifacts/task_condition_benchmark_20261023/task_matrix_v8'
    fig,axes=plt.subplots(2,1,figsize=(3.45,2.6),sharex=True)
    fig.subplots_adjust(left=.18,right=.98,bottom=.16,top=.89,hspace=.30)
    sources=[]
    for key,color,ls,label in [('fixed',gray,'--','Fixed boundaries'),('hybrid',teal,'-','Searched boundaries')]:
        path=batch/f'heading_passing_heading_clearance_{key}_s41/episode/trace.jsonl';sources.append(path)
        data=rows(path);t=np.array([r['elapsed'] for r in data]);yaw=np.unwrap([r['pose'][2] for r in data])
        wheel=[r['actuation']['wheel_steering_rad'][0] for r in data]
        axes[0].plot(t,np.rad2deg(yaw-yaw[0]),color=color,ls=ls,lw=1.0,label=label)
        axes[1].plot(t,np.rad2deg(wheel),color=color,ls=ls,lw=1.0)
    axes[0].set_ylabel('Heading change (deg)');axes[1].set_ylabel(r'$\delta_{FL}$ (deg)')
    axes[1].set_xlabel('Execution time (s)');axes[0].legend(loc='upper left',bbox_to_anchor=(-.05,1.32),fontsize=7,ncol=2,frameon=False,handlelength=1.4,columnspacing=.8)
    require_matplotlib_panel_alignment(fig,json_out=out/'executed_heading_steering.alignment.json',strict=True)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'executed_heading_steering.{ext}',dpi=400)
    (out/'executed_heading_steering.provenance.json').write_text(json.dumps({'kind':'Actual executed pose and model wheel state; internal timing pair',
        'seed':41,'transformation':'Unsmoothed original samples, yaw unwrapped and relative to initial heading; FL wheel radians converted to degrees.',
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')

if __name__=='__main__':main()
