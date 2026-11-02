"""Single executed 614-to-619 service leg from the integrated Gazebo mission."""
from pathlib import Path
import hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic')]
from robotics_plot_primitives import draw_map,vehicle
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph

def main():
    run=ROOT/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run';ep=run/'episode'
    read=lambda f:[json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    events=read(ep/'selection_events.jsonl');trace=read(ep/'trace.jsonl')
    ack=[e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED'];start,end=ack[2],ack[3]
    rows=[r for r in trace if start['sim_time']<=r['timestamp']<=end['sim_time']];xy=np.asarray([r['pose'][:2] for r in rows])
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(3.45,2.55));fig.subplots_adjust(left=.12,right=.99,bottom=.24,top=.97)
    lim=(2.4,13.5,-5.9,1.6);draw_map(ax,SpatialGraph(run/'world.sdf').static_boxes,lim)
    colors={'PARALLEL':'#137E83','DUAL_ACKERMANN':'#777E84','SPINNING':'#242424','STOPPED':'#B8BEC2'}
    segments=np.stack([xy[:-1],xy[1:]],axis=1)
    ax.add_collection(LineCollection(segments,colors=[colors[r['actuation']['mode']] for r in rows[:-1]],linewidths=1.15,zorder=3))
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    indices=[]
    for mode,tag in [('SPINNING','R'),('DUAL_ACKERMANN','C'),('PARALLEL','P')]:
        candidates=[i for i,r in enumerate(rows) if r['actuation']['mode']==mode]
        if candidates:
            j=candidates[len(candidates)//2];indices.append(j)
            vehicle(ax,rows[j],'#26343a',width=.55)
            ax.annotate(tag,xy=xy[j],xytext=(5,5),textcoords='offset points',fontsize=7)

    spin=[i for i,r in enumerate(rows) if r['actuation']['mode']=='SPINNING' and (i==0 or rows[i-1]['actuation']['mode']!='SPINNING')]
    for i in spin:ax.plot(*xy[i],marker='o',ms=3.5,mfc='white',mec='#242424',mew=.8,zorder=7)
    ax.plot(*xy[0],marker='s',ms=3,mfc='white',mec='#26343a');ax.plot(*xy[-1],marker='*',ms=6,c='#26343a')
    ax.annotate('614',xy=xy[0],xytext=(8,-7),textcoords='offset points',fontsize=7)
    ax.annotate('619',xy=xy[-1],xytext=(-7,7),textcoords='offset points',fontsize=7)
    ax.set(xlabel='x (m)',ylabel='y (m)')
    fig.legend([Patch(fc=colors['DUAL_ACKERMANN']),Patch(fc=colors['PARALLEL']),Line2D([0],[0],marker='o',markersize=3.5,color='#242424',markerfacecolor='white',linestyle='None')],['Curved','Parallel','Spin'],loc='lower center',ncol=3,fontsize=7,frameon=False,bbox_to_anchor=(.5,.0),handlelength=.8,columnspacing=1.1)
    out=ROOT/'paper/figures/service_transfer_compact'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
    sources=[ep/'trace.jsonl',ep/'selection_events.jsonl',run/'world.sdf']
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'One executed service leg from receipt acquisition at 614 to delivery at 619',
       'start_event':start,'end_event':end,'vehicle_glyph_sample_times':[rows[j]['timestamp'] for j in indices],
       'mode_source':'Executed kinematic actuation feedback, not requested planner mode. Circle marks the start of a logged spin interval.',
       'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},indent=2)+'\n')
if __name__=='__main__':main()
