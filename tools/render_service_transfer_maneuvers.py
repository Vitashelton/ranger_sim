"""Actual document and receipt transport legs from one integrated service mission."""
from pathlib import Path
import hashlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic'),'/home/USER/.codex/skills/nature-figure/scripts']
from robotics_plot_primitives import draw_map,vehicle
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    run=ROOT/'artifacts/task_guided_online_20261023/mission_v6_cpp_s41/run'
    ep=run/'episode';read=lambda p:[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    trace=read(ep/'trace.jsonl');events=read(ep/'selection_events.jsonl')
    ack=[e for e in events if e['event']=='SERVICE_ACTION_CONFIRMED']
    graph=SpatialGraph(run/'world.sdf')
    plt.rcParams.update({'font.family':'Liberation Serif','font.size':8,'axes.labelsize':8,
        'xtick.labelsize':7,'ytick.labelsize':7,'axes.linewidth':.6,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.75))
    fig.subplots_adjust(left=.07,right=.99,bottom=.23,top=.89,wspace=.18)
    colors={'PARALLEL':'#137E83','DUAL_ACKERMANN':'#777E84','SPINNING':'#242424','STOPPED':'#B8BEC2'}
    info=[]
    definitions=[(ack[2],ack[3],'(a) 614: departure',(1.9,4.9,-6.3,-1.8)),
                 (ack[2],ack[3],'(b) 619: delivery',(10.8,13.8,-2.8,1.7))]
    for a,(start,end,title,lim) in zip(axes,definitions):
        data=[r for r in trace if start['sim_time']<=r['timestamp']<=end['sim_time']]
        xy=np.asarray([r['pose'][:2] for r in data])
        draw_map(a,graph.static_boxes,lim);a.set_aspect('equal',adjustable='box')
        shown=xy.copy()
        shown[(shown[:,0]<lim[0])|(shown[:,0]>lim[1])|(shown[:,1]<lim[2])|(shown[:,1]>lim[3])]=np.nan
        seg=np.stack([shown[:-1],shown[1:]],axis=1)
        a.add_collection(LineCollection(seg,colors=[colors[r['actuation']['mode']] for r in data[:-1]],linewidths=1.3,zorder=3))
        # Body glyphs follow actual travel progress, independent of control mode.
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
        visible=np.flatnonzero((xy[:,0]>lim[0]+.25)&(xy[:,0]<lim[1]-.25)&(xy[:,1]>lim[2]+.25)&(xy[:,1]<lim[3]-.25))
        local=s[visible]-s[visible[0]]
        indices=sorted(set([int(visible[np.argmin(abs(local-f*local[-1]))]) for f in (.12,.5,.82)]))
        for j in indices:
            vehicle(a,data[j],'#25343C',width=.6)
        # Mark start of each actual spinning interval, not an assigned mode.
        spin=[j for j,r in enumerate(data) if r['actuation']['mode']=='SPINNING' and
              (j==0 or data[j-1]['actuation']['mode']!='SPINNING')]
        for j in spin:
            if lim[0]<xy[j,0]<lim[1] and lim[2]<xy[j,1]<lim[3]:a.plot(*xy[j],marker='o',ms=3,mfc='white',mec='#242424',mew=.8,zorder=8)
        if lim[0]<xy[0,0]<lim[1] and lim[2]<xy[0,1]<lim[3]:a.plot(*xy[0],marker='s',ms=3.5,mfc='white',mec='#242424',mew=.7,zorder=8)
        if lim[0]<xy[-1,0]<lim[1] and lim[2]<xy[-1,1]<lim[3]:a.plot(*xy[-1],marker='*',ms=6,color='#242424',zorder=8)
        for e in (start,end):
            r=min(data,key=lambda r:abs(r['timestamp']-e['sim_time']))
            if not (lim[0]<r['pose'][0]<lim[1] and lim[2]<r['pose'][1]<lim[3]):continue
            offset=(0,9) if e['room']!='614' else (13,-10)
            a.annotate(e['room'],xy=r['pose'][:2],xytext=offset,textcoords='offset points',ha='center',fontsize=7)
        a.set(xlabel='x (m)',ylabel='y (m)');a.set_title(title,loc='left',fontsize=7.5,pad=4)
        info.append({'start_event':start,'end_event':end,'body_glyph_trace_timestamps':[data[j]['timestamp'] for j in indices],
                     'spin_start_timestamps':[data[j]['timestamp'] for j in spin],'path_m':float(s[-1])})
    fig.legend([Patch(fc=colors[k]) for k in ('DUAL_ACKERMANN','PARALLEL','SPINNING')],
               ['Curved motion','Parallel translation','Reorientation'],loc='lower center',ncol=3,frameon=False,
               fontsize=7,bbox_to_anchor=(.5,.015),handlelength=.9,columnspacing=1.2)
    require_matplotlib_panel_alignment(fig,json_out=ROOT/'paper/figures/service_transfer_maneuvers.alignment.json',strict=True)
    out=ROOT/'paper/figures/service_transfer_maneuvers'
    for ext in ('pdf','svg','png'):fig.savefig(str(out)+'.'+ext,dpi=450)
    src=[ep/'trace.jsonl',ep/'selection_events.jsonl',run/'world.sdf']
    Path(str(out)+'.provenance.json').write_text(json.dumps({'kind':'Receipt acquisition/departure and delivery-approach regions within one measured service mission',
        'selection':'Exact logged simulated pickup/delivery confirmation boundaries. No independent episodes joined.',
        'glyph_rule':'Nearest logged samples at 12%, 50%, 82% of the visible cumulative path length in each metric window.',
        'mode_source':'Executed kinematic actuation mode, not requested planner phase.',
        'legs':info,'metric_windows':[x[3] for x in definitions],'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in src}},indent=2)+'\n')

if __name__=='__main__':main()
