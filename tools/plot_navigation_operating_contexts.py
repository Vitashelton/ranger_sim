"""Recorded detour, heading-constrained passage and complete service mission.

The first two panels compare the frozen family-based controller with the
public Nav2 optimizer. They are not trials of the newer hybrid timing search.
"""
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rangermini_dynamic_semantic'))
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,'/home/USER/.codex/skills/nature-figure/scripts')
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from plot_nav2_matched_cases import corners,clipped_polygon
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.65,
        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.16,2.3))
    axes=[fig.add_axes(r)for r in ([.08,.32,.24,.59],[.40,.32,.24,.59],[.72,.32,.24,.59])]
    sources=[Path(__file__)]
    for ax,scene,label in zip(axes[:2],['static_detour','heading_constrained_passage'],
                              ['(a) Obstacle detour','(b) Heading-constrained passage']):
        lim=(7.,10.5,-3.5,-1.3)
        for method,col in [('nav2-omni','#ca7a35'),('multimode','#168c89')]:
            batch='path_window_fix_v4_baseline' if method=='nav2-omni' else 'frozen_routefix_v3'
            run=ROOT/f'artifacts/nav2_matched_cases_20261021/{batch}/{scene}_{method}_s17'
            source=run/'episode/trace.jsonl';sources.extend([source,run/'world.sdf'])
            trace=[json.loads(l)for l in source.open()];poses=np.array([r['pose']for r in trace])
            if method=='nav2-omni':
                for box in SpatialGraph(run/'world.sdf').static_boxes:
                    pts=clipped_polygon(corners(box[:3],*box[3:]),lim)
                    if pts:ax.add_patch(Polygon(pts,fc='#d6d9db',ec='#939a9e',lw=.4))
            visible=(poses[:,0]>=lim[0])&(poses[:,0]<=lim[1])
            xy=np.where(visible[:,None],poses[:,:2],np.nan)
            ax.plot(xy[:,0],xy[:,1],c=col,lw=1.4)
        ax.set(xlim=lim[:2],ylim=lim[2:],xlabel='$x$ (m)',ylabel='$y$ (m)')
        ax.set_title(label,loc='left',fontsize=7.5,pad=6)
        ax.set_aspect('equal',adjustable='box')
    ax=axes[2]
    run=ROOT/'artifacts/hybrid_sequence_20261022/service_static_native_v3'
    source=run/'episode/trace.jsonl';events=run/'episode/selection_events.jsonl'
    sources.extend([source,events,run/'world.sdf'])
    trace=[json.loads(l)for l in source.open()];poses=np.array([r['pose']for r in trace]);lim=(2,14,-5.8,1.7)
    for box in SpatialGraph(run/'world.sdf').static_boxes:
        pts=clipped_polygon(corners(box[:3],*box[3:]),lim)
        if pts:ax.add_patch(Polygon(pts,fc='#d6d9db',ec='#939a9e',lw=.3))
    colors={'DUAL_ACKERMANN':'#ca7a35','PARALLEL':'#168c89','SPINNING':'#785b96','STOPPED':'#939a9e'}
    segments=np.stack([poses[:-1,:2],poses[1:,:2]],axis=1)
    ax.add_collection(LineCollection(segments,colors=[colors[r['actuation']['mode']]for r in trace[:-1]],lw=1.25))
    for text in ('617','614','619'):
        # Place room labels near actual confirmation positions from recorded events.
        room_events=[json.loads(l)for l in events.open()if json.loads(l).get('event')=='SERVICE_ACTION_CONFIRMED' and json.loads(l).get('room')==text]
        stamp=room_events[-1]['sim_time'];index=min(range(len(trace)),key=lambda i:abs(trace[i]['timestamp']-stamp))
        position=poses[index,:2]
        ax.plot(*position,'o',mfc='white',mec='#34414b',ms=3)
        offsets={'617':(4,0),'614':(7,3),'619':(-18,0)}
        ax.annotate(text,position,xytext=offsets[text],textcoords='offset points',fontsize=6.5)
    ax.set(xlim=lim[:2],ylim=lim[2:],xlabel='$x$ (m)',ylabel='$y$ (m)')
    ax.set_title('(c) Multistop service mission',loc='left',fontsize=7.5,pad=6)
    ax.set_aspect('equal',adjustable='box')
    fig.legend(handles=[Line2D([0],[0],c='#ca7a35',lw=1.4,label='Nav2 Omni'),
                        Line2D([0],[0],c='#168c89',lw=1.4,label='Family-based')],
        loc='lower left',bbox_to_anchor=(.06,.005),ncol=2,frameon=False,fontsize=6.5,handlelength=1.4)
    fig.legend(handles=[Line2D([0],[0],c=colors[m],lw=1.4,label=l)for m,l in
        [('DUAL_ACKERMANN','Curved'),('PARALLEL','Parallel'),('SPINNING','Spin')]],
        loc='lower right',bbox_to_anchor=(1.,.005),ncol=3,frameon=False,fontsize=6.5,handlelength=1.,columnspacing=.7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abc'),row_groups=[list('abc')],
        json_out=ROOT/'paper/figures/navigation_operating_contexts.alignment.json')
    output=ROOT/'paper/figures'
    fig.savefig(output/'navigation_operating_contexts.pdf')
    fig.savefig(output/'navigation_operating_contexts.svg')
    fig.savefig(output/'navigation_operating_contexts.png',dpi=600)
    provenance=dict(scope='Two frozen controller local-navigation cases and one separate full service mission',
        panels_ab='Seed17, earlier family controller without hybrid sequence search; public Nav2 retains Omni sampling',
        panel_c='Recorded service_static_native_v3 mission; archived model plan and simulated service confirmations',
        transforms='Local XY display crop in a,b; original measured poses with no smoothing',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources})
    (output/'navigation_operating_contexts.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Rendered obstacle detour, heading-constrained passage, and multistop service.')


if __name__=='__main__':main()
