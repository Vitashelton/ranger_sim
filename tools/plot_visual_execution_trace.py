"""Actual observation -> VLM condition -> measured multimode execution.

This figure documents the planning chain; comparative performance remains in
the complete archived pair and is not inferred from this single execution.
"""
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/rangermini_dynamic_semantic'))
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
sys.path.insert(0, '/home/USER/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

RUN = ROOT/'artifacts/hybrid_sequence_20261022/task_condition_pair_v6/vlm'
EP = RUN/'episode'
OUT = ROOT/'paper/figures'


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    trace = rows(EP/'trace.jsonl')
    events = rows(EP/'selection_events.jsonl')
    response_file = EP/'proposal_002_response.json'
    input_file = EP/'proposal_002_input.json'
    image_file = EP/'proposal_002_input.jpg'
    raw_file = EP/'proposal_002_raw_rgb.jpg'
    proposal = json.loads(response_file.read_text())['proposal']
    accepted = next(e for e in events if e['event']=='TASK_CONDITION_BOUND' and e['proposal']==proposal)
    geometry = json.loads((RUN/'clearance_trace.json').read_text())
    assert len(trace)==len(geometry)
    poses = np.asarray([r['pose'] for r in trace])
    modes = [(r.get('actuation') or {}).get('mode','STOPPED') for r in trace]
    selected_indices = [next(i for i,r in enumerate(trace) if r['timestamp']>=accepted['sim_time']),
        min(range(len(geometry)),key=lambda i:geometry[i]['trolley_signed_SAT_gap_m']),
        next(i for i,r in enumerate(trace) if r['pose'][0]>=8.2)]
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'legend.frameon':False})
    fig = plt.figure(figsize=(7.16,2.4))
    a,b,c = [fig.add_axes(rect) for rect in
             ([.01,.28,.27,.59],[.33,.28,.27,.59],[.69,.28,.30,.59])]
    h,w = plt.imread(raw_file).shape[:2]
    a.imshow(plt.imread(image_file)[38:38+h,:w],aspect='auto');a.set_axis_off()
    a.set_title('(a) Visual observation',loc='left',fontsize=8,pad=7)
    b.set_axis_off();b.set_title('(b) Recorded VLM condition',loc='left',fontsize=8,pad=7)
    fields=[('Task anchor','Room 619'),('Object','track_1: '+proposal['object_labels']['track_1']),
            ('Clearance preference',f"{proposal['preferred_clearance_m']:.2f} m"),
            ('Soft motion preference','Retain heading; parallel')]
    for index,(label,value) in enumerate(fields):
        y=.91-index*.245
        b.text(0,y,label,fontsize=7,fontweight='bold',va='top')
        b.text(0,y-.105,value,fontsize=7.5,va='top')
    colors={'DUAL_ACKERMANN':'#ce7c35','PARALLEL':'#168c89','SPINNING':'#785a98','STOPPED':'#969da1'}
    # Explicit local display crop; original poses and all paired metrics remain intact.
    in_view=(poses[:,0]>=4)&(poses[:,0]<=8.5)
    segments=np.stack((poses[:-1,:2],poses[1:,:2]),axis=1)
    visible=in_view[:-1]&in_view[1:]
    c.add_collection(LineCollection(segments[visible],colors=[colors.get(m,'#969da1') for m,v in zip(modes[:-1],visible) if v],linewidths=1.5))
    cart=np.asarray([r['trolley_pose'] for r in geometry]);keep=(cart[:,0]>=4)&(cart[:,0]<=8.5)
    c.plot(cart[keep,0],cart[keep,1],ls='--',color='#777b7e',lw=.8)
    for number,index in enumerate(selected_indices,1):
        pose=poses[index];co,si=math.cos(pose[2]),math.sin(pose[2])
        corners=np.array([[-.3755,-.25],[.3755,-.25],[.3755,.25],[-.3755,.25]])
        outline=corners@np.array([[co,si],[-si,co]])+pose[:2]
        c.add_patch(Polygon(outline,fill=False,edgecolor='#34414b',lw=.65))
        c.plot(*pose[:2],marker='o',mfc='white',mec='#34414b',ms=3)
        c.text(pose[0],pose[1]+(.95 if number==2 else .6),str(number),ha='center',fontsize=7)
    c.set(xlim=(4,8.5),ylim=(-3.6,-.6),xlabel='$x$ (m)',ylabel='$y$ (m)')
    c.set_aspect('equal',adjustable='box')
    c.set_title('(c) Measured MPPI execution',loc='left',fontsize=8,pad=7)
    fig.legend(handles=[Line2D([0],[0],color=color,lw=1.5,label=label,ls='--' if label=='Trolley' else '-') for label,color in
        [('Curved',colors['DUAL_ACKERMANN']),('Parallel',colors['PARALLEL']),('Trolley','#777b7e')]],
        loc='lower center',bbox_to_anchor=(.79,.005),ncol=3,fontsize=6.5,
        handlelength=1.2,columnspacing=.8)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=[a,b,c],panel_ids=['a','b','c'],
        row_groups=[['a','b']],json_out=OUT/'visual_execution_trace.alignment.json')
    fig.savefig(OUT/'visual_execution_trace.pdf')
    fig.savefig(OUT/'visual_execution_trace.svg')
    fig.savefig(OUT/'visual_execution_trace.png',dpi=600)
    plt.close(fig)
    files=[input_file,response_file,image_file,raw_file,EP/'trace.jsonl',EP/'selection_events.jsonl',
           RUN/'clearance_trace.json',Path(__file__)]
    provenance=dict(run=str(RUN.relative_to(ROOT)),claim='visual task conditioning and observed motion execution',
        comparison_claim=False,image_crop=[0,38,w,38+h],image_adjustments='crop and resize only',
        trajectory='measured Gazebo poses; no smoothing or synthesized points',
        xy_display_region=dict(x=[4,8.5],y=[-3.6,-.6]),
        proposal=proposal,accepted_sim_time=accepted['sim_time'],
        selected_poses=[dict(index=i,sim_time=trace[i]['timestamp'],pose=trace[i]['pose'],mode=modes[i]) for i in selected_indices],
        complete_pair='artifacts/hybrid_sequence_20261022/task_condition_pair_v6/analysis/metrics.json',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (OUT/'visual_execution_trace.provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    print('Rendered actual VLM-condition and motion-execution trace.')


if __name__=='__main__':
    main()
