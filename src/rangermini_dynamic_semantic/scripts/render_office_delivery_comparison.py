#!/usr/bin/env python3
"""Render recorded Office A challenge episodes as RViz-style top-down panels."""
import argparse, json, math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np

try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ImportError:
    require_matplotlib_panel_alignment=None

mpl.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['DejaVu Sans','sans-serif'],
    'font.size':8,'axes.titlesize':10,'axes.labelsize':8,
    'svg.fonttype':'none','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,
})

PERSON=(10.10,-2.10); CART=(7.35,-1.05); GOAL=(11.79,1.00)
WALL='#343a40'; ROUTE='#2367a5'; CANDIDATE='#efa14b'; SELECTED='#14966b'

def episode(path):
    p=Path(path)
    if (p/'episode').is_dir():p=p/'episode'
    result=json.loads((p/'result.json').read_text())
    trace=[json.loads(s) for s in (p/'trace.jsonl').read_text().splitlines() if s.strip()]
    rollouts=sorted(p.glob('selection_*_rollouts.json'))
    candidates=json.loads(rollouts[0].read_text()) if rollouts else []
    selected=None
    events=p/'selection_events.jsonl'
    if events.exists():
        for line in events.read_text().splitlines():
            row=json.loads(line)
            if row.get('event')=='selected':selected=row.get('candidate_id')
    return p,result,trace,candidates,selected

def wall(ax,x1,x2,y,linewidth=2.6):
    ax.plot([x1,x2],[y,y],color=WALL,linewidth=linewidth,solid_capstyle='butt',zorder=2)

def draw_scene(ax,person,cart):
    ax.set_facecolor('#fbfcfd')
    ax.fill_between([2.4,13.35],[-.3,-.3],[-3.9,-3.9],color='#f1f4f6',zorder=0)
    ax.fill_between([10.08,13.50],[-.3,-.3],[3.9,3.9],color='#f6f2e9',zorder=0)
    # Corridor and Office 619 boundaries from the generated Gazebo layout.
    for a,b in ((2.4,4.58),(5.50,7.64),(8.56,12.33),(13.25,13.55)):
        wall(ax,a,b,-.30)
    for a,b in ((3.70,7.18),(8.12,12.65)):
        wall(ax,a,b,-3.90)
    ax.plot([10.08,10.08],[-.3,3.9],color=WALL,lw=2.6,zorder=2)
    ax.plot([13.50,13.50],[-.3,3.9],color=WALL,lw=2.6,zorder=2)
    wall(ax,10.08,13.50,3.9)
    # Door jambs and the open leaf; the remaining aperture is visible.
    ax.plot([12.33,13.25],[-.3,-.3],color='#fbfcfd',lw=3.2,zorder=3)
    for x in (12.33,13.25):
        ax.add_patch(Rectangle((x-.0225,-.39),.045,.18,facecolor=WALL,edgecolor='none',zorder=4))
    ax.plot([12.355,12.355],[-.30,.54],color='#9a693d',lw=2.2,zorder=3)
    ax.text(11.8,3.55,'Office A · Room 619',ha='center',fontsize=8,color='#4c5149',fontweight='bold')
    # Existing desk with the generated red document inbox target on top.
    ax.add_patch(Rectangle((11.79-.4565,2.10-.4565),.913,.913,facecolor='#c9c1b3',
                           edgecolor='#6d675d',lw=1.0,zorder=4))
    ax.add_patch(Rectangle((11.79-.17,2.10-.14),.34,.28,facecolor='#d83328',
                           edgecolor='#8e211a',lw=1.0,zorder=5))
    ax.add_patch(Rectangle((11.79-.125,2.10-.10),.25,.20,facecolor='#f4f0df',
                           edgecolor='none',zorder=6))
    ax.text(12.30,2.35,'red document inbox',fontsize=6.7,color='#a32920',zorder=7)
    # Open fire-door leaves and frame cues at x=10.305.
    ax.plot([10.305,10.305],[-1.23,-.39],color='#7f8b94',lw=1.8,zorder=3)
    ax.plot([10.305,10.305],[-3.81,-2.97],color='#7f8b94',lw=1.8,zorder=3)
    # South-side lab entrances and jambs.
    for x in (3.24,7.65):
        ax.plot([x-.46,x-.46],[-3.98,-3.82],color='#68737b',lw=1.2,zorder=3)
        ax.plot([x+.46,x+.46],[-3.98,-3.82],color='#68737b',lw=1.2,zorder=3)
    # Task-relevant obstacle proxies.
    ax.add_patch(Circle(person,.18,facecolor='#bd6b5e',edgecolor='#8f4038',lw=1.0,zorder=5))
    ax.add_patch(Circle(person,1.23,facecolor='none',edgecolor='#b84b40',lw=1.0,
                        linestyle=(0,(3,2)),alpha=.72,zorder=3))
    ax.text(person[0]-.88,person[1]+.94,'0.60 m task buffer*',fontsize=6.5,color='#9b3e35')
    ax.text(person[0]-.95,person[1]+.36,'waiting person',ha='center',va='bottom',fontsize=7,color='#7f3730')
    ax.add_patch(Rectangle((cart[0]-.35,cart[1]-.21),.70,.42,facecolor='#8798a4',
                           edgecolor='#53616b',lw=1.0,zorder=4))
    ax.add_patch(Rectangle((cart[0]-.95,cart[1]-.81),1.90,1.62,facecolor='none',
                           edgecolor='#627580',lw=.9,linestyle=(0,(2,2)),alpha=.75,zorder=3))
    ax.text(cart[0],cart[1]-.32,'equipment trolley · 0.35 m buffer*',ha='center',va='top',fontsize=6.5,color='#3f4b53')
    ax.add_patch(Circle(GOAL,.20,facecolor='#e7bd42',edgecolor='#806317',lw=1.1,zorder=8))
    ax.text(GOAL[0]-.12,GOAL[1]-.39,'desk-side approach pose',fontsize=6.5,color='#70581b')
    ax.set_xlim(2.35,13.65);ax.set_ylim(-4.25,4.15);ax.set_aspect('equal',adjustable='box')
    ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
    ax.grid(color='#aeb8bf',lw=.45,alpha=.42,zorder=1)

def plot_episode(ax,data,title,person,cart,show_candidates=True):
    p,result,trace,candidates,selected=data
    draw_scene(ax,person,cart)
    if show_candidates:
        candidate_label_added=False
        for c in candidates:
            if c.get('motion_behavior')=='STOP':continue
            traj=np.asarray(c.get('trajectory',[]),dtype=float)
            if len(traj)>1:
                ax.plot(traj[:,0],traj[:,1],color=CANDIDATE,lw=.8,alpha=.30,zorder=3,
                        label='Feasible MPPI rollouts' if not candidate_label_added else None)
                candidate_label_added=True
        if selected:
            row=next((c for c in candidates if c.get('candidate_id')==selected),None)
            if row:
                traj=np.asarray(row['trajectory'],dtype=float)
                ax.plot(traj[:,0],traj[:,1],color=SELECTED,lw=2.0,ls='--',zorder=4,
                label=f'Visual choice: {selected}')
    poses=np.asarray([r['pose'][:3] for r in trace if r.get('pose')],dtype=float)
    if len(poses):
        ax.plot(poses[:,0],poses[:,1],color=ROUTE,lw=2.4,zorder=6,label='Executed Gazebo path')
        ax.scatter(*poses[0,:2],s=34,marker='o',facecolor='white',edgecolor=ROUTE,lw=1.4,zorder=7)
        ax.scatter(*poses[-1,:2],s=45,marker='X',color=ROUTE,zorder=7)
        x,y,yaw=poses[-1]
        body=Rectangle((x-.751/2,y-.5/2),.751,.5,facecolor='#2367a5',
                       edgecolor='white',alpha=.65,zorder=7)
        body.set_transform(Affine2D().rotate_around(x,y,yaw)+ax.transData)
        ax.add_patch(body)
    ax.set_title(title,loc='left',fontweight='bold')
    if show_candidates or len(poses):ax.legend(loc='upper left',fontsize=7,framealpha=.95)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--geometry',required=True,help='geometry episode directory')
    ap.add_argument('--visual',help='visual MPPI episode directory (optional for pilot review)')
    ap.add_argument('--layout',choices=['person_clearance','person_clearance_mirror','doorway_delivery'],default='person_clearance')
    ap.add_argument('--output',required=True,help='output basename, no extension')
    a=ap.parse_args()
    cfg={'person_clearance':((6.5,-2.1),(8.65,-1.05)),
         'person_clearance_mirror':((6.5,-3.15),(8.65,-1.05)),
         'doorway_delivery':((10.1,-2.1),(7.35,-1.05))}
    person,cart=cfg[a.layout]
    cases=[('Geometry MPPI',episode(a.geometry))]
    if a.visual:cases.append(('Visual MPPI',episode(a.visual)))
    fig,axes=plt.subplots(1,len(cases),figsize=(7.2,4.5 if len(cases)>1 else 4.2),squeeze=False)
    for ax,(title,data) in zip(axes[0],cases):plot_episode(ax,data,title,person,cart)
    fig.suptitle('Office A service mission: corridor passage → doorway entry → inbox approach',
                 fontsize=11,fontweight='bold')
    fig.tight_layout(rect=(0,.22,1,.93))
    fig.canvas.draw()
    for ax,(_,data) in zip(axes[0],cases):
        result=data[1];bbox=ax.get_position()
        clr=result.get('person_proxy_min_clearance_m')
        cartclr=result.get('trolley_proxy_min_clearance_m')
        line1=(f"{result.get('status')} | person {clr:.2f}/{result.get('person_clearance_threshold_m',.60):.2f} m"
               if clr is not None else 'Person clearance unavailable')
        line2=(f"Cart {cartclr:.2f}/{result.get('trolley_clearance_threshold_m',.35):.2f} m | "
               f"door {'✓' if result.get('doorway_crossed_into_room_619') else '×'} | "
               f"inbox RGB {'✓' if result.get('inbox_rgb_confirmed') else '×'} | "
               f"mission {'PASS' if result.get('task_success') else 'FAIL'}"
               if cartclr is not None else f"Delivery {'PASS' if result.get('task_success') else 'FAIL'}")
        fig.text(bbox.x0,.115,line1+'\n'+line2,ha='left',va='center',fontsize=7.0,
                 color='#28343b',linespacing=1.2)
    fig.text(.5,.025,'Dashed buffers are diagrammatic; scoring uses the simulated oriented Ranger footprint and recorded RGB.',
             ha='center',va='bottom',fontsize=6.5,color='#4f5960')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    if len(cases)>1:
        if require_matplotlib_panel_alignment is None:
            raise SystemExit('alignment audit helper unavailable; cannot export paired figure')
        for label,ax in zip(('a','b'),axes[0]):
            ax.annotate(label,xy=(0,1),xycoords='axes fraction',xytext=(-14,9),
                        textcoords='offset points',fontweight='bold',fontsize=9,ha='left',va='bottom')
        require_matplotlib_panel_alignment(fig,json_out=str(out)+'.alignment.json',
            overlay_svg=str(out)+'.alignment.svg',tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    else:
        Path(str(out)+'.alignment.json').write_text(json.dumps({'status':'NOT APPLICABLE','panels':1},indent=2))
    fig.savefig(str(out)+'.png',dpi=600,bbox_inches='tight')
    fig.savefig(str(out)+'.svg',bbox_inches='tight')
    fig.savefig(str(out)+'.pdf',bbox_inches='tight')
    fig.savefig(str(out)+'.tiff',dpi=600,bbox_inches='tight')
    print(json.dumps({'outputs':[str(out)+ext for ext in ('.png','.svg','.pdf','.tiff')],
        'episodes':[{'path':str(d[0]),'status':d[1].get('status'),
                     'person_clearance_m':d[1].get('person_proxy_min_clearance_m'),
                     'task_success':d[1].get('task_success')} for _,d in cases]},
        ensure_ascii=False,indent=2))

if __name__=='__main__':main()
