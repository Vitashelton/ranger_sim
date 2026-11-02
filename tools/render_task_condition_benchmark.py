"""Separate task-response and planner-comparison visuals from actual trajectories."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch, Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'src/rangermini_dynamic_semantic'),
              '/home/USER/.codex/skills/nature-figure/scripts']
from plot_nav2_matched_cases import corners,clipped_polygon
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from audit_panel_alignment import require_matplotlib_panel_alignment
from robotics_plot_primitives import vehicle, mode_strip, MODE_COLORS


def read_rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main(a):
    batch=a.batch.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((batch/'protocol.json').read_text())
    data=json.loads((batch/'analysis/metrics.json').read_text())
    seed=protocol['seeds'][0]
    contexts=protocol['contexts'];sources={};records={}
    for context in contexts:
        for condition in protocol['tasks']:
            for method in protocol['methods']:
                run=batch/f'{context}_{condition}_{method}_s{seed}'
                p=run/'episode/trace.jsonl'
                if not (run/'episode/result.json').exists():
                    raise ValueError('Representative matched cell is incomplete: '+str(run))
                records[context,condition,method]=read_rows(p)
                for f in [p,run/'world.sdf',run/'episode/result.json']:
                    sources[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    plt.rcParams.update({'font.family':['Liberation Serif','Arial'],'font.size':8,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,
        'svg.fonttype':'none','axes.spines.right':False,'axes.spines.top':False,
        'axes.linewidth':.65,'legend.frameon':False})
    teal,orange,gray='#14858C','#D98243','#858C91'
    bounds={'heading_passing':(2.5,13.4,-4.0,-.35),'static_detour':(6.4,13.4,-3.5,-.35)}
    def environment(ax,context):
        run=batch/f'{context}_ordinary_hybrid_s{seed}'
        lim=bounds[context]
        for box in SpatialGraph(run/'world.sdf').static_boxes:
            poly=corners(box[:3],*box[3:])
            if poly:ax.add_patch(Polygon(poly,fc='#D4D8DA',ec='#9CA3A7',lw=.4,zorder=0))
        if context=='heading_passing':
            # The actor path is spatial, not a synchronized actor snapshot.
            p=batch/f'{context}_heading_clearance_hybrid_s{seed}'/'clearance_trace.json'
            cart=np.array([x['trolley_pose'][:2] for x in json.loads(p.read_text())])
            cart[(cart[:,0]<lim[0])|(cart[:,0]>lim[1])|(cart[:,1]<lim[2])|(cart[:,1]>lim[3])]=np.nan
            ax.plot(cart[:,0],cart[:,1],color='#5B656B',lw=.8,ls=':',zorder=1)
            sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
        ax.set(xlim=lim[:2],ylim=lim[2:],xlabel='x (m)',ylabel='y (m)')
        ax.set_aspect('equal',adjustable='datalim')
        ax.plot(12.8,-2.1,marker='*',ms=7,mec='#35434A',mfc='white',mew=.8,zorder=8)
        start=(3.,-2.1) if context=='heading_passing' else (7.,-2.1)
        ax.plot(*start,marker='o',ms=3.5,mfc='white',mec='#35434A',mew=.7,zorder=8)
    def trajectory(ax,context,condition,method,color,label,ls='-'):
        poses=np.asarray([r['pose'] for r in records[context,condition,method]])
        ax.plot(poses[:,0],poses[:,1],c=color,lw=1.35,label=label,ls=ls)
        result=batch/f'{context}_{condition}_{method}_s{seed}'/'episode/result.json'
        if json.loads(result.read_text())['status']!='ARRIVED':
            ax.plot(poses[-1,0],poses[-1,1],marker='x',color=color,ms=6,mew=1.1)
        if context=='heading_passing':
            source=batch/f'{context}_{condition}_{method}_s{seed}'/'clearance_trace.json'
            sampled=json.loads(source.read_text())
            closest=min(sampled,key=lambda r:r['trolley_signed_SAT_gap_m'])
            index=min(range(len(poses)),key=lambda i:abs(records[context,condition,method][i]['timestamp']-closest['timestamp']))
            cart_pose=closest['trolley_pose']
            ax.add_patch(Polygon(corners(cart_pose,.35,.21),fill=False,ec=color,
                                 lw=.8,ls=ls,zorder=5))
            sources[str(source.relative_to(ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
        else:
            index=int(np.argmin(np.linalg.norm(poses[:,:2]-np.array([9.,-2.1]),axis=1)))
        vehicle(ax,records[context,condition,method][index],color)
    def save(fig,name,roles):
        fig.canvas.draw()
        for ax in fig.axes:
            lim=(*ax.get_xlim(),*ax.get_ylim())
            for line in ax.lines:
                x=np.asarray(line.get_xdata(),dtype=float);y=np.asarray(line.get_ydata(),dtype=float)
                valid=(x>=lim[0])&(x<=lim[1])&(y>=lim[2])&(y<=lim[3])
                line.set_data(np.where(valid,x,np.nan),np.where(valid,y,np.nan))
            for patch in list(ax.patches):
                if isinstance(patch,Polygon):
                    poly=clipped_polygon([tuple(x) for x in patch.get_xy()[:-1]],lim)
                    if poly:patch.set_xy(poly)
                    else:patch.remove()
        require_matplotlib_panel_alignment(fig,json_out=out/f'{name}.alignment.json',strict=True)
        fig.savefig(out/f'{name}.pdf');fig.savefig(out/f'{name}.svg');fig.savefig(out/f'{name}.png',dpi=600)
        (out/f'{name}.provenance.json').write_text(json.dumps(dict(
            kind='Actual executed Gazebo trajectories; interaction regions shown at equal spatial scale within each panel.',
            representative_seed=seed,selection_rule='First preregistered seed, all required conditions/methods retained.',
            panel_roles=roles,source_sha256=sources,
            renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            glyph_source_sha256=hashlib.sha256((ROOT/'tools/robotics_plot_primitives.py').read_bytes()).hexdigest(),
            footprint_selection="Trolley: each run's independently evaluated minimum-gap timestamp, with synchronized actor outline. Equipment: nearest recorded body-center pose to equipment. True metric body and logged steering angles.",
            transformations='Original pose samples, no smoothing; full start/goal map extents.'),indent=2)+'\n')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(7.16,2.45))
    fig.subplots_adjust(left=.075,right=.99,bottom=.27,top=.88,wspace=.20)
    names={'heading_passing':'Moving trolley','static_detour':'Stationary equipment'}
    for i,(ax,context) in enumerate(zip(axes,contexts)):
        title=f'({chr(97+i)}) '+names[context]
        environment(ax,context)
        trajectory(ax,context,'ordinary','hybrid',orange,'Ordinary request','--')
        trajectory(ax,context,'heading_clearance','hybrid',teal,'Heading + clearance')
        ax.set_title(title,loc='left',fontsize=8,pad=5)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=7.5,bbox_to_anchor=(.5,.005))
    save(fig,'task_condition_behavior',['Instruction contrast with identical hybrid planner','Same contrast in a second navigation context'])

    fig=plt.figure(figsize=(7.16,2.45))
    grid=fig.add_gridspec(1,2,left=.065,right=.99,bottom=.22,top=.82,
                          width_ratios=[1.4,1.],wspace=.22)
    axes=[fig.add_subplot(grid[0,i]) for i in range(2)]
    method_defs=[('nav2',orange,'Nav2 Omni/Ranger','--'),
                 ('fixed',gray,'Fixed boundaries',':'),('hybrid',teal,'Searched boundaries','-')]
    context='heading_passing'
    for i,ax in enumerate(axes):
        environment(ax,context)
        for method,color,label,ls in method_defs:
            trajectory(ax,context,'heading_clearance',method,color,label,ls)
        if i==0:
            ax.set_title('(a) Complete navigation',loc='left',fontsize=8,pad=4)
            ax.add_patch(Rectangle((4.9,-3.85),3.3,2.2,fill=False,ec='#48565F',lw=.6,ls='--'))
        else:
            ax.set(xlim=(4.9,8.2),ylim=(-3.85,-1.65))
            ax.set_aspect('equal',adjustable='box')
            ax.set_anchor('E')
            ax.set_title('(b) Encounter detail',loc='left',fontsize=8,pad=4)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,fontsize=7.5,
               bbox_to_anchor=(.5,1.01),handlelength=2)
    save(fig,'task_condition_planner_comparison',[
        'Full moving-trolley task and encounter zoom of the same first-seed matched episodes',
        'Metric enlargement of the same recorded trajectory region with body and wheel poses'])

    # Never insert a partially filled experiment table into the manuscript.
    if data['completed_episodes'] != protocol['planned_episodes']:
        print('Figures rendered; final table withheld until all frozen trials finish.')
        return
    tex=[r'\begin{table*}[t]\centering',r'\caption{Task-conditioned navigation: two contexts, two requests and three planners (36 episodes, three sampler seeds per cell). Collision-free arrival counts reaching the destination without sampled proxy overlap. Minimum object gap is the episode minimum to the task-referenced object. It is reported separately from arrival and compared with the 0.8 m preferred clearance in the heading-and-clearance request. Heading RMS is measured relative to the initial corridor heading within 2.5 m of the object. Entries are means over all attempts; mode changes and heading RMS include sample standard deviations.}',r'\label{tab:taskbenchmark}',r'\small',r'\begin{tabular}{@{}llrrrrr@{}}\toprule',r'Task & Planner & Collision-free arrival & Time (s) & Mode changes & Heading RMS ($^\circ$) & Min object gap (m) \\\midrule']
    labels={'nav2':'Nav2 Omni/Ranger','fixed':'Fixed boundaries','hybrid':'Ours'}
    for context in contexts:
        title=names[context]
        tex.append(r'\multicolumn{7}{l}{\textit{'+title+r'}} \\')
        for condition,label in [('ordinary','Ordinary'),('heading_clearance','Heading + clearance')]:
            for method in protocol['methods']:
                cell=[r for r in data['metrics'] if (r['context'],r['condition'],r['method'])==(context,condition,method)]
                assert {r['seed'] for r in cell}==set(protocol['seeds'])
                mean=lambda key:statistics.mean(r[key] for r in cell)
                spread=lambda key:statistics.stdev(r[key] for r in cell)
                tex.append(f"{label} & {labels[method]} & {sum(r['collision_free_arrival'] for r in cell)}/{len(cell)} & {mean('elapsed_s'):.2f} & ${mean('nonzero_mode_changes'):.1f}\\pm{spread('nonzero_mode_changes'):.1f}$ & ${mean('interaction_heading_rms_deg'):.2f}\\pm{spread('interaction_heading_rms_deg'):.2f}$ & {mean('interaction_gap_m'):.3f} "+r'\\')
        tex.append(r'\midrule')
    tex[-1]=r'\bottomrule';tex += [r'\end{tabular}',r'\end{table*}']
    (out/'task_condition_benchmark.tex').write_text('\n'.join(tex)+'\n')
    print('Rendered task contrast, planner comparison and complete experiment table.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('batch',type=Path)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
