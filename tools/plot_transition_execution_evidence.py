"""Measured motion continuity: trajectories, executed modes, paired trials.

Contract: expose the actual mode changes and steering holds underlying the
transition-search comparison. Recorded poses only; no interpolation. The
representative run is preselected seed 37; all five seeds appear in statistics.
"""
import csv
import hashlib
import json
from pathlib import Path
import statistics
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
sys.path.insert(0, str(ROOT/'tools'))
sys.path.insert(0, '/home/USER/.codex/skills/nature-figure/scripts')
from rangermini_dynamic_semantic.grounded_navigation_harness import SpatialGraph
from plot_nav2_matched_cases import corners, clipped_polygon
from audit_panel_alignment import require_matplotlib_panel_alignment

BATCH = ROOT/'artifacts/hybrid_sequence_20261022/offset_portal_domain_isolated_matrix_v6'
OUT = ROOT/'paper/figures'
METHODS = ('primitive', 'fixed', 'hybrid')
NAMES = ('Primitive', 'Fixed timing', 'Optimized timing')
MODE_COLOR = {'DUAL_ACKERMANN':'#ca7a35', 'PARALLEL':'#168c89',
              'SPINNING':'#785b96', 'STOPPED':'#8d949a'}
MODE_Y = {'STOPPED':0, 'DUAL_ACKERMANN':1, 'PARALLEL':2, 'SPINNING':3}


def main():
    sources = [BATCH/'analysis/metrics.json', Path(__file__)]
    metrics = json.loads(sources[0].read_text())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7, 'axes.labelsize':7, 'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'axes.linewidth':.65, 'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig = plt.figure(figsize=(7.16,3.85))
    gs = fig.add_gridspec(2,3,left=.075,right=.985,bottom=.13,top=.88,
                         height_ratios=[1.7,1],wspace=.30,hspace=.72)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]
    lim = (10.65,13.35,-1.95,1.05)
    for j, (method,name) in enumerate(zip(METHODS,NAMES)):
        run = BATCH/f'offset_pose_{method}_s37'
        source = run/'episode/trace.jsonl'
        sources.extend([source,run/'world.sdf'])
        trace = [json.loads(l) for l in source.open()]
        poses = np.array([r['pose'] for r in trace])
        times = np.array([r['elapsed'] for r in trace])
        modes = []
        for r in trace:
            act=r['actuation'];v=act.get('executed')or{}
            moving=np.hypot(v.get('vx',0),v.get('vy',0))>.001 or abs(v.get('wz',0))>.001
            modes.append(act['mode'] if moving else 'STOPPED')
        alignment = [r['actuation'].get('steering_transition_stop',False) for r in trace]
        top, bottom = axes[j],axes[j+3]
        for box in SpatialGraph(run/'world.sdf').static_boxes:
            pts = clipped_polygon(corners(box[:3],*box[3:]),lim)
            if pts:
                top.add_patch(Polygon(pts,fc='#d6d9db',ec='#8d969b',lw=.4))
        segments=np.stack([poses[:-1,:2],poses[1:,:2]],axis=1)
        top.add_collection(LineCollection(segments,colors=[MODE_COLOR.get(m,'#8d949a')for m in modes[:-1]],lw=1.8))
        top.plot(*poses[0,:2],marker='o',ms=3,mfc='white',mec='#34414b',zorder=5)
        top.plot(12.8,.55,marker='*',ms=5,color='#34414b',zorder=5)
        # Footprints at the recorded start and final poses, not a schematic pose.
        for index in (0,len(trace)-1):
            top.add_patch(Polygon(corners(poses[index]),fill=False,ec='#4a555d',lw=.55))
        top.set(xlim=lim[:2],ylim=lim[2:],xlabel='$x$ (m)')
        top.set_aspect('equal',adjustable='box')
        top.set_title(f'({chr(97+j)}) {name}',loc='left',fontsize=8,pad=5)
        if j==0:top.set_ylabel('$y$ (m)')
        for i in range(len(times)-1):
            bottom.plot(times[i:i+2],[MODE_Y[modes[i]]]*2,color=MODE_COLOR[modes[i]],lw=1.6,solid_capstyle='butt')
            if alignment[i]:
                bottom.axvspan(times[i],times[i+1],color='#d1d5d8',alpha=.65,lw=0,zorder=0)
        bottom.set(xlim=(0,25),ylim=(-.35,3.35),xlabel='Elapsed time (s)',xticks=[0,10,20])
        bottom.set_yticks([0,1,2,3],['Stop','Curved','Parallel','Spin'] if j==0 else ['']*4)
        bottom.set_title(f'({chr(100+j)}) Executed modes',loc='left',fontsize=7.5,pad=4)
    handles = [Line2D([0],[0],color=MODE_COLOR[m],lw=1.8,label=label)for m,label in
        [('DUAL_ACKERMANN','Curved'),('PARALLEL','Parallel'),('SPINNING','Spin')]]
    handles.append(Line2D([0],[0],color='#c1c7cb',lw=4,label='Steering hold'))
    fig.legend(handles=handles,ncol=4,loc='upper center',bbox_to_anchor=(.52,1.005),fontsize=7,handlelength=1.4,columnspacing=1.8)
    fig.canvas.draw()
    for top,bottom in zip(axes[:3],axes[3:]):
        rect,old=top.get_position(),bottom.get_position()
        bottom.set_position([rect.x0,old.y0,rect.width,old.height])
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=axes,panel_ids=list('abcdef'),
        row_groups=[list('abc'),list('def')],json_out=OUT/'transition_execution_evidence.alignment.json')
    fig.savefig(OUT/'transition_execution_evidence.pdf')
    fig.savefig(OUT/'transition_execution_evidence.svg')
    fig.savefig(OUT/'transition_execution_evidence.png',dpi=600)
    plt.close(fig)

    # Supplementary/display paired outcomes: seeds are paired units, not a trend axis.
    fig, axs = plt.subplots(1,3,figsize=(7.16,2.05))
    fig.subplots_adjust(left=.075,right=.99,bottom=.23,top=.84,wspace=.35)
    cols=['#92979d','#d58a4d','#167f83']
    for ax,key,label,letter in zip(axs,['elapsed_sim_s','nonzero_mode_changes','sampled_alignment_s'],
            ['Arrival time (s)','Mode changes','Steering hold (s)'],list('abc')):
        for index,seed in enumerate(metrics['protocol']['seeds']):
            y=[next(r[key] for r in metrics['metrics'] if r['seed']==seed and r['method']==m)for m in METHODS]
            offset=(index-2)*.035
            ax.plot(np.arange(3)+offset,y,lw=.6,color='#c3c7ca',zorder=1)
            ax.scatter(np.arange(3)+offset,y,s=13,c=cols,zorder=3,edgecolors='none')
        ax.set_xticks(range(3),['Primitive','Fixed','Optimized'])
        ax.set_title(f'({letter}) {label}',loc='left',fontsize=8)
        ax.set_ylim(bottom=0);ax.set_xlim(-.3,2.3)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,axes=list(axs),panel_ids=list('abc'),
        row_groups=[list('abc')],json_out=OUT/'transition_paired_outcomes.alignment.json')
    fig.savefig(OUT/'transition_paired_outcomes.pdf')
    fig.savefig(OUT/'transition_paired_outcomes.svg')
    fig.savefig(OUT/'transition_paired_outcomes.png',dpi=600)
    plt.close(fig)

    # Mean and sample SD across the five paired trials, not across time samples.
    table=[r'\begin{table*}[t]\centering\footnotesize',
           r'\caption{Motion-sequence optimization over five paired trials. Values are mean $\pm$ sample standard deviation; $d_{\min}$ is the minimum sampled static separation over all five trials. Every configuration completes all five trials.}\label{tab:transitionoutcomes}',
           r'\setlength{\tabcolsep}{6pt}',r'\begin{tabular}{lrrrrr}\toprule',
           r'Configuration & Path length (m) & Arrival time (s) & $d_{\min}$ (m) & Mode changes & Steering hold (s) \\ \midrule']
    summary=[]
    for method,name in zip(METHODS,NAMES):
        rr=[r for r in metrics['metrics'] if r['method']==method]
        stats={key:dict(mean=statistics.mean(r[key]for r in rr),sd=statistics.stdev(r[key]for r in rr))
               for key in ('path_m','elapsed_sim_s','nonzero_mode_changes','sampled_alignment_s')}
        value=lambda key:f"{stats[key]['mean']:.2f} $\\pm$ {stats[key]['sd']:.2f}"
        minimum=min(r['min_static_signed_SAT_m']for r in rr)
        table.append(name+' & '+' & '.join([value('path_m'),value('elapsed_sim_s'),f'{minimum:.3f}',value('nonzero_mode_changes'),value('sampled_alignment_s')])+r' \\')
        summary.append(dict(method=method,n=len(rr),statistics=stats,minimum_static_gap_m=minimum))
    table.extend([r'\bottomrule\end{tabular}\end{table*}'])
    (OUT/'transition_full_statistics.tex').write_text('\n'.join(table)+'\n')
    provenance=dict(claim='Mode continuity and steering holds in measured execution',
        representative_seed=37,all_summary_seeds=metrics['protocol']['seeds'],
        transforms='Measured poses only; no interpolation; states held over next logged interval for mode timeline',
        uncertainty='sample standard deviation over five trials',statistics=summary,
        rollout_budget='cap of 101, not an identical realized count',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources})
    for name in ('transition_execution_evidence','transition_paired_outcomes','transition_full_statistics'):
        (OUT/f'{name}.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Rendered measured mode continuity, paired outcomes, and primary quantitative table.')


if __name__=='__main__':main()
